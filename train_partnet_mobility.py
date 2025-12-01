#!/usr/bin/env python3
"""
Fine-tuning script for PartField on PartNet Mobility Dataset

This script allows fine-tuning the PartField model on the PartNet Mobility dataset
for articulated object part segmentation.

Usage:
    python train_partnet_mobility.py -c configs/final/partnet_mobility.yaml
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
import yaml
from yacs.config import CfgNode
from loguru import logger

from partfield.dataloader import PartNetMobilityDataset
from partfield.model.model_trainer_pvcnn_only_demo import PartFieldModel


class PartNetMobilityDataModule(L.LightningDataModule):
    """Lightning DataModule for PartNet Mobility dataset."""
    
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.batch_size = cfg.dataset.train_batch_size
        self.num_workers = cfg.dataset.train_num_workers
        self.data_path = cfg.dataset.data_path
        
    def setup(self, stage=None):
        """Setup train/val/test datasets."""
        if stage == 'fit' or stage is None:
            self.train_dataset = PartNetMobilityDataset(
                data_path=os.path.join(self.data_path, 'train'),
                n_point_per_face=self.cfg.n_point_per_face,
                n_sample_each=self.cfg.n_sample_each,
                split='train'
            )
            self.val_dataset = PartNetMobilityDataset(
                data_path=os.path.join(self.data_path, 'val'),
                n_point_per_face=self.cfg.n_point_per_face,
                n_sample_each=self.cfg.n_sample_each,
                split='val'
            )
            
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True
        )
        
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.cfg.dataset.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )


class PartFieldFineTuner(L.LightningModule):
    """Lightning Module for fine-tuning PartField on PartNet Mobility."""
    
    def __init__(self, cfg, pretrained_ckpt=None):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters()
        
        # Initialize model
        self.model = PartFieldModel(cfg)
        
        # Load pretrained weights if provided
        if pretrained_ckpt is not None:
            logger.info(f"Loading pretrained checkpoint from {pretrained_ckpt}")
            checkpoint = torch.load(pretrained_ckpt, map_location='cpu')
            if 'state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['state_dict'], strict=False)
            else:
                self.model.load_state_dict(checkpoint, strict=False)
                
        # Loss function
        self.criterion = nn.TripletMarginLoss(
            margin=cfg.loss.triplet,
            p=2
        )
        
    def forward(self, batch):
        return self.model(batch)
        
    def training_step(self, batch, batch_idx):
        features = self(batch)
        
        # Compute triplet loss with part labels
        loss = self.compute_part_loss(features, batch['part_labels'])
        
        self.log('train/loss', loss, prog_bar=True)
        self.log('train/lr', self.optimizers().param_groups[0]['lr'])
        
        return loss
        
    def validation_step(self, batch, batch_idx):
        features = self(batch)
        loss = self.compute_part_loss(features, batch['part_labels'])
        
        self.log('val/loss', loss, prog_bar=True, sync_dist=True)
        
        return loss
        
    def compute_part_loss(self, features, part_labels):
        """Compute contrastive loss for part segmentation.
        
        Args:
            features: Predicted feature field [B, N, D]
            part_labels: Ground truth part labels [B, N]
        """
        B, N, D = features.shape
        
        # Sample triplets: anchor, positive, negative
        loss = 0.0
        num_triplets = 0
        
        for b in range(B):
            feat = features[b]  # [N, D]
            labels = part_labels[b]  # [N]
            
            unique_parts = torch.unique(labels)
            
            for part_id in unique_parts:
                # Get points belonging to this part
                part_mask = (labels == part_id)
                if part_mask.sum() < 2:
                    continue
                    
                part_indices = torch.where(part_mask)[0]
                non_part_indices = torch.where(~part_mask)[0]
                
                if len(non_part_indices) == 0:
                    continue
                
                # Sample triplets
                n_samples = min(len(part_indices), 100)
                for _ in range(n_samples):
                    if len(part_indices) >= 2:
                        # Random anchor and positive from same part
                        idx = torch.randperm(len(part_indices))[:2]
                        anchor = feat[part_indices[idx[0]]]
                        positive = feat[part_indices[idx[1]]]
                        
                        # Random negative from different part
                        neg_idx = torch.randint(0, len(non_part_indices), (1,))
                        negative = feat[non_part_indices[neg_idx]]
                        
                        loss += self.criterion(
                            anchor.unsqueeze(0),
                            positive.unsqueeze(0),
                            negative.unsqueeze(0)
                        )
                        num_triplets += 1
                        
        if num_triplets > 0:
            loss = loss / num_triplets
        else:
            loss = torch.tensor(0.0, device=features.device)
            
        return loss
        
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.cfg.optimizer.lr,
            weight_decay=self.cfg.optimizer.weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.cfg.optimizer.max_epochs,
            eta_min=self.cfg.optimizer.lr_min
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch'
            }
        }


def load_config(config_path, opts=None):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        cfg = CfgNode(yaml.safe_load(f))
    
    if opts:
        cfg.merge_from_list(opts)
        
    return cfg


def main():
    parser = argparse.ArgumentParser(description='Fine-tune PartField on PartNet Mobility')
    parser.add_argument('-c', '--config', type=str, required=True,
                       help='Path to config file')
    parser.add_argument('--opts', nargs='+', default=[],
                       help='Modify config options')
    parser.add_argument('--pretrained', type=str, default='model/model_objaverse.ckpt',
                       help='Path to pretrained checkpoint')
    parser.add_argument('--gpus', type=int, default=1,
                       help='Number of GPUs to use')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from checkpoint')
    
    args = parser.parse_args()
    
    # Load config
    cfg = load_config(args.config, args.opts)
    
    # Setup logger
    logger.info(f"Configuration:\n{cfg}")
    
    # Create data module
    data_module = PartNetMobilityDataModule(cfg)
    
    # Create model
    if args.resume:
        model = PartFieldFineTuner.load_from_checkpoint(
            args.resume,
            cfg=cfg,
            pretrained_ckpt=None
        )
    else:
        model = PartFieldFineTuner(cfg, pretrained_ckpt=args.pretrained)
    
    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"checkpoints/{cfg.result_name}",
        filename='partfield-{epoch:02d}-{val/loss:.4f}',
        monitor='val/loss',
        mode='min',
        save_top_k=3,
        save_last=True
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    
    # Setup logger
    tb_logger = TensorBoardLogger(
        save_dir='logs',
        name=cfg.result_name
    )
    
    # Create trainer
    trainer = L.Trainer(
        max_epochs=cfg.optimizer.max_epochs,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=args.gpus,
        callbacks=[checkpoint_callback, lr_monitor],
        logger=tb_logger,
        precision='16-mixed',
        gradient_clip_val=1.0,
        accumulate_grad_batches=cfg.get('accumulate_grad_batches', 1),
        log_every_n_steps=10
    )
    
    # Train
    logger.info("Starting training...")
    trainer.fit(model, data_module, ckpt_path=args.resume)
    
    logger.info("Training completed!")
    logger.info(f"Best checkpoint: {checkpoint_callback.best_model_path}")


if __name__ == '__main__':
    main()
