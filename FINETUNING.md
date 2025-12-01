# Fine-tuning PartField on PartNet Mobility Dataset

This guide explains how to fine-tune the PartField model on the PartNet Mobility dataset for improved part segmentation of articulated objects.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Dataset Preparation](#dataset-preparation)
- [Training](#training)
- [Evaluation](#evaluation)
- [Tips and Best Practices](#tips-and-best-practices)

## Prerequisites

### Environment Setup

Make sure you have set up the PartField environment following the main README instructions:

```bash
conda env create -f environment.yml
conda activate partfield
```

### Download Pretrained Model

Download the pretrained PartField model:

```bash
mkdir -p model
cd model
wget https://huggingface.co/mikaelaangel/partfield-ckpt/resolve/main/model_objaverse.ckpt
cd ..
```

## Dataset Preparation

### Download PartNet Mobility Dataset

The PartNet Mobility dataset can be downloaded from the [official website](https://sapien.ucsd.edu/browse).

```bash
# Create data directory
mkdir -p data/partnet_mobility
cd data/partnet_mobility

# Download and extract the dataset
# Follow instructions from https://sapien.ucsd.edu/browse
# The dataset should be organized as:
# data/partnet_mobility/
#   ├── train/
#   │   ├── <obj_id_1>/
#   │   │   ├── meta.json
#   │   │   ├── mobility.obj
#   │   │   └── semantics.txt
#   │   └── <obj_id_2>/
#   └── val/
```

### Data Structure

Each object directory should contain:
- `meta.json`: Metadata including object category
- `mobility.obj` or `textured_objs/merged.obj`: 3D mesh file
- `semantics.txt` or `result.json`: Part annotations

### Prepare Data (Optional)

For faster training, you can preprocess the data into HDF5 format:

```bash
python prepare_partnet_mobility_data.py \
  --data_path data/partnet_mobility/train \
  --output_path data/partnet_mobility_processed.h5 \
  --split train

python prepare_partnet_mobility_data.py \
  --data_path data/partnet_mobility/val \
  --output_path data/partnet_mobility_processed.h5 \
  --split val
```

## Training

### Basic Training

Train the model with default settings:

```bash
python train_partnet_mobility.py \
  -c configs/final/partnet_mobility.yaml \
  --pretrained model/model_objaverse.ckpt \
  --gpus 1
```

### Multi-GPU Training

For training on multiple GPUs:

```bash
python train_partnet_mobility.py \
  -c configs/final/partnet_mobility.yaml \
  --pretrained model/model_objaverse.ckpt \
  --gpus 4
```

### Resume from Checkpoint

To resume training from a checkpoint:

```bash
python train_partnet_mobility.py \
  -c configs/final/partnet_mobility.yaml \
  --resume checkpoints/partnet_mobility_finetune/last.ckpt \
  --gpus 1
```

### Training on Specific Categories

To fine-tune on specific object categories:

```bash
python train_partnet_mobility.py \
  -c configs/final/partnet_mobility.yaml \
  --pretrained model/model_objaverse.ckpt \
  --opts dataset.categories "['Door', 'Drawer', 'Laptop']" \
  --gpus 1
```

## Configuration

Key configuration parameters in `configs/final/partnet_mobility.yaml`:

### Model Settings
- `triplane_channels_low`: Low-resolution triplane features (default: 128)
- `triplane_channels_high`: High-resolution triplane features (default: 512)
- `triplane_resolution`: Triplane resolution (default: 128)

### Training Hyperparameters
- `optimizer.lr`: Learning rate (default: 1e-4)
- `optimizer.max_epochs`: Maximum training epochs (default: 100)
- `optimizer.weight_decay`: Weight decay for regularization (default: 0.01)

### Dataset Settings
- `dataset.data_path`: Path to PartNet Mobility data
- `dataset.train_batch_size`: Training batch size (default: 4)
- `dataset.categories`: Object categories to include (default: null for all)

### Data Sampling
- `n_point_per_face`: Points sampled per mesh face (default: 2000)
- `n_sample_each`: Total points sampled per mesh (default: 10000)

## Evaluation

After training, extract features from your test set:

```bash
python partfield_inference.py \
  -c configs/final/partnet_mobility.yaml \
  --opts continue_ckpt checkpoints/partnet_mobility_finetune/best.ckpt \
         result_name partfield_features/partnet_test \
         dataset.data_path data/partnet_mobility/test
```

Then run part segmentation:

```bash
python run_part_clustering.py \
  --root exp_results/partfield_features/partnet_test \
  --dump_dir exp_results/clustering/partnet_test \
  --source_dir data/partnet_mobility/test \
  --use_agglo True \
  --max_num_clusters 20 \
  --option 0
```

## Monitoring Training

### TensorBoard

Monitor training progress with TensorBoard:

```bash
tensorboard --logdir logs/partnet_mobility_finetune
```

### Checkpoints

Checkpoints are saved in `checkpoints/partnet_mobility_finetune/`:
- `last.ckpt`: Most recent checkpoint
- `partfield-epoch=XX-val/loss=X.XXXX.ckpt`: Best checkpoints based on validation loss

## Tips and Best Practices

### 1. Learning Rate

Start with a lower learning rate (1e-4 to 1e-5) when fine-tuning from the pretrained model to avoid catastrophic forgetting.

### 2. Batch Size

Adjust batch size based on your GPU memory:
- 4-8 for 24GB GPU (RTX 3090/4090, A5000)
- 2-4 for 16GB GPU (RTX 4000, V100 16GB)
- Use gradient accumulation for larger effective batch sizes

### 3. Data Augmentation

Enable data augmentation in the config for better generalization:
```yaml
augmentation:
  random_rotation: True
  random_scale: True
  random_jitter: True
```

### 4. Category-Specific Fine-tuning

For best results on specific categories, fine-tune separate models:
- Train on category-specific data
- Use longer training (more epochs)
- Adjust number of clusters based on typical part count

### 5. Mesh Preprocessing

If encountering issues with mesh connectivity:
```yaml
preprocess_mesh: True  # Enable mesh cleanup
```

### 6. Part Segmentation Quality

For better segmentation quality:
- Use agglomerative clustering (`--use_agglo True`)
- For fragmented meshes, use KNN-based adjacency (`--with_knn True`)
- Adjust `max_num_clusters` based on expected number of parts

## Troubleshooting

### Out of Memory Errors

1. Reduce batch size in config
2. Reduce `n_sample_each` (number of points sampled)
3. Enable gradient accumulation
4. Use mixed precision training (enabled by default)

### Poor Segmentation Results

1. Train for more epochs
2. Adjust triplet loss margin
3. Enable mesh preprocessing
4. Try different clustering methods
5. Verify part annotations in dataset

### Slow Training

1. Increase number of workers (`train_num_workers`)
2. Preprocess data to HDF5 format
3. Use faster storage (SSD)
4. Enable multi-GPU training

## Citation

If you use this fine-tuning code, please cite both PartField and PartNet Mobility:

```bibtex
@inproceedings{partfield2025,
  title={PartField: Learning 3D Feature Fields for Part Segmentation and Beyond},
  author={Minghua Liu and Mikaela Angelina Uy and Donglai Xiang and Hao Su and Sanja Fidler and Nicholas Sharp and Jun Gao},
  year={2025}
}

@inproceedings{xiang2020sapien,
  title={SAPIEN: A SimulAted Part-based Interactive ENvironment},
  author={Xiang, Fanbo and Qin, Yuzhe and Mo, Kaichun and Xia, Yikuan and Zhu, Hao and Liu, Fangchen and Liu, Minghua and Jiang, Hanxiao and Yuan, Yifu and Wang, He and others},
  booktitle={CVPR},
  year={2020}
}
```

## Support

For questions or issues:
1. Check the main [PartField README](README.md)
2. Review [PartNet Mobility documentation](https://sapien.ucsd.edu/)
3. Open an issue on GitHub
