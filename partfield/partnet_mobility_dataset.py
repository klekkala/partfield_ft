"""PartNet Mobility Dataset Loader for PartField Fine-tuning."""

import os
import json
import torch
import trimesh
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
import h5py


class PartNetMobilityDataset(Dataset):
    """Dataset class for PartNet Mobility articulated object dataset.
    
    PartNet Mobility contains 3D articulated objects with semantic part annotations
    and motion parameters. This loader prepares data for part segmentation training.
    
    Args:
        data_path: Path to PartNet Mobility dataset directory
        n_point_per_face: Number of points to sample per mesh face
        n_sample_each: Total number of points to sample from each mesh
        split: Dataset split ('train', 'val', or 'test')
        categories: List of object categories to include (None for all)
    """
    
    def __init__(
        self,
        data_path,
        n_point_per_face=1000,
        n_sample_each=10000,
        split='train',
        categories=None
    ):
        super().__init__()
        self.data_path = Path(data_path)
        self.n_point_per_face = n_point_per_face
        self.n_sample_each = n_sample_each
        self.split = split
        self.categories = categories
        
        # Load metadata
        self.samples = self._load_samples()
        
    def _load_samples(self):
        """Load list of valid samples from dataset."""
        samples = []
        
        # Iterate through dataset directory
        if not self.data_path.exists():
            raise ValueError(f"Dataset path {self.data_path} does not exist")
            
        for obj_dir in self.data_path.iterdir():
            if not obj_dir.is_dir():
                continue
                
            # Check if metadata exists
            meta_file = obj_dir / 'meta.json'
            if not meta_file.exists():
                continue
                
            # Load metadata
            with open(meta_file, 'r') as f:
                metadata = json.load(f)
                
            # Filter by category if specified
            if self.categories is not None:
                if metadata.get('model_cat') not in self.categories:
                    continue
                    
            # Check if mesh and annotation files exist
            mesh_file = obj_dir / 'textured_objs' / 'merged.obj'
            if not mesh_file.exists():
                # Try alternative mesh location
                mesh_file = obj_dir / 'mobility.obj'
                if not mesh_file.exists():
                    continue
                    
            samples.append({
                'obj_id': obj_dir.name,
                'obj_dir': obj_dir,
                'mesh_file': mesh_file,
                'metadata': metadata
            })
            
        return samples
        
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load mesh
        mesh = trimesh.load(sample['mesh_file'], force='mesh')
        
        # Load part annotations
        part_labels = self._load_part_labels(sample)
        
        # Sample points from mesh
        points, face_indices = self._sample_points_from_mesh(
            mesh, 
            n_samples=self.n_sample_each
        )
        
        # Get part labels for sampled points
        point_part_labels = part_labels[face_indices]
        
        # Normalize points to unit cube
        points_centered = points - points.mean(axis=0)
        max_dist = np.abs(points_centered).max()
        points_normalized = points_centered / (max_dist + 1e-8)
        
        return {
            'points': torch.from_numpy(points_normalized).float(),
            'part_labels': torch.from_numpy(point_part_labels).long(),
            'obj_id': sample['obj_id'],
            'mesh_faces': torch.from_numpy(np.asarray(mesh.faces)).long(),
            'face_indices': torch.from_numpy(face_indices).long()
        }
        
    def _load_part_labels(self, sample):
        """Load part labels for each face in the mesh."""
        obj_dir = sample['obj_dir']
        
        # Try to load from semantics file
        semantics_file = obj_dir / 'semantics.txt'
        if semantics_file.exists():
            with open(semantics_file, 'r') as f:
                lines = f.readlines()
            
            part_labels = []
            for line in lines:
                if line.strip():
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        part_labels.append(int(parts[1]))
                        
            return np.array(part_labels)
            
        # Try alternative annotation format
        result_file = obj_dir / 'result.json'
        if result_file.exists():
            with open(result_file, 'r') as f:
                result = json.load(f)
                
            # Extract part labels from result
            part_labels = self._extract_labels_from_result(result)
            return part_labels
            
        # If no annotations found, create dummy labels
        mesh = trimesh.load(sample['mesh_file'], force='mesh')
        return np.zeros(len(mesh.faces), dtype=np.int64)
        
    def _extract_labels_from_result(self, result):
        """Extract part labels from result.json structure."""
        labels = []
        
        for part_idx, part_info in enumerate(result):
            if isinstance(part_info, dict) and 'objs' in part_info:
                n_faces = len(part_info['objs'])
                labels.extend([part_idx] * n_faces)
                
        return np.array(labels)
        
    def _sample_points_from_mesh(self, mesh, n_samples):
        """Sample points uniformly from mesh surface.
        
        Returns:
            points: [N, 3] sampled point coordinates
            face_indices: [N] indices of faces each point belongs to
        """
        # Sample points with face indices
        points, face_indices = trimesh.sample.sample_surface(
            mesh, 
            n_samples
        )
        
        return points, face_indices
        
    @staticmethod
    def collate_fn(batch):
        """Custom collate function for batching variable-size samples."""
        # Stack tensors that have same size
        points = torch.stack([item['points'] for item in batch])
        part_labels = torch.stack([item['part_labels'] for item in batch])
        
        return {
            'points': points,
            'part_labels': part_labels,
            'obj_ids': [item['obj_id'] for item in batch],
            'mesh_faces': [item['mesh_faces'] for item in batch],
            'face_indices': [item['face_indices'] for item in batch]
        }


class PartNetMobilityH5Dataset(Dataset):
    """Alternative loader using preprocessed HDF5 files for faster loading."""
    
    def __init__(self, h5_path, split='train'):
        super().__init__()
        self.h5_path = Path(h5_path)
        self.split = split
        
        # Load data from HDF5
        with h5py.File(self.h5_path, 'r') as f:
            split_group = f[split]
            self.n_samples = len(split_group.keys())
            
    def __len__(self):
        return self.n_samples
        
    def __getitem__(self, idx):
        with h5py.File(self.h5_path, 'r') as f:
            sample = f[self.split][str(idx)]
            
            points = torch.from_numpy(sample['points'][...]).float()
            part_labels = torch.from_numpy(sample['part_labels'][...]).long()
            obj_id = sample.attrs['obj_id']
            
        return {
            'points': points,
            'part_labels': part_labels,
            'obj_id': obj_id
        }
