#!/usr/bin/env python3
"""
Preprocess PartNet Mobility Dataset into HDF5 format for faster training.

Usage:
    python prepare_partnet_mobility_data.py \
        --data_path data/partnet_mobility/train \
        --output_path data/partnet_mobility_processed.h5 \
        --split train
"""

import argparse
import h5py
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys

# Add partfield to path
sys.path.append('.')
from partfield.partnet_mobility_dataset import PartNetMobilityDataset


def main():
    parser = argparse.ArgumentParser(
        description='Preprocess PartNet Mobility data to HDF5'
    )
    parser.add_argument(
        '--data_path',
        type=str,
        required=True,
        help='Path to PartNet Mobility dataset directory'
    )
    parser.add_argument(
        '--output_path',
        type=str,
        required=True,
        help='Output HDF5 file path'
    )
    parser.add_argument(
        '--split',
        type=str,
        default='train',
        choices=['train', 'val', 'test'],
        help='Dataset split'
    )
    parser.add_argument(
        '--n_points',
        type=int,
        default=10000,
        help='Number of points to sample per mesh'
    )
    parser.add_argument(
        '--categories',
        type=str,
        nargs='+',
        default=None,
        help='Object categories to include'
    )
    
    args = parser.parse_args()
    
    print(f"Loading PartNet Mobility dataset from {args.data_path}")
    print(f"Split: {args.split}")
    print(f"Points per mesh: {args.n_points}")
    
    # Create dataset
    dataset = PartNetMobilityDataset(
        data_path=args.data_path,
        n_sample_each=args.n_points,
        split=args.split,
        categories=args.categories
    )
    
    print(f"Found {len(dataset)} objects")
    
    # Create output directory if it doesn't exist
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create or open HDF5 file
    with h5py.File(output_path, 'a') as h5f:
        # Create split group if it doesn't exist
        if args.split in h5f:
            print(f"Warning: Split '{args.split}' already exists. Overwriting...")
            del h5f[args.split]
            
        split_group = h5f.create_group(args.split)
        
        # Process each object
        for idx in tqdm(range(len(dataset)), desc=f"Processing {args.split}"):
            try:
                sample = dataset[idx]
                
                # Create group for this sample
                sample_group = split_group.create_group(str(idx))
                
                # Save data
                sample_group.create_dataset(
                    'points',
                    data=sample['points'].numpy(),
                    compression='gzip'
                )
                sample_group.create_dataset(
                    'part_labels',
                    data=sample['part_labels'].numpy(),
                    compression='gzip'
                )
                
                # Save metadata as attributes
                sample_group.attrs['obj_id'] = sample['obj_id']
                
            except Exception as e:
                print(f"\nError processing sample {idx}: {e}")
                continue
                
    print(f"\nPreprocessing complete!")
    print(f"Output saved to: {output_path}")
    print(f"Total samples: {len(dataset)}")


if __name__ == '__main__':
    main()
