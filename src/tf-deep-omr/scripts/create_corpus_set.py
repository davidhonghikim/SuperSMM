#!/usr/bin/env python
"""
Create corpus set files for training from existing primus dataset directories.

This script scans primus or camera_primus directories and creates:
1. A primus.txt file listing all available samples
2. Set files for training and validation splits
"""

import os
import glob
import random
import argparse
from pathlib import Path

def scan_dataset_dir(directory):
    """
    Scan a dataset directory and return a list of valid sample IDs.
    
    Args:
        directory: Path to primus or camera_primus directory
        
    Returns:
        List of valid sample IDs
    """
    print(f"Scanning dataset directory: {directory}")
    
    sample_ids = []
    
    # Get all subdirectories (each should be a sample)
    sample_dirs = [d for d in Path(directory).glob("*") if d.is_dir()]
    print(f"Found {len(sample_dirs)} potential samples")
    
    # Verify each sample has required files
    valid_samples = 0
    for sample_dir in sample_dirs:
        sample_id = sample_dir.name
        
        # Check for image file (png)
        image_file = sample_dir / f"{sample_id}.png"
        
        # Check for semantic/agnostic annotations
        semantic_file = sample_dir / f"{sample_id}.semantic"
        agnostic_file = sample_dir / f"{sample_id}.agnostic"
        
        if image_file.exists() and (semantic_file.exists() or agnostic_file.exists()):
            sample_ids.append(sample_id)
            valid_samples += 1
    
    print(f"Found {valid_samples} valid samples with images and annotations")
    return sample_ids

def create_corpus_files(dataset_dir, output_dir, validation_split=0.1, seed=42):
    """
    Create corpus set files for the dataset.
    
    Args:
        dataset_dir: Path to dataset directory
        output_dir: Path to output directory
        validation_split: Fraction of samples to use for validation
        seed: Random seed for reproducibility
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Get dataset name
    dataset_name = Path(dataset_dir).name
    
    # Scan dataset for valid samples
    sample_ids = scan_dataset_dir(dataset_dir)
    
    if not sample_ids:
        print(f"Error: No valid samples found in {dataset_dir}")
        return False
    
    # Create primus.txt file in the dataset directory
    primus_file = Path(dataset_dir) / "primus.txt"
    with open(primus_file, 'w') as f:
        for sample_id in sample_ids:
            f.write(f"{sample_id}\n")
    
    print(f"Created primus.txt with {len(sample_ids)} samples at {primus_file}")
    
    # Shuffle and split samples
    random.seed(seed)
    random.shuffle(sample_ids)
    
    validation_count = int(len(sample_ids) * validation_split)
    train_samples = sample_ids[validation_count:]
    validation_samples = sample_ids[:validation_count]
    
    # Create train set file
    train_file = Path(output_dir) / f"{dataset_name}_set.txt"
    with open(train_file, 'w') as f:
        for sample_id in train_samples:
            f.write(f"{sample_id}\n")
    
    print(f"Created training set file with {len(train_samples)} samples at {train_file}")
    
    # Create validation set file
    val_file = Path(output_dir) / f"{dataset_name}_val_set.txt"
    with open(val_file, 'w') as f:
        for sample_id in validation_samples:
            f.write(f"{sample_id}\n")
    
    print(f"Created validation set file with {len(validation_samples)} samples at {val_file}")
    
    return True

def main():
    parser = argparse.ArgumentParser(description='Create corpus set files for primus dataset.')
    parser.add_argument('--dataset', type=str, help='Path to dataset directory')
    parser.add_argument('--output', type=str, help='Path to output directory for set files')
    parser.add_argument('--validation-split', type=float, default=0.1, help='Fraction of samples to use for validation')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    args = parser.parse_args()
    
    # Determine paths
    script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    project_dir = script_dir.parent
    
    # If dataset not specified, look in standard locations
    if not args.dataset:
        # Try primus
        primus_dir = project_dir / "src" / "Data" / "primus"
        if primus_dir.exists() and primus_dir.is_dir():
            args.dataset = str(primus_dir)
        
        # Try camera_primus
        if not args.dataset:
            camera_primus_dir = project_dir / "src" / "Data" / "camera_primus"
            corpus_dir = camera_primus_dir / "Corpus"
            
            if corpus_dir.exists() and corpus_dir.is_dir():
                args.dataset = str(corpus_dir)
            elif camera_primus_dir.exists() and camera_primus_dir.is_dir():
                args.dataset = str(camera_primus_dir)
    
    # If output not specified, use data directory
    if not args.output:
        args.output = str(project_dir / "data")
    
    if not args.dataset:
        print("Error: Could not find dataset directory. Please specify with --dataset.")
        return False
    
    # Create corpus files
    return create_corpus_files(
        args.dataset, 
        args.output, 
        validation_split=args.validation_split,
        seed=args.seed
    )

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 