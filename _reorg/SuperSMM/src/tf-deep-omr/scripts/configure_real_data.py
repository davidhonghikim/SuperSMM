#!/usr/bin/env python
"""
Configure the training system to use real data instead of sample data.

This script:
1. Creates a primus.txt file from the real primus dataset
2. Creates a proper set file for training
3. Updates configuration to point to the real data
"""

import os
import sys
import glob
import random
from pathlib import Path

def scan_real_data():
    """Scan the real primus data directory and collect available datasets"""
    base_dir = Path(os.path.dirname(os.path.abspath(__file__))).parent
    
    # Check both potential dataset locations
    primus_dir = base_dir / "src" / "Data" / "primus"
    camera_primus_dir = base_dir / "src" / "Data" / "camera_primus"
    
    available_datasets = []
    
    # Check standard primus directory
    if primus_dir.exists() and primus_dir.is_dir():
        print(f"Found primus dataset at: {primus_dir}")
        
        # Find all subdirectories - each one should be a sample
        sample_dirs = [d for d in primus_dir.glob("*") if d.is_dir()]
        print(f"Found {len(sample_dirs)} samples in primus dataset")
        
        # Check a few samples to make sure they have the required files
        valid_samples = 0
        for sample_dir in sample_dirs:
            sample_id = sample_dir.name
            if (sample_dir / f"{sample_id}.png").exists() and (
                (sample_dir / f"{sample_id}.semantic").exists() or 
                (sample_dir / f"{sample_id}.agnostic").exists()):
                valid_samples += 1
        
        print(f"Found {valid_samples} valid samples in primus dataset")
        
        if valid_samples > 0:
            available_datasets.append({
                "name": "primus",
                "path": str(primus_dir),
                "samples": [d.name for d in sample_dirs if (d / f"{d.name}.png").exists()],
                "count": valid_samples
            })
    
    # Check camera primus directory
    if camera_primus_dir.exists() and camera_primus_dir.is_dir():
        print(f"Found camera_primus dataset at: {camera_primus_dir}")
        
        # Find all subdirectories - each one should be a sample
        sample_dirs = [d for d in camera_primus_dir.glob("*") if d.is_dir()]
        print(f"Found {len(sample_dirs)} samples in camera_primus dataset")
        
        # Check a few samples to make sure they have the required files
        valid_samples = 0
        corpus_dir = camera_primus_dir / "Corpus" if (camera_primus_dir / "Corpus").exists() else camera_primus_dir
        sample_dirs = [d for d in corpus_dir.glob("*") if d.is_dir()]
        
        for sample_dir in sample_dirs:
            sample_id = sample_dir.name
            if (sample_dir / f"{sample_id}.png").exists() and (
                (sample_dir / f"{sample_id}.semantic").exists() or 
                (sample_dir / f"{sample_id}.agnostic").exists()):
                valid_samples += 1
        
        print(f"Found {valid_samples} valid samples in camera_primus dataset")
        
        if valid_samples > 0:
            available_datasets.append({
                "name": "camera_primus",
                "path": str(corpus_dir),
                "samples": [d.name for d in sample_dirs if (d / f"{d.name}.png").exists()],
                "count": valid_samples
            })
    
    return available_datasets

def create_set_file(dataset):
    """Create a set file for the given dataset"""
    base_dir = Path(os.path.dirname(os.path.abspath(__file__))).parent
    
    # Create the data directory if it doesn't exist
    data_dir = base_dir / "data"
    data_dir.mkdir(exist_ok=True)
    
    # Create the set file
    set_file_path = data_dir / f"{dataset['name']}_set.txt"
    
    print(f"Creating set file: {set_file_path}")
    
    with open(set_file_path, 'w') as f:
        for sample_id in dataset['samples']:
            f.write(f"{sample_id}\n")
    
    print(f"Created set file with {len(dataset['samples'])} samples")
    
    return set_file_path

def create_primus_txt(dataset):
    """Create a primus.txt file for the dataset"""
    base_dir = Path(os.path.dirname(os.path.abspath(__file__))).parent
    dataset_dir = Path(dataset['path'])
    primus_txt_path = dataset_dir / "primus.txt"
    
    print(f"Creating primus.txt file: {primus_txt_path}")
    
    with open(primus_txt_path, 'w') as f:
        for sample_id in dataset['samples']:
            f.write(f"{sample_id}\n")
    
    print(f"Created primus.txt file with {len(dataset['samples'])} samples")
    
    return primus_txt_path

def create_training_script(dataset, set_file_path):
    """Create a training script for the given dataset"""
    base_dir = Path(os.path.dirname(os.path.abspath(__file__))).parent
    
    # Create the script
    script_path = base_dir / "scripts" / f"train_{dataset['name']}.sh"
    
    print(f"Creating training script: {script_path}")
    
    with open(script_path, 'w') as f:
        f.write("#!/bin/bash\n\n")
        f.write("# Training script for real data using the primus dataset\n\n")
        f.write(f"# Configuration\n")
        f.write(f"CORPUS=\"{dataset['path']}\"\n")
        f.write(f"SET=\"{set_file_path}\"\n")
        f.write(f"VOCABULARY=\"{base_dir}/data/vocabulary_semantic.txt\"\n")
        f.write(f"MODEL_DIR=\"{base_dir}/model/{dataset['name']}_model\"\n")
        f.write(f"TOTAL_EPOCHS=500\n")
        f.write(f"BATCH_SIZE=10\n\n")
        
        f.write("# Allow memory growth instead of pre-allocating all GPU memory\n")
        f.write("export TF_FORCE_GPU_ALLOW_GROWTH=true\n\n")
        
        f.write("# Create model directory if it doesn't exist\n")
        f.write('mkdir -p "$MODEL_DIR"\n')
        f.write('mkdir -p "$MODEL_DIR/logs"\n\n')
        
        f.write("echo \"=================================================================\"\n")
        f.write(f"echo \"Starting OMR model training with {dataset['name']} dataset\"\n")
        f.write("echo \"==================================================================\"\n")
        f.write("echo \"Dataset: $CORPUS\"\n")
        f.write("echo \"Set file: $SET\"\n")
        f.write("echo \"Vocabulary: $VOCABULARY\"\n")
        f.write("echo \"Model directory: $MODEL_DIR\"\n")
        f.write("echo \"Total epochs: $TOTAL_EPOCHS\"\n")
        f.write("echo \"==================================================================\"\n\n")
        
        f.write("# Run the training\n")
        f.write("cd \"$(dirname \"$0\")/..\" || exit\n")
        f.write("python src/ctc_training.py \\\n")
        f.write("  -corpus \"$CORPUS\" \\\n")
        f.write("  -set \"$SET\" \\\n")
        f.write("  -save_model \"$MODEL_DIR/model\" \\\n")
        f.write("  -vocabulary \"$VOCABULARY\" \\\n")
        f.write("  -semantic 2>&1 | tee -a \"$MODEL_DIR/logs/training_output.log\"\n\n")
        
        f.write("echo \"==================================================================\"\n")
        f.write("echo \"Training complete!\"\n")
        f.write("echo \"==================================================================\"\n")
    
    # Make the script executable
    os.chmod(script_path, 0o755)
    
    print(f"Created training script: {script_path}")
    
    return script_path

def create_continue_script(dataset, set_file_path):
    """Create a script to continue training from checkpoints"""
    base_dir = Path(os.path.dirname(os.path.abspath(__file__))).parent
    
    # Create the script
    script_path = base_dir / "scripts" / f"continue_{dataset['name']}_training.sh"
    
    print(f"Creating continue training script: {script_path}")
    
    with open(script_path, 'w') as f:
        f.write("#!/bin/bash\n\n")
        f.write("# Script to continue training from the latest checkpoint\n\n")
        f.write(f"# Configuration\n")
        f.write(f"CORPUS=\"{dataset['path']}\"\n")
        f.write(f"SET=\"{set_file_path}\"\n")
        f.write(f"VOCABULARY=\"{base_dir}/data/vocabulary_semantic.txt\"\n")
        f.write(f"MODEL_DIR=\"{base_dir}/model/{dataset['name']}_model\"\n")
        f.write(f"TOTAL_EPOCHS=500\n")
        f.write(f"BATCH_SIZE=10\n\n")
        
        f.write("# Allow memory growth instead of pre-allocating all GPU memory\n")
        f.write("export TF_FORCE_GPU_ALLOW_GROWTH=true\n\n")
        
        f.write("echo \"=================================================================\"\n")
        f.write(f"echo \"Resuming OMR model training with {dataset['name']} dataset\"\n")
        f.write("echo \"==================================================================\"\n")
        f.write("echo \"Dataset: $CORPUS\"\n")
        f.write("echo \"Set file: $SET\"\n")
        f.write("echo \"Model directory: $MODEL_DIR\"\n")
        f.write("echo \"==================================================================\"\n\n")
        
        f.write("# Run the training with resume flag\n")
        f.write("cd \"$(dirname \"$0\")/..\" || exit\n")
        f.write("python src/ctc_training.py \\\n")
        f.write("  -corpus \"$CORPUS\" \\\n")
        f.write("  -set \"$SET\" \\\n")
        f.write("  -save_model \"$MODEL_DIR/model\" \\\n")
        f.write("  -vocabulary \"$VOCABULARY\" \\\n")
        f.write("  -semantic \\\n")
        f.write("  -resume 2>&1 | tee -a \"$MODEL_DIR/logs/training_output.log\"\n\n")
        
        f.write("echo \"==================================================================\"\n")
        f.write("echo \"Training complete!\"\n")
        f.write("echo \"==================================================================\"\n")
    
    # Make the script executable
    os.chmod(script_path, 0o755)
    
    print(f"Created continue training script: {script_path}")
    
    return script_path

def configure_real_data():
    """Configure the training system to use real data"""
    print("Configuring training system to use real data...")
    
    # Scan for available datasets
    datasets = scan_real_data()
    
    if not datasets:
        print("ERROR: No valid datasets found!")
        print("Please ensure that at least one of these directories exists and contains data:")
        print("  - /Users/danger/CascadeProjects/LOO/SuperSMM/src/tf-deep-omr/src/Data/primus")
        print("  - /Users/danger/CascadeProjects/LOO/SuperSMM/src/tf-deep-omr/src/Data/camera_primus")
        return False
    
    # Process each dataset
    for dataset in datasets:
        print(f"\nConfiguring dataset: {dataset['name']} ({dataset['count']} samples)")
        
        # Create the set file
        set_file_path = create_set_file(dataset)
        
        # Create the primus.txt file
        primus_txt_path = create_primus_txt(dataset)
        
        # Create the training script
        train_script_path = create_training_script(dataset, set_file_path)
        
        # Create the continue script
        continue_script_path = create_continue_script(dataset, set_file_path)
        
        print(f"\nDataset {dataset['name']} configured successfully!")
        print(f"To start training, run: {train_script_path}")
        print(f"To continue training later, run: {continue_script_path}")
    
    print("\nAll datasets configured successfully!")
    return True

if __name__ == "__main__":
    configure_real_data() 