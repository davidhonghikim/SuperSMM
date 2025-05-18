#!/usr/bin/env python
"""
Fix dataset issues for SuperSMM training.

This script:
1. Generates a primus.txt file from sample_set.txt
2. Creates necessary directories/files in the corpus structure
3. Creates dummy sample data if real data is missing
"""

import os
import sys
import glob
import random
import shutil
from pathlib import Path

def generate_primus_txt():
    """Generate a primus.txt file from sample_set.txt"""
    base_dir = Path(os.path.dirname(os.path.abspath(__file__))).parent
    sample_set_path = base_dir / "sample_data" / "sample_set.txt"
    primus_txt_path = base_dir / "sample_data" / "corpus" / "primus.txt"
    
    print(f"Generating {primus_txt_path} from {sample_set_path}")
    
    # Read sample set
    if not sample_set_path.exists():
        print(f"ERROR: {sample_set_path} does not exist!")
        return False
    
    try:
        with open(sample_set_path, 'r') as f:
            sample_lines = f.read().splitlines()
        
        # Create directory if needed
        primus_txt_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write primus.txt with the same content
        with open(primus_txt_path, 'w') as f:
            for line in sample_lines:
                f.write(f"{line}\n")
        
        print(f"Successfully created {primus_txt_path} with {len(sample_lines)} entries")
        return True
    except Exception as e:
        print(f"ERROR: Failed to create primus.txt: {e}")
        return False

def create_dummy_samples(num_samples=10):
    """Create dummy sample images and annotations if real data is missing"""
    base_dir = Path(os.path.dirname(os.path.abspath(__file__))).parent
    corpus_dir = base_dir / "sample_data" / "corpus" / "primus"
    
    # Ensure the directory exists
    corpus_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Creating {num_samples} dummy sample files in {corpus_dir}")
    
    # Get list of sample IDs from sample_set.txt
    sample_set_path = base_dir / "sample_data" / "sample_set.txt"
    sample_ids = []
    
    if sample_set_path.exists():
        with open(sample_set_path, 'r') as f:
            sample_ids = [line.strip() for line in f.read().splitlines()]
    
    # If we don't have enough sample IDs, generate some
    if len(sample_ids) < num_samples:
        for i in range(num_samples - len(sample_ids)):
            # Generate ID in format like: 000051650-1_1_1
            random_id = f"{random.randint(100000000, 999999999)}-1_1_1"
            sample_ids.append(random_id)
    
    # Use only the first num_samples
    sample_ids = sample_ids[:num_samples]
    
    # Create dummy files for each sample ID
    for sample_id in sample_ids:
        # Create directory for the sample
        sample_dir = corpus_dir / sample_id
        sample_dir.mkdir(exist_ok=True)
        
        # Create dummy PNG file (1x1 pixel)
        png_path = sample_dir / f"{sample_id}.png"
        if not png_path.exists():
            with open(png_path, 'wb') as f:
                # Minimal valid PNG file (1x1 pixel, black)
                f.write(b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDAT\x08\xd7c\xf8\xff\xff?\x00\x05\xfe\x02\xfe\xdc\xccY\xe7\x00\x00\x00\x00IEND\xaeB`\x82')
        
        # Create dummy semantic file
        semantic_path = sample_dir / f"{sample_id}.semantic"
        if not semantic_path.exists():
            with open(semantic_path, 'w') as f:
                f.write("clef-G2 note-C4_quarter note-D4_quarter note-E4_quarter note-F4_quarter barline")
        
        # Create dummy agnostic file
        agnostic_path = sample_dir / f"{sample_id}.agnostic"
        if not agnostic_path.exists():
            with open(agnostic_path, 'w') as f:
                f.write("clef note note note note barline")
    
    # Update the primus.txt file
    primus_txt_path = base_dir / "sample_data" / "corpus" / "primus.txt"
    with open(primus_txt_path, 'w') as f:
        for sample_id in sample_ids:
            f.write(f"{sample_id}\n")
    
    # Update sample_set.txt as well
    with open(sample_set_path, 'w') as f:
        for sample_id in sample_ids:
            f.write(f"{sample_id}\n")
    
    print(f"Successfully created {num_samples} dummy samples")
    return True

def fix_dataset():
    """Fix dataset issues"""
    print("Fixing dataset issues for SuperSMM training...")
    
    # Generate primus.txt
    if not generate_primus_txt():
        # If failed, create dummy samples
        print("Creating dummy samples instead...")
        create_dummy_samples(20)
    
    print("Dataset fix complete!")

if __name__ == "__main__":
    fix_dataset() 