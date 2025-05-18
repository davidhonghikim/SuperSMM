#!/usr/bin/env python3
"""
Fix sample data by ensuring all semantic and agnostic files exist.
"""

import os
import sys
import glob
import shutil

def fix_sample_data(sample_data_dir):
    """
    Ensure all images in the sample data directory have corresponding semantic and agnostic files.
    
    Args:
        sample_data_dir: Path to the sample data directory
    """
    print(f"Fixing sample data in {sample_data_dir}")
    
    # Find all PNG images in the sample data directory
    image_files = glob.glob(os.path.join(sample_data_dir, "**/*.png"), recursive=True)
    print(f"Found {len(image_files)} image files")
    
    fixed_count = 0
    for img_path in image_files:
        base_path = img_path[:-4]  # Remove .png extension
        
        # Check and create semantic file if needed
        semantic_path = base_path + ".semantic"
        if not os.path.exists(semantic_path):
            with open(semantic_path, 'w') as f:
                f.write("clef.G-2 note.quarter-2")
            fixed_count += 1
            print(f"Created missing semantic file: {os.path.basename(semantic_path)}")
        
        # Check and create agnostic file if needed
        agnostic_path = base_path + ".agnostic"
        if not os.path.exists(agnostic_path):
            with open(agnostic_path, 'w') as f:
                f.write("clef.G-2 note.quarter-2")
            fixed_count += 1
            print(f"Created missing agnostic file: {os.path.basename(agnostic_path)}")
    
    print(f"Fixed {fixed_count} missing files")

def fix_sample_set_file(sample_set_file, sample_data_dir):
    """
    Fix the sample set file to ensure all referenced images exist and have semantic files.
    
    Args:
        sample_set_file: Path to the sample set file
        sample_data_dir: Path to the sample data directory
    """
    print(f"Fixing sample set file: {sample_set_file}")
    
    # Read the sample set file
    with open(sample_set_file, 'r') as f:
        lines = f.readlines()
    
    # Check each line and ensure the referenced image exists
    valid_lines = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 1:
            img_path = parts[0]
            full_path = os.path.join(sample_data_dir, img_path)
            
            if os.path.isfile(full_path):
                # Check if semantic file exists
                base_path = full_path[:-4] if full_path.endswith('.png') else full_path
                semantic_path = base_path + ".semantic"
                
                if os.path.exists(semantic_path):
                    valid_lines.append(line)
                else:
                    print(f"Missing semantic file for {img_path}, creating it")
                    with open(semantic_path, 'w') as f:
                        f.write("clef.G-2 note.quarter-2")
                    valid_lines.append(line)
            else:
                print(f"Image file not found: {full_path}")
    
    # Write the fixed sample set file
    with open(sample_set_file, 'w') as f:
        f.writelines(valid_lines)
    
    print(f"Fixed sample set file now contains {len(valid_lines)} entries")

def main():
    if len(sys.argv) < 2:
        print("Usage: python fix_sample_data.py <sample_data_dir> [sample_set_file]")
        sys.exit(1)
    
    sample_data_dir = sys.argv[1]
    
    if not os.path.isdir(sample_data_dir):
        print(f"Error: Sample data directory {sample_data_dir} does not exist")
        sys.exit(1)
    
    fix_sample_data(sample_data_dir)
    
    if len(sys.argv) >= 3:
        sample_set_file = sys.argv[2]
        if os.path.isfile(sample_set_file):
            fix_sample_set_file(sample_set_file, sample_data_dir)
        else:
            print(f"Error: Sample set file {sample_set_file} does not exist")
            sys.exit(1)

if __name__ == "__main__":
    main()
