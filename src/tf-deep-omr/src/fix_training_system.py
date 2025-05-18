#!/usr/bin/env python3
"""
Comprehensive fix for the OMR training system
This script fixes path handling issues and ensures the sample dataset is created correctly
"""

import os
import shutil
import argparse
import glob
import sys

def fix_primus_paths():
    """Fix path handling in primus.py"""
    print("Fixing path handling in primus.py...")
    
    # Read the current file
    with open('primus.py', 'r') as f:
        content = f.read()
    
    # Fix path construction in nextBatch method
    content = content.replace(
        "sample_fullpath = os.path.join(self.corpus_dirpath, sample_filepath, sample_filepath)",
        """# Fix path construction to avoid duplication
                    # The sample_filepath already contains the full relative path
                    if sample_filepath.endswith('.png'):
                        image_path = os.path.join(self.corpus_dirpath, sample_filepath)
                    else:
                        # For cases where the extension is missing
                        if self.distortions:
                            image_path = os.path.join(self.corpus_dirpath, sample_filepath) + '_distorted.jpg'
                        else:
                            image_path = os.path.join(self.corpus_dirpath, sample_filepath) + '.png'"""
    )
    
    # Remove the old image path construction
    content = content.replace(
        """                    # IMAGE
                    if self.distortions:
                        image_path = sample_fullpath + '_distorted.jpg'
                    else:
                        image_path = sample_fullpath + '.png'""", 
        ""
    )
    
    # Fix ground truth path construction in nextBatch
    content = content.replace(
        """                    # Load ground truth
                    if self.semantic:
                        gt_path = sample_fullpath + '.semantic'
                    else:
                        gt_path = sample_fullpath + '.agnostic'""",
        """                    # Load ground truth
                    # Extract the base path without extension
                    if sample_filepath.endswith('.png'):
                        base_path = os.path.join(self.corpus_dirpath, sample_filepath[:-4])
                    else:
                        base_path = os.path.join(self.corpus_dirpath, sample_filepath)
                    
                    if self.semantic:
                        gt_path = base_path + '.semantic'
                    else:
                        gt_path = base_path + '.agnostic'"""
    )
    
    # Fix path construction in getValidation method
    content = content.replace(
        "sample_fullpath = self.corpus_dirpath + '/' + sample_filepath + '/' + sample_filepath",
        """# Fix path construction to avoid duplication
                # The sample_filepath already contains the full relative path
                if sample_filepath.endswith('.png'):
                    image_path = os.path.join(self.corpus_dirpath, sample_filepath)
                else:
                    image_path = os.path.join(self.corpus_dirpath, sample_filepath) + '.png'"""
    )
    
    # Remove the old image path construction in getValidation
    content = content.replace(
        """                # IMAGE
                image_path = sample_fullpath + '.png'""",
        ""
    )
    
    # Fix ground truth path construction in getValidation
    content = content.replace(
        """                # GROUND TRUTH
                if self.semantic:
                    sample_full_filepath = sample_fullpath + '.semantic'
                else:
                    sample_full_filepath = sample_fullpath + '.agnostic'""",
        """                # GROUND TRUTH
                # Extract the base path without extension
                if sample_filepath.endswith('.png'):
                    base_path = os.path.join(self.corpus_dirpath, sample_filepath[:-4])
                else:
                    base_path = os.path.join(self.corpus_dirpath, sample_filepath)
                
                if self.semantic:
                    sample_full_filepath = base_path + '.semantic'
                else:
                    sample_full_filepath = base_path + '.agnostic'"""
    )
    
    # Write the fixed content back
    with open('primus.py', 'w') as f:
        f.write(content)
    
    print("Fixed path handling in primus.py")

def fix_prepare_corpus():
    """Fix the prepare_corpus.py script to handle paths correctly"""
    print("Fixing prepare_corpus.py...")
    
    # Read the current file
    with open('prepare_corpus.py', 'r') as f:
        content = f.read()
    
    # Fix the create_sample_corpus function to handle paths correctly
    content = content.replace(
        """    # Create sample corpus directory
    os.makedirs(os.path.join(output_dir, 'corpus'), exist_ok=True)
    
    # Create sample set file
    sample_set_file = os.path.join(output_dir, 'sample_set.txt')
    with open(sample_set_file, 'w') as f:
        for img_path in sample_paths:
            f.write(img_path + '\\n')
            
            # Extract directory and filename from the path
            img_dir = os.path.dirname(img_path)
            img_name = os.path.basename(img_path)
            
            # Create directory in sample corpus
            sample_img_dir = os.path.join(output_dir, 'corpus', img_dir)
            os.makedirs(sample_img_dir, exist_ok=True)
            
            # Copy image to sample corpus
            src_img_path = os.path.join(corpus_path, img_dir, img_name)
            dst_img_path = os.path.join(sample_img_dir, img_name)
            
            # Check if source image exists
            if not os.path.exists(src_img_path):
                print(f"Warning: Source image not found: {src_img_path}")
                continue
                
            shutil.copy2(src_img_path, dst_img_path)
            
            # Copy ground truth files if they exist
            for ext in ['.semantic', '.agnostic']:
                src_gt_path = src_img_path.replace('.png', ext)
                if os.path.exists(src_gt_path):
                    dst_gt_path = dst_img_path.replace('.png', ext)
                    shutil.copy2(src_gt_path, dst_gt_path)""",
        
        """    # Create sample corpus directory
    os.makedirs(os.path.join(output_dir, 'corpus'), exist_ok=True)
    
    # Create sample set file
    sample_set_file = os.path.join(output_dir, 'sample_set.txt')
    with open(sample_set_file, 'w') as f:
        for img_path in sample_paths:
            f.write(img_path + '\\n')
            
            # Create directory in sample corpus
            # The img_path is already the relative path from corpus_path
            sample_img_dir = os.path.join(output_dir, 'corpus', os.path.dirname(img_path))
            os.makedirs(sample_img_dir, exist_ok=True)
            
            # Copy image to sample corpus
            src_img_path = os.path.join(corpus_path, img_path)
            dst_img_path = os.path.join(output_dir, 'corpus', img_path)
            
            # Check if source image exists
            if not os.path.exists(src_img_path):
                print(f"Warning: Source image not found: {src_img_path}")
                continue
                
            shutil.copy2(src_img_path, dst_img_path)
            
            # Copy ground truth files if they exist
            base_src_path = src_img_path[:-4] if src_img_path.endswith('.png') else src_img_path
            base_dst_path = dst_img_path[:-4] if dst_img_path.endswith('.png') else dst_img_path
            
            for ext in ['.semantic', '.agnostic']:
                src_gt_path = base_src_path + ext
                if os.path.exists(src_gt_path):
                    dst_gt_path = base_dst_path + ext
                    shutil.copy2(src_gt_path, dst_gt_path)"""
    )
    
    # Write the fixed content back
    with open('prepare_corpus.py', 'w') as f:
        f.write(content)
    
    print("Fixed prepare_corpus.py")

def clean_sample_data():
    """Clean up and regenerate the sample data directory"""
    print("Cleaning up sample data directory...")
    
    # Remove existing sample data
    if os.path.exists('./sample_data'):
        shutil.rmtree('./sample_data')
    
    # Create sample data from fixed set file
    print("Regenerating sample data from fixed set file...")
    os.system('python /ml/models/resources/tf-deep-omr/src/prepare_corpus.py -corpus "/ml/models/resources/tf-deep-omr/Data" -set "/ml/models/resources/tf-deep-omr/Data/train_fixed.txt" -sample 100 -output "/ml/models/resources/tf-deep-omr/sample_data"')
    
    # Verify sample data was created correctly
    if os.path.exists('/ml/models/resources/tf-deep-omr/sample_data/corpus') and os.path.exists('/ml/models/resources/tf-deep-omr/sample_data/sample_set.txt'):
        print(f"Number of samples in set file: {len(open('/ml/models/resources/tf-deep-omr/sample_data/sample_set.txt').readlines())}")
        print(f"Number of image files in corpus: {len(glob.glob('/ml/models/resources/tf-deep-omr/sample_data/corpus/**/*.png', recursive=True))}")
    else:
        print("ERROR: Failed to create sample dataset!")
        return False
    
    return True

def update_train_script():
    """Update the training script to use the fixed sample data"""
    print("Updating training script...")
    
    # Read the current file
    with open('train_with_recovery.sh', 'r') as f:
        content = f.read()
    
    # Update the script to use the fixed sample data
    content = content.replace(
        """# If sample dataset doesn't exist, create it from the fixed set file
if [ ! -d "$CORPUS" ] || [ ! -f "$SET" ]; then
    echo "Sample dataset not found. Creating from fixed set file..."
    python prepare_corpus.py -corpus "./Data" -set "./data/train_fixed.txt" -sample 100 -output "./sample_data"
    
    # Verify sample data was created
    if [ ! -d "$CORPUS" ] || [ ! -f "$SET" ]; then
        echo "ERROR: Failed to create sample dataset!"
        exit 1
    fi
fi""",
        
        """# If sample dataset doesn't exist, create it from the fixed set file
if [ ! -d "$CORPUS" ] || [ ! -f "$SET" ]; then
    echo "Sample dataset not found. Creating from fixed set file..."
    python prepare_corpus.py -corpus "./data" -set "./data/train_fixed.txt" -sample 100 -output "./sample_data"
    
    # Verify sample data was created
    if [ ! -d "$CORPUS" ] || [ ! -f "$SET" ]; then
        echo "ERROR: Failed to create sample dataset!"
        exit 1
    fi
fi"""
    )
    
    # Write the fixed content back
    with open('train_with_recovery.sh', 'w') as f:
        f.write(content)
    
    print("Updated training script")

def main():
    """Main function to fix all issues"""
    parser = argparse.ArgumentParser(description='Fix OMR training system issues')
    parser.add_argument('--skip-clean', action='store_true', help='Skip cleaning sample data')
    args = parser.parse_args()
    
    print("=== Starting comprehensive fix for OMR training system ===")
    
    # Fix path handling in primus.py
    fix_primus_paths()
    
    # Fix prepare_corpus.py
    fix_prepare_corpus()
    
    # Update training script
    update_train_script()
    
    # Clean and regenerate sample data
    if not args.skip_clean:
        if not clean_sample_data():
            print("Failed to clean and regenerate sample data")
            return 1
    
    print("=== All fixes applied successfully ===")
    print("You can now run the training script: ./train_with_recovery.sh")
    return 0

if __name__ == "__main__":
    sys.exit(main())
