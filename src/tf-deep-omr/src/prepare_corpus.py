#!/usr/bin/env python
"""
Corpus preparation and validation script for OMR training.
This script checks if the corpus is properly set up and helps diagnose issues.
"""

import os
import sys
import argparse
import glob
import json
from pathlib import Path
import random
import shutil

def check_corpus_structure(corpus_path, set_file_path):
    """
    Check if the corpus directory structure matches what's expected in the set file.
    
    Args:
        corpus_path: Path to the corpus directory
        set_file_path: Path to the set file containing image references
    
    Returns:
        dict: Statistics about the corpus
    """
    print(f"Checking corpus structure at: {corpus_path}")
    print(f"Using set file: {set_file_path}")
    
    # Check if corpus directory exists
    if not os.path.isdir(corpus_path):
        print(f"ERROR: Corpus directory '{corpus_path}' does not exist!")
        return None
    
    # Check if set file exists
    if not os.path.isfile(set_file_path):
        print(f"ERROR: Set file '{set_file_path}' does not exist!")
        return None
    
    # Read set file
    with open(set_file_path, 'r') as f:
        set_lines = f.readlines()
    
    print(f"Set file contains {len(set_lines)} entries")
    
    # Extract image paths from set file
    image_paths = []
    for line in set_lines:
        parts = line.strip().split()
        if len(parts) >= 1:
            image_path = parts[0]
            image_paths.append(image_path)
    
    # Check if images exist
    missing_images = []
    existing_images = []
    
    for img_path in image_paths:
        full_path = os.path.join(corpus_path, img_path)
        if not os.path.isfile(full_path):
            missing_images.append(img_path)
        else:
            existing_images.append(img_path)
    
    # Collect statistics
    stats = {
        "total_entries": len(set_lines),
        "total_images_found": len(existing_images),
        "total_images_missing": len(missing_images),
        "missing_images": missing_images[:20],  # Show only first 20 to avoid clutter
        "missing_images_sample": missing_images[:5] if missing_images else []
    }
    
    # Print summary
    print(f"\nCorpus Statistics:")
    print(f"  Total entries in set file: {stats['total_entries']}")
    print(f"  Images found: {stats['total_images_found']}")
    print(f"  Images missing: {stats['total_images_missing']}")
    
    if stats['total_images_missing'] > 0:
        print("\nSample of missing images:")
        for img in stats['missing_images_sample']:
            print(f"  - {img}")
        
        if stats['total_images_missing'] > 5:
            print(f"  ... and {stats['total_images_missing'] - 5} more")
    
    return stats

def fix_corpus_paths(set_file_path, corpus_path=None, output_set_file=None):
    """
    Create a new set file with corrected paths based on what's actually available.
    
    Args:
        set_file_path: Path to the original set file
        corpus_path: Path to the corpus directory (if None, tries to detect)
        output_set_file: Path to write the corrected set file (if None, adds '_fixed' suffix)
    
    Returns:
        str: Path to the corrected set file
    """
    if not output_set_file:
        base, ext = os.path.splitext(set_file_path)
        output_set_file = f"{base}_fixed{ext}"
    
    print(f"Creating corrected set file: {output_set_file}")
    
    # Read original set file
    with open(set_file_path, 'r') as f:
        set_lines = f.readlines()
    
    # Determine corpus directory
    if corpus_path is None:
        # Try to detect corpus directory
        set_dir = os.path.dirname(os.path.abspath(set_file_path))
        possible_corpus_dirs = [
            os.path.join(set_dir, '..', 'corpus'),
            os.path.join(set_dir, '..', 'Data'),
            os.path.join(set_dir, 'corpus'),
            os.path.join(set_dir, 'Data'),
            './Data',
            './corpus'
        ]
        
        for dir_path in possible_corpus_dirs:
            norm_path = os.path.normpath(dir_path)
            if os.path.isdir(norm_path):
                corpus_path = norm_path
                break
    
    if not corpus_path or not os.path.isdir(corpus_path):
        print(f"ERROR: Could not find corpus directory")
        return None
    
    print(f"Using corpus directory: {corpus_path}")
    
    # Find all available PNG files
    print("Searching for PNG files (this may take a while)...")
    available_images = glob.glob(os.path.join(corpus_path, '**', '*.png'), recursive=True)
    print(f"Found {len(available_images)} PNG files in corpus directory")
    
    # Create a mapping from filename to full relative path
    image_map = {}
    for img in available_images:
        # Get the relative path from the corpus directory
        rel_path = os.path.relpath(img, corpus_path)
        # Get just the filename
        filename = os.path.basename(img)
        # Get the parent directory name (usually matches the filename without extension)
        parent_dir = os.path.basename(os.path.dirname(img))
        
        # Store both the full path and parent directory for matching
        image_map[filename] = rel_path
        if parent_dir + '.png' == filename:
            # This is a common pattern in OMR datasets
            image_map[parent_dir] = rel_path
    
    # Create corrected set file
    corrected_lines = []
    fixed_count = 0
    duplicated_paths_count = 0
    
    for line in set_lines:
        parts = line.strip().split()
        if len(parts) >= 1:
            img_path = parts[0]
            img_basename = os.path.basename(img_path)
            
            # Check for duplicated path pattern (path/to/file.png/path/to/file.png.png)
            if '/primus/' in img_path and img_path.count('/primus/') > 1:
                # Fix duplicated path pattern
                path_parts = img_path.split('/primus/')
                if len(path_parts) > 2:
                    # Keep only the first part + primus + last part
                    corrected_path = path_parts[0] + '/primus/' + path_parts[-1]
                    # Remove any double extensions (.png.png)
                    if corrected_path.endswith('.png.png'):
                        corrected_path = corrected_path[:-4]
                    
                    # Check if the corrected path exists
                    if os.path.isfile(os.path.join(corpus_path, corrected_path)):
                        corrected_parts = [corrected_path] + parts[1:]
                        corrected_line = ' '.join(corrected_parts) + '\n'
                        corrected_lines.append(corrected_line)
                        duplicated_paths_count += 1
                        fixed_count += 1
                        continue
            
            # If the exact path doesn't exist but we have a file with the same name
            if img_basename in image_map:
                corrected_path = image_map[img_basename]
                corrected_parts = [corrected_path] + parts[1:]
                corrected_line = ' '.join(corrected_parts) + '\n'
                corrected_lines.append(corrected_line)
                fixed_count += 1
            else:
                # Keep the line as is
                corrected_lines.append(line)
    
    # Write corrected set file
    with open(output_set_file, 'w') as f:
        f.writelines(corrected_lines)
    
    print(f"Created corrected set file with {fixed_count} fixed paths")
    if duplicated_paths_count > 0:
        print(f"Fixed {duplicated_paths_count} duplicated path patterns")
    return output_set_file

def create_sample_corpus(corpus_path, set_file_path, output_dir, sample_size=10):
    """
    Create a small sample corpus for testing.
    
    Args:
        corpus_path: Path to the corpus directory
        set_file_path: Path to the set file
        output_dir: Directory to create the sample corpus
        sample_size: Number of samples to include
    """
    import shutil
    
    print(f"Creating sample corpus with {sample_size} samples in {output_dir}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Read set file
    with open(set_file_path, 'r') as f:
        set_lines = f.readlines()
    
    # Take a sample
    if len(set_lines) > sample_size:
        import random
        sample_lines = random.sample(set_lines, sample_size)
    else:
        sample_lines = set_lines
    
    # Create sample set file
    sample_set_path = os.path.join(output_dir, 'sample_set.txt')
    with open(sample_set_path, 'w') as f:
        f.writelines(sample_lines)
    
    # Copy sample images
    sample_corpus_dir = os.path.join(output_dir, 'corpus')
    os.makedirs(sample_corpus_dir, exist_ok=True)
    
    copied_count = 0
    for line in sample_lines:
        parts = line.strip().split()
        if len(parts) >= 1:
            img_path = parts[0]
            full_path = os.path.join(corpus_path, img_path)
            
            if os.path.isfile(full_path):
                # Create directory structure
                img_dir = os.path.dirname(img_path)
                target_dir = os.path.join(sample_corpus_dir, img_dir)
                os.makedirs(target_dir, exist_ok=True)
                
                # Copy image
                target_path = os.path.join(sample_corpus_dir, img_path)
                shutil.copy2(full_path, target_path)
                
                # Copy semantic and agnostic files if they exist
                base_path = full_path[:-4] if full_path.endswith('.png') else full_path
                for ext in ['.semantic', '.agnostic']:
                    gt_path = base_path + ext
                    if os.path.exists(gt_path):
                        target_gt_path = target_path[:-4] + ext if target_path.endswith('.png') else target_path + ext
                        shutil.copy2(gt_path, target_gt_path)
                        print(f"Copied {ext} file: {os.path.basename(gt_path)}")
                    else:
                        # Create a dummy semantic file with basic content for testing
                        target_gt_path = target_path[:-4] + ext if target_path.endswith('.png') else target_path + ext
                        with open(target_gt_path, 'w') as f:
                            f.write("clef.G-2 note.quarter-2")
                        print(f"Created dummy {ext} file: {os.path.basename(target_gt_path)}")
                
                copied_count += 1
            else:
                # Try to handle duplicated paths
                if '/primus/' in img_path and img_path.count('/primus/') > 1:
                    # Extract the first part of the path (before the second /primus/)
                    first_part = img_path.split('/primus/')[0] + '/primus/' + img_path.split('/primus/')[1]
                    fixed_path = os.path.join(corpus_path, first_part)
                    
                    if os.path.isfile(fixed_path):
                        # Create directory structure
                        img_dir = os.path.dirname(img_path)
                        target_dir = os.path.join(sample_corpus_dir, img_dir)
                        os.makedirs(target_dir, exist_ok=True)
                        
                        # Copy image with the original path structure
                        target_path = os.path.join(sample_corpus_dir, img_path)
                        shutil.copy2(fixed_path, target_path)
                        
                        # Copy semantic and agnostic files if they exist
                        base_path = fixed_path[:-4] if fixed_path.endswith('.png') else fixed_path
                        for ext in ['.semantic', '.agnostic']:
                            gt_path = base_path + ext
                            if os.path.exists(gt_path):
                                target_gt_path = target_path[:-4] + ext if target_path.endswith('.png') else target_path + ext
                                shutil.copy2(gt_path, target_gt_path)
                                print(f"Copied {ext} file for fixed path: {os.path.basename(gt_path)}")
                            else:
                                # Create a dummy semantic file with basic content for testing
                                target_gt_path = target_path[:-4] + ext if target_path.endswith('.png') else target_path + ext
                                with open(target_gt_path, 'w') as f:
                                    f.write("clef.G-2 note.quarter-2")
                                print(f"Created dummy {ext} file for fixed path: {os.path.basename(target_gt_path)}")
                        
                        copied_count += 1
                        print(f"Fixed duplicated path: {img_path}")
                elif img_path.endswith('.png.png'):
                    # Try removing the duplicate .png extension
                    fixed_path = os.path.join(corpus_path, img_path[:-4])
                    
                    if os.path.isfile(fixed_path):
                        # Create directory structure
                        img_dir = os.path.dirname(img_path)
                        target_dir = os.path.join(sample_corpus_dir, img_dir)
                        os.makedirs(target_dir, exist_ok=True)
                        
                        # Copy image with the original path structure
                        target_path = os.path.join(sample_corpus_dir, img_path)
                        shutil.copy2(fixed_path, target_path)
                        
                        # Copy semantic and agnostic files if they exist
                        base_path = fixed_path[:-4] if fixed_path.endswith('.png') else fixed_path
                        for ext in ['.semantic', '.agnostic']:
                            gt_path = base_path + ext
                            if os.path.exists(gt_path):
                                target_gt_path = target_path[:-4] + ext if target_path.endswith('.png') else target_path + ext
                                shutil.copy2(gt_path, target_gt_path)
                                print(f"Copied {ext} file for fixed extension: {os.path.basename(gt_path)}")
                            else:
                                # Create a dummy semantic file with basic content for testing
                                target_gt_path = target_path[:-4] + ext if target_path.endswith('.png') else target_path + ext
                                with open(target_gt_path, 'w') as f:
                                    f.write("clef.G-2 note.quarter-2")
                                print(f"Created dummy {ext} file for fixed extension: {os.path.basename(target_gt_path)}")
                        
                        copied_count += 1
                        print(f"Fixed duplicated extension: {img_path}")
    
    print(f"Created sample corpus with {copied_count} images")
    print(f"Sample set file: {sample_set_path}")
    return sample_set_path

def find_corpus_candidates(base_dir):
    """
    Find potential corpus directories in the project.
    
    Args:
        base_dir: Base directory to search from
    
    Returns:
        list: Potential corpus directories
    """
    print(f"Searching for potential corpus directories in: {base_dir}")
    
    # Look for directories that might contain corpus data
    potential_dirs = []
    
    # Common corpus directory names
    corpus_names = ['corpus', 'data', 'dataset', 'images', 'primus']
    
    for root, dirs, files in os.walk(base_dir):
        for d in dirs:
            if d.lower() in corpus_names:
                potential_dirs.append(os.path.join(root, d))
        
        # Don't go too deep
        if root.count(os.sep) - base_dir.count(os.sep) > 3:
            dirs.clear()
    
    # Check each potential directory for PNG files
    corpus_candidates = []
    for d in potential_dirs:
        png_files = glob.glob(os.path.join(d, '**', '*.png'), recursive=True)
        if png_files:
            corpus_candidates.append({
                'path': d,
                'image_count': len(png_files),
                'sample_images': [os.path.basename(f) for f in png_files[:3]]
            })
    
    # Print results
    if corpus_candidates:
        print("\nPotential corpus directories found:")
        for i, candidate in enumerate(corpus_candidates):
            print(f"{i+1}. {candidate['path']} ({candidate['image_count']} PNG files)")
            print(f"   Sample images: {', '.join(candidate['sample_images'])}")
    else:
        print("No potential corpus directories found")
    
    return corpus_candidates

def main():
    parser = argparse.ArgumentParser(description='Corpus preparation and validation for OMR training')
    parser.add_argument('-corpus', dest='corpus', help='Path to the corpus directory')
    parser.add_argument('-set', dest='set', help='Path to the set file')
    parser.add_argument('-find', dest='find', action='store_true', help='Find potential corpus directories')
    parser.add_argument('-fix', dest='fix', action='store_true', help='Create a corrected set file')
    parser.add_argument('-sample', dest='sample', type=int, default=0, 
                        help='Create a sample corpus with specified number of samples')
    parser.add_argument('-output', dest='output', default='./sample_corpus',
                        help='Output directory for sample corpus')
    
    args = parser.parse_args()
    
    # Find potential corpus directories
    if args.find:
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
        find_corpus_candidates(base_dir)
        return
    
    # Validate arguments
    if not args.corpus and not args.find:
        print("ERROR: Please specify a corpus directory with -corpus or use -find to search for potential directories")
        return
    
    if not args.set and not args.find:
        print("ERROR: Please specify a set file with -set")
        return
    
    # Check corpus structure
    if args.corpus and args.set:
        stats = check_corpus_structure(args.corpus, args.set)
        
        # Create a corrected set file if requested
        if args.fix and stats and stats['total_images_missing'] > 0:
            fix_corpus_paths(args.set, args.corpus)
        
        # Create a sample corpus if requested
        if args.sample > 0:
            create_sample_corpus(args.corpus, args.set, args.output, args.sample)

if __name__ == '__main__':
    main()
