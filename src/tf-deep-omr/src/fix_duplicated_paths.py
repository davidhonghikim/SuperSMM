#!/usr/bin/env python
"""
Script to fix duplicated paths in the set file.
This specifically addresses the issue where paths are duplicated like:
./Data/primus/ID/ID.png/primus/ID/ID.png.png
"""

import os
import sys
import argparse
import re

def fix_duplicated_paths(set_file_path, corpus_path, output_file=None):
    """
    Fix duplicated paths in the set file.
    
    Args:
        set_file_path: Path to the set file
        corpus_path: Path to the corpus directory
        output_file: Path to write the fixed set file (if None, adds '_fixed' suffix)
    
    Returns:
        str: Path to the fixed set file
    """
    if not output_file:
        base, ext = os.path.splitext(set_file_path)
        output_file = f"{base}_fixed{ext}"
    
    print(f"Reading set file: {set_file_path}")
    
    # Read original set file
    with open(set_file_path, 'r') as f:
        set_lines = f.readlines()
    
    print(f"Found {len(set_lines)} entries in set file")
    
    # Pattern to match duplicated paths
    duplicated_pattern = re.compile(r'(.*?/primus/[^/]+/[^/]+\.png)/primus/([^/]+/[^/]+\.png.*)')
    
    # Fix duplicated paths
    fixed_lines = []
    fixed_count = 0
    error_count = 0
    
    for line in set_lines:
        parts = line.strip().split()
        if len(parts) >= 1:
            img_path = parts[0]
            full_path = os.path.join(corpus_path, img_path)
            
            # Check if the image exists
            if not os.path.isfile(full_path):
                error_count += 1
                
                # Try to fix common path issues
                # 1. Check for duplicated path pattern
                match = duplicated_pattern.match(img_path)
                if match:
                    # Get the first part of the path
                    first_part = match.group(1)
                    
                    # Check if this path exists
                    if os.path.isfile(os.path.join(corpus_path, first_part)):
                        # Use the first part as the corrected path
                        corrected_parts = [first_part] + parts[1:]
                        corrected_line = ' '.join(corrected_parts) + '\n'
                        fixed_lines.append(corrected_line)
                        fixed_count += 1
                        continue
                
                # 2. Check if removing .png.png works
                if img_path.endswith('.png.png'):
                    fixed_path = img_path[:-4]  # Remove last .png
                    if os.path.isfile(os.path.join(corpus_path, fixed_path)):
                        corrected_parts = [fixed_path] + parts[1:]
                        corrected_line = ' '.join(corrected_parts) + '\n'
                        fixed_lines.append(corrected_line)
                        fixed_count += 1
                        continue
                
                # 3. Check if the path exists without 'primus/' prefix
                if img_path.startswith('primus/'):
                    fixed_path = img_path[7:]  # Remove 'primus/'
                    if os.path.isfile(os.path.join(corpus_path, fixed_path)):
                        corrected_parts = [fixed_path] + parts[1:]
                        corrected_line = ' '.join(corrected_parts) + '\n'
                        fixed_lines.append(corrected_line)
                        fixed_count += 1
                        continue
                
                # If we couldn't fix it, keep the original line
                fixed_lines.append(line)
            else:
                # Path is already correct
                fixed_lines.append(line)
    
    # Write fixed set file
    with open(output_file, 'w') as f:
        f.writelines(fixed_lines)
    
    print(f"Found {error_count} paths with errors")
    print(f"Fixed {fixed_count} paths")
    print(f"Wrote fixed set file to: {output_file}")
    
    return output_file

def main():
    parser = argparse.ArgumentParser(description='Fix duplicated paths in set file')
    parser.add_argument('-set', dest='set', required=True, help='Path to the set file')
    parser.add_argument('-corpus', dest='corpus', required=True, help='Path to the corpus directory')
    parser.add_argument('-output', dest='output', help='Path to write the fixed set file')
    
    args = parser.parse_args()
    
    fix_duplicated_paths(args.set, args.corpus, args.output)

if __name__ == '__main__':
    main()
