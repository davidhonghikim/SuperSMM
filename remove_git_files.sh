#!/bin/bash

# This script will remove all files that were tracked by Git
# It will preserve any new files that weren't in the Git repository

# First, get the list of files that were tracked by Git
if [ -f .git ]; then
    echo "Error: .git is a file, not a directory. This doesn't look like a git repository."
    exit 1
fi

if [ ! -d .git ]; then
    echo "Error: .git directory not found. This doesn't look like a git repository."
    exit 1
fi

# Get the list of files that were tracked by Git
git ls-files | while read -r file; do
    # Remove the file if it exists
    if [ -e "$file" ]; then
        echo "Removing: $file"
        rm -f "$file"
        
        # Remove parent directories if they're empty
        dir=$(dirname "$file")
        while [ "$dir" != "." ] && [ -d "$dir" ] && [ -z "$(ls -A "$dir" 2>/dev/null)" ]; do
            echo "Removing empty directory: $dir"
            rmdir "$dir"
            dir=$(dirname "$dir")
        done
    fi
done

# Remove .git directory
echo "Removing .git directory"
rm -rf .git

echo "Done. All files that were tracked by Git have been removed."
echo "Your local files that weren't in Git have been preserved."
