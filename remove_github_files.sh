#!/bin/bash

# List of files and directories that were in the GitHub repository
GITHUB_FILES=(
    "docs"
    "requirements.txt"
    "setup.py"
    "src/__init__.py"
    "src/cli"
    "src/config"
    "src/config_manager.py"
    "src/core/__init__.py"
    "src/core/advanced_symbol_recognizer.py"
    "src/core/omr_exceptions.py"
    "src/core/omr_pipeline.py"
    "src/core/omr_processor.py"
    "src/core/symbol_preprocessor.py"
    "src/core/test_write.txt"
    "src/dashboard"
    "src/export"
    "src/logging_config.py"
    "src/main.py"
    "src/performance"
    "src/preprocessing"
    "src/recognition"
    "src/segmentation"
    "src/ui"
    "src/utils"
    "tests"
)

# Remove each file/directory if it exists
for item in "${GITHUB_FILES[@]}"; do
    if [ -e "$item" ]; then
        echo "Removing: $item"
        rm -rf "$item"
    fi
done

echo "Done. All files that were in the GitHub repository have been removed."
echo "Your local files that weren't in GitHub have been preserved."
