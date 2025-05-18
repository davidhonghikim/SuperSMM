#!/bin/bash

# Script to check the project structure and verify that all required files are in place

# Set the base directory
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/.."
cd "$BASE_DIR"

echo "Checking project structure..."
echo "Base directory: $BASE_DIR"

# Check directories
DIRS=(
    "src"
    "src/models"
    "src/data"
    "src/utils"
    "src/deep_omr"
    "src/deep_omr/cli"
    "src/deep_omr/models"
    "src/deep_omr/data"
    "src/deep_omr/utils"
    "scripts"
    "config"
    "backups"
)

for dir in "${DIRS[@]}"; do
    if [ ! -d "$dir" ]; then
        echo "ERROR: Directory $dir does not exist!"
        echo "Creating directory $dir..."
        mkdir -p "$dir"
    fi
done

# Check required files
FILES=(
    # Legacy files (for backward compatibility)
    "src/models/ctc_model.py"
    "src/models/ctc_training.py"
    "src/data/primus.py"
    "src/utils/ctc_utils.py"
    
    # New package structure
    "src/deep_omr/__init__.py"
    "src/deep_omr/cli/__init__.py"
    "src/deep_omr/cli/main.py"
    "src/deep_omr/cli/train.py"
    "src/deep_omr/models/__init__.py"
    "src/deep_omr/models/ctc_model.py"
    "src/deep_omr/models/ctc_training.py"
    "src/deep_omr/data/__init__.py"
    "src/deep_omr/data/primus.py"
    "src/deep_omr/utils/__init__.py"
    "src/deep_omr/utils/ctc_utils.py"
    "src/deep_omr/utils/config_loader.py"
    
    # Main files
    "train.py"
    "setup.py"
    "config/training_config.yaml"
    "config/test_config.yaml"
    
    # Scripts
    "scripts/run_training.sh"
    "scripts/test_training.sh"
    "scripts/check_structure.sh"
    "scripts/restore_working_ctc.sh"
)

MISSING=0
for file in "${FILES[@]}"; do
    if [ ! -f "$file" ]; then
        echo "ERROR: File $file does not exist!"
        MISSING=$((MISSING+1))
    fi
done

# Check if __init__.py files exist in all Python directories
PYTHON_DIRS=(
    "src"
    "src/models"
    "src/data"
    "src/utils"
    "src/deep_omr"
    "src/deep_omr/cli"
    "src/deep_omr/models"
    "src/deep_omr/data"
    "src/deep_omr/utils"
)

for dir in "${PYTHON_DIRS[@]}"; do
    if [ ! -f "$dir/__init__.py" ]; then
        echo "WARNING: File $dir/__init__.py does not exist!"
        echo "Creating empty $dir/__init__.py file..."
        touch "$dir/__init__.py"
    fi
done

# Check if the package is installable
check_package_installable() {
    echo "Checking if the package is installable..."
    
    # Check if setup.py exists
    if [ ! -f "$BASE_DIR/setup.py" ]; then
        echo "WARNING: setup.py does not exist. The package may not be installable."
        return 1
    fi
    
    # Check if the package can be installed in development mode
    echo "Testing package installation in development mode..."
    if pip install -e "$BASE_DIR" --no-deps &> /dev/null; then
        echo "Package is installable in development mode."
        return 0
    else
        echo "WARNING: Package installation failed. Check setup.py for errors."
        return 1
    fi
}

# Check if the CLI is working
check_cli() {
    echo "Checking if the CLI is working..."
    
    # Check if the deep-omr command is available
    if command -v deep-omr &> /dev/null; then
        echo "deep-omr command is available."
        
        # Check if the CLI shows help
        if deep-omr --help &> /dev/null; then
            echo "CLI help command works."
            return 0
        else
            echo "WARNING: CLI help command failed."
            return 1
        fi
    else
        echo "WARNING: deep-omr command is not available. Try installing the package with 'pip install -e .'."
        return 1
    fi
}

# Summary
echo "\n=== Summary ==="
if [ $MISSING -eq 0 ]; then
    echo "✅ All required files are present!"
    echo "✅ Project structure is valid."
else
    echo "❌ $MISSING required files are missing!"
    echo "❌ Please restore or create the missing files."
fi

# Check package installability
if check_package_installable; then
    echo "✅ Package is installable."
else
    echo "❌ Package installation check failed."
fi

# Check CLI functionality
if check_cli; then
    echo "✅ CLI is working."
else
    echo "❌ CLI check failed."
fi

echo "\nDone."
