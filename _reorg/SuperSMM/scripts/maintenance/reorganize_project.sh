#!/bin/bash
# reorganize_project.sh
# Reorganizes project structure for better organization

set -e

# Simple logging function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Move function with logging
move_if_exists() {
    local src="$1"
    local dest="$2"
    if [ -e "$src" ]; then
        log "Moving $src to $dest"
        if [ -d "$src" ] && [ -n "$(ls -A "$src" 2>/dev/null)" ]; then
            cp -rv "$src"/* "$dest/" 2>/dev/null || true
            rm -rf "$src"
        else
            mv -v "$src" "$dest" 2>/dev/null || true
        fi
    else
        log "Source $src does not exist, skipping"
    fi
}

# Change to project root
cd ../..

# 1. Create new directory structure
log "Creating directory structure"

# ML directories
log "Creating ML directories"
mkdir -pv ml/{models,datasets,checkpoints,notebooks}
mkdir -pv ml/data/{raw,processed}/{training,validation,test}

# Data directories
log "Creating data directories"
mkdir -pv data/{input/{scores,audio,midi},output/{exports,archives},temp/test_artifacts}

# Log directories
log "Creating log directories"
mkdir -pv logs/{app,debug,ml}

# Development directory
log "Creating development directory"
mkdir -pv .dev

# Archive directory
log "Creating archive directory"
mkdir -pv archive

# 2. Move files to new structure
log "Moving files to new structure"

# ML-related files
for dir in datasets models "src/models" "data/raw_scans" "data/training" "data/validation" "resources/ml_models"; do
    if [ -d "$dir" ]; then
        log "Processing $dir"
        case "$dir" in
            "datasets") dest="ml/datasets" ;;
            "models"|"src/models"|"resources/ml_models") dest="ml/models" ;;
            "data/raw_scans") dest="ml/data/raw/training" ;;
            "data/training") dest="ml/data/processed/training" ;;
            "data/validation") dest="ml/data/processed/validation" ;;
        esac
        move_if_exists "$dir" "$dest"
    fi
done

# Data files
for dir in output outputs exports imports temp_test_dir; do
    if [ -d "$dir" ]; then
        log "Processing $dir"
        case "$dir" in
            "output"|"outputs") dest="data/output" ;;
            "exports") dest="data/output/exports" ;;
            "imports") dest="data/input" ;;
            "temp_test_dir") dest="data/temp/test_artifacts" ;;
        esac
        move_if_exists "$dir" "$dest"
    fi
done

# 3. Move notebooks
log "Moving notebooks"
if [ -d notebooks ]; then
    move_if_exists "notebooks" "ml/notebooks"
fi

# 4. Consolidate logs
log "Consolidating logs"
if [ -d debug_logs ]; then
    move_if_exists "debug_logs" "logs/debug"
fi

# 5. Create development directory
log "Creating development directory"
mkdir -pv .dev
for file in pytest.ini setup.py requirements.txt; do
    if [ -f "$file" ]; then
        mv -v "$file" ".dev/"
        ln -sfv ".dev/$file" "$file"
    fi
done

# 6. Archive old ML implementation
log "Archiving old ML implementation"
mkdir -pv archive
if [ -d resources/tf-deep-omr ]; then
    mv -v resources/tf-deep-omr archive/tf-deep-omr-old
fi

# Create README files
cat > ml/README.md << 'EOF'
# Machine Learning Directory

This directory contains all ML-related files and resources for the SuperSMM project.

## Structure

```
ml/
├── models/           # Model definitions and saved models
├── datasets/         # Preprocessed datasets ready for training
├── checkpoints/      # Model checkpoints during training
├── data/            # Raw and processed data
│   ├── raw/         # Original, unprocessed data
│   └── processed/   # Processed data ready for training
├── notebooks/       # Jupyter notebooks for experimentation
```

## Usage

- Place raw data in `data/raw/`
- Processed data goes in `data/processed/`
- Keep notebooks in `notebooks/` for experiments
- Store trained models in `models/`
- Use `checkpoints/` for training snapshots
EOF

cat > .dev/README.md << 'EOF'
# Development Configuration

This directory contains development configuration files:

- `pytest.ini`: PyTest configuration
- `setup.py`: Package setup configuration
- `requirements.txt`: Python dependencies

Note: Symlinks to these files exist in the project root for compatibility.
EOF

# Create .gitkeep files
find ml -type d -empty -exec touch {}/.gitkeep \;

log "Project reorganization complete"

# Cleanup
rm "$(dirname "$0")/reorg_log_helper.py"
