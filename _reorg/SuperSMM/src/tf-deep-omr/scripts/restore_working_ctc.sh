#!/bin/bash

# Script to restore the working CTC training files
# This will restore the files that were fixed to work with the tuple format returned by primus.py

# Set the base directory
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/.."
cd "$BASE_DIR"

echo "Restoring working CTC training files..."
echo "Base directory: $BASE_DIR"

# Create backup of current files in the new structure
BACKUP_DIR="backups/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

# Backup current files in the legacy structure
if [ -f "src/models/ctc_training.py" ]; then
  cp "src/models/ctc_training.py" "$BACKUP_DIR/legacy_ctc_training.py"
fi

if [ -f "src/data/primus.py" ]; then
  cp "src/data/primus.py" "$BACKUP_DIR/legacy_primus.py"
fi

if [ -f "src/models/ctc_model.py" ]; then
  cp "src/models/ctc_model.py" "$BACKUP_DIR/legacy_ctc_model.py"
fi

if [ -f "src/utils/ctc_utils.py" ]; then
  cp "src/utils/ctc_utils.py" "$BACKUP_DIR/legacy_ctc_utils.py"
fi

# Backup current files in the new package structure
if [ -f "src/deep_omr/models/ctc_training.py" ]; then
  cp "src/deep_omr/models/ctc_training.py" "$BACKUP_DIR/deep_omr_ctc_training.py"
fi

if [ -f "src/deep_omr/data/primus.py" ]; then
  cp "src/deep_omr/data/primus.py" "$BACKUP_DIR/deep_omr_primus.py"
fi

if [ -f "src/deep_omr/models/ctc_model.py" ]; then
  cp "src/deep_omr/models/ctc_model.py" "$BACKUP_DIR/deep_omr_ctc_model.py"
fi

if [ -f "src/deep_omr/utils/ctc_utils.py" ]; then
  cp "src/deep_omr/utils/ctc_utils.py" "$BACKUP_DIR/deep_omr_ctc_utils.py"
fi

# Restore from backups directory to the legacy structure
cp backups/ctc_training.py.working src/models/ctc_training.py
cp backups/primus.py.working src/data/primus.py
cp backups/ctc_model.py.working src/models/ctc_model.py
cp backups/ctc_utils.py.working src/utils/ctc_utils.py

# Also restore to the new package structure
cp backups/ctc_training.py.working src/deep_omr/models/ctc_training.py
cp backups/primus.py.working src/deep_omr/data/primus.py
cp backups/ctc_model.py.working src/deep_omr/models/ctc_model.py
cp backups/ctc_utils.py.working src/deep_omr/utils/ctc_utils.py

# Also restore to the root directory for backward compatibility
cp backups/ctc_training.py.working ctc_training.py
cp backups/primus.py.working primus.py
cp backups/ctc_model.py.working ctc_model.py
cp backups/ctc_utils.py.working ctc_utils.py

echo "Files restored successfully to both the legacy structure, new package structure, and the root directory!"
echo "Backup of current files created in $BACKUP_DIR"
