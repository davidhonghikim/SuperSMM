#!/bin/bash
# reorganize_data_dirs.sh
# Reorganizes data directories into a more structured layout

set -e

# Setup logging
TIMESTAMP=$(date +%Y-%m-%d)
LOG_DIR="../../logs/maintenance/$TIMESTAMP"
mkdir -p "$LOG_DIR"

# Python script for structured logging
cat > "$(dirname "$0")/reorg_log_helper.py" << 'EOF'
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent / 'src'))
from utils.logging_utils import get_logger
logger = get_logger('reorganize_dirs', 'maintenance', structured=True)
logger.info(sys.stdin.read())
EOF

# Log function that uses structured logging
log() {
    echo "$1" | python3 "$(dirname "$0")/reorg_log_helper.py"
}

# Change to project root
cd ../..

# Create new directory structure
log "Creating new directory structure"
mkdir -p data/{input,output,temp}
mkdir -p data/input/{scores,audio,midi}
mkdir -p data/output/{exports,archives}
mkdir -p data/temp/test_artifacts

# Move files from imports to appropriate input directories
log "Moving files from imports directory"
[ -d imports ] && {
    mv imports/*.pdf data/input/scores/ 2>/dev/null || true
    mv imports/*.mxl data/input/midi/ 2>/dev/null || true
    mv imports/*.mp3 data/input/audio/ 2>/dev/null || true
    rm -rf imports
}

# Move files from exports to output/exports
log "Moving files from exports directory"
[ -d exports ] && {
    mv exports/* data/output/exports/ 2>/dev/null || true
    rm -rf exports
}

# Move files from outputs to output
log "Moving files from outputs directory"
[ -d outputs ] && {
    mv outputs/archive/* data/output/archives/ 2>/dev/null || true
    mv outputs/* data/output/ 2>/dev/null || true
    rm -rf outputs
}

# Move files from output to data/output
log "Moving files from output directory"
[ -d output ] && {
    mv output/* data/output/ 2>/dev/null || true
    rm -rf output
}

# Move temp test files
log "Moving temporary test files"
[ -d temp_test_dir ] && {
    mv temp_test_dir/* data/temp/test_artifacts/ 2>/dev/null || true
    rm -rf temp_test_dir
}

# Create .gitkeep files to preserve directory structure
find data -type d -empty -exec touch {}/.gitkeep \;

# Create README.md for data directory
cat > data/README.md << 'EOF'
# Data Directory Structure

This directory contains all data files used by the SuperSMM project.

## Structure

```
data/
├── input/              # Input files
│   ├── scores/        # Sheet music PDFs
│   ├── audio/         # Audio files
│   └── midi/          # MIDI files
├── output/            # Generated outputs
│   ├── exports/       # Exported files
│   └── archives/      # Archived outputs
└── temp/              # Temporary files
    └── test_artifacts/ # Test-generated files
```

## Usage

- Place input files in the appropriate input subdirectory
- Generated files will appear in output/exports
- Older outputs are automatically archived in output/archives
- Temporary files are cleaned up periodically

## Notes

- Do not store large binary files in Git
- Use .gitignore patterns for temporary files
- Archive old outputs regularly
EOF

log "Directory reorganization complete"

# Cleanup
rm "$(dirname "$0")/reorg_log_helper.py"
