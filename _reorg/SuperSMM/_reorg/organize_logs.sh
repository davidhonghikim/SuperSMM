#!/bin/bash

# Script to reorganize log files into a centralized structure
# Run this from the project root directory

# Exit on error
set -e

# Define project root and log directories
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOGS_DIR="$PROJECT_ROOT/logs"
REORG_LOGS_DIR="$PROJECT_ROOT/_reorg/logs"

# Create log directory structure
echo "Creating log directory structure..."
mkdir -p "$LOGS_DIR"/{app,debug,linting,maintenance,ml,system}

# Function to move log files
move_logs() {
    local src_pattern="$1"
    local dest_dir="$2"
    
    find "$PROJECT_ROOT" -type f -path "$src_pattern" -not -path "$LOGS_DIR/*" -print0 | while IFS= read -r -d $'\0' file; do
        if [ -f "$file" ]; then
            echo "Moving $file to $dest_dir/"
            mkdir -p "$dest_dir"
            mv "$file" "$dest_dir/"
        fi
    done
}

# Move different types of logs
echo "Organizing log files..."
move_logs "*error*.log" "$LOGS_DIR/debug"
move_logs "*debug*.log" "$LOGS_DIR/debug"
move_logs "*lint*.log" "$LOGS_DIR/linting"
move_logs "*maintain*.log" "$LOGS_DIR/maintenance"
move_logs "*.log" "$LOGS_DIR/app"  # Default location for other logs

# Special handling for ML-related logs
find "$PROJECT_ROOT" \( -name "*.log" -o -name "*.csv" \) -not -path "$LOGS_DIR/*" -exec grep -l -i "train\\|model\\|epoch\\|loss\\|accuracy" {} \; | while read -r file; do
    echo "Moving ML log: $file"
    mv "$file" "$LOGS_DIR/ml/"
done

# Create a README for the logs directory
cat > "$LOGS_DIR/README.md" << 'EOF'
# Log Files

This directory contains all log files organized by category:

- \`app/\` - General application logs
- \`debug/\` - Debug and error logs
- \`linting/\` - Linting logs
- \`maintenance/\` - Maintenance logs
- \`ml/\` - Machine learning training logs
- \`system/\` - System-level logs

## Usage

To log to these directories, use project-relative paths:

\`\`\`python
from pathlib import Path
LOG_FILE = Path(__file__).resolve().parents[2] / "logs" / "app" / "my_log.log"
\`\`\`

Or use the centralized logger utility:

\`\`\`python
from utils.logger import setup_logger
logger = setup_logger(__name__)
logger.info("Your log message")
\`\`\`
EOF

echo "Log organization complete!"
echo "Review the logs in: $LOGS_DIR"
echo "A README.md has been created in the logs directory."
