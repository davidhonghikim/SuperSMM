#!/bin/bash

# Enhanced linting and fixing script with structured logging
set -e

# Check if target directory is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <target_directory>"
    exit 1
fi

TARGET_DIR="$1"
if [ ! -d "$TARGET_DIR" ]; then
    echo "Error: Directory '$TARGET_DIR' does not exist"
    exit 1
fi

# Setup logging
TIMESTAMP=$(date +%Y-%m-%d)
LOG_DIR="logs/linting/$TIMESTAMP"
mkdir -p "$LOG_DIR"

# Log function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Find Python files to process
FILES=$(find "$TARGET_DIR" -type f -name "*.py" -not -path "*/venv/*")

log "Starting linting and fixing process in $TARGET_DIR..."

# Check for required tools
if ! command -v pycodestyle &> /dev/null; then
    echo "Error: pycodestyle is not installed. Please install it with: pip install pycodestyle"
    exit 1
fi

if ! command -v autopep8 &> /dev/null; then
    echo "Error: autopep8 is not installed. Please install it with: pip install autopep8"
    exit 1
fi

# Process each Python file
for FILE in $FILES; do
    log "Processing file: $FILE"

    # Initial pycodestyle check
    INITIAL_CHECK=$(pycodestyle "$FILE" 2>&1) || true
    if [ -n "$INITIAL_CHECK" ]; then
        log "Found style issues in $FILE:\n$INITIAL_CHECK"
        
        # Run autopep8 with more aggressive fixes
        log "Running autopep8 on $FILE"
        autopep8 --in-place --aggressive --aggressive --max-line-length 120 "$FILE"
        
        # Check again
        FINAL_CHECK=$(pycodestyle "$FILE" 2>&1) || true
        if [ -n "$FINAL_CHECK" ]; then
            log "WARNING: Remaining style issues in $FILE after autopep8:\n$FINAL_CHECK"
        else
            log "Successfully fixed all style issues in $FILE"
        fi
    else
        log "No style issues found in $FILE"
    fi
    
    # Run additional fixes for common issues
    log "Running additional fixes for $FILE"
    
    # Fix trailing whitespace
    sed -i '' 's/[[:space:]]*$//' "$FILE"
    
    # Fix blank lines containing whitespace
    sed -i '' 's/^[[:space:]]*$//' "$FILE"
    
    # Fix multiple consecutive blank lines
    cat -s "$FILE" > "${FILE}.tmp" && mv "${FILE}.tmp" "$FILE"
done

log "Linting and fixing process completed in $TARGET_DIR"
log "Check $LOG_DIR for detailed logs"
