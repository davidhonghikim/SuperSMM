#!/bin/bash
# Script to sync training logs from tf-deep-omr to the main logs directory

# Navigate to the script directory
cd "$(dirname "$0")" || exit

# Define paths
ROOT_DIR="$(dirname "$(dirname "$PWD")")"
TF_LOGS_DIR="../tf-deep-omr/logs"
ROOT_LOGS_DIR="$ROOT_DIR/logs"

# Create root logs directory if it doesn't exist
mkdir -p "$ROOT_LOGS_DIR"

# Check if tf-deep-omr logs exist
if [ -f "$TF_LOGS_DIR/training_log.csv" ]; then
    echo "Found training log at: $TF_LOGS_DIR/training_log.csv"
    
    # Check if the log has content (more than header line)
    if [ "$(wc -l < "$TF_LOGS_DIR/training_log.csv")" -gt 1 ]; then
        echo "Copying training log to $ROOT_LOGS_DIR/training_log.csv"
        cp "$TF_LOGS_DIR/training_log.csv" "$ROOT_LOGS_DIR/training_log.csv"
        
        echo "Creating backup in $ROOT_LOGS_DIR/training_run.csv"
        cp "$TF_LOGS_DIR/training_log.csv" "$ROOT_LOGS_DIR/training_run.csv"
        
        echo "Logs synchronized successfully"
    else
        echo "Warning: Training log exists but appears to be empty (header only)"
    fi
else
    echo "Warning: No training log found at $TF_LOGS_DIR/training_log.csv"
    
    # Check if the root log exists
    if [ -f "$ROOT_LOGS_DIR/training_log.csv" ]; then
        echo "Found training log at: $ROOT_LOGS_DIR/training_log.csv"
        
        # Copy back to tf-deep-omr if needed
        if [ ! -f "$TF_LOGS_DIR/training_log.csv" ] || [ "$(wc -l < "$ROOT_LOGS_DIR/training_log.csv")" -gt "$(wc -l < "$TF_LOGS_DIR/training_log.csv" 2>/dev/null || echo 0)" ]; then
            echo "Copying training log to $TF_LOGS_DIR/training_log.csv"
            mkdir -p "$TF_LOGS_DIR"
            cp "$ROOT_LOGS_DIR/training_log.csv" "$TF_LOGS_DIR/training_log.csv"
            echo "Logs synchronized successfully"
        fi
    else
        echo "Error: No training logs found in either location"
        exit 1
    fi
fi

echo "Done" 