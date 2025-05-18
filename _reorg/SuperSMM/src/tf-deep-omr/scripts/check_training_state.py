#!/usr/bin/env python
"""
Check and update the training state file based on latest checkpoint or CSV log.

This script:
1. Checks if the training_state.txt file exists
2. If not, tries to find the latest checkpoint
3. If no checkpoint, checks the training_log.csv file
4. Updates training_state.txt with the current epoch
"""

import os
import glob
import re
import csv
from datetime import datetime
from pathlib import Path

def find_latest_checkpoint_epoch(model_dir):
    """Find the latest checkpoint in the model directory and extract its epoch."""
    checkpoint_pattern = os.path.join(model_dir, "model-*.meta")
    checkpoints = glob.glob(checkpoint_pattern)
    
    if not checkpoints:
        print(f"No checkpoints found in {model_dir}")
        return None
    
    # Extract epoch numbers from checkpoint filenames
    checkpoint_epochs = []
    for ckpt in checkpoints:
        # Extract epoch number from filename (e.g., model-42.meta)
        match = re.search(r'model-(\d+)\.meta', ckpt)
        if match:
            epoch = int(match.group(1))
            checkpoint_epochs.append((epoch, ckpt))
    
    if not checkpoint_epochs:
        print("Could not extract epoch numbers from checkpoint filenames")
        return None
    
    # Find the highest epoch
    checkpoint_epochs.sort(reverse=True)
    latest_epoch, latest_checkpoint = checkpoint_epochs[0]
    
    print(f"Found latest checkpoint at epoch {latest_epoch}: {latest_checkpoint}")
    return latest_epoch

def find_latest_log_epoch(log_file):
    """Find the latest epoch from the training log CSV file."""
    if not os.path.exists(log_file):
        print(f"Log file not found: {log_file}")
        return None
    
    try:
        with open(log_file, 'r') as f:
            # Read CSV and get the last line
            csv_reader = csv.reader(f)
            # Skip header
            next(csv_reader)
            latest_epoch = 0
            for row in csv_reader:
                if len(row) > 1:  # Make sure row has enough data
                    # Epoch number should be in the second column (index 1)
                    try:
                        epoch = int(row[1])
                        latest_epoch = max(latest_epoch, epoch)
                    except (ValueError, IndexError):
                        pass
        
        if latest_epoch > 0:
            print(f"Found latest epoch in log: {latest_epoch}")
            return latest_epoch
        else:
            print("Could not extract valid epoch number from log file")
            return None
    except Exception as e:
        print(f"Error reading log file: {e}")
        return None

def update_training_state(state_file, epoch):
    """Update the training state file with the given epoch."""
    directory = os.path.dirname(state_file)
    os.makedirs(directory, exist_ok=True)
    
    with open(state_file, 'w') as f:
        f.write(str(epoch))
    
    print(f"Updated training state file: {state_file} with epoch {epoch}")

def main():
    # Define paths
    script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    project_dir = script_dir.parent
    
    model_dir = os.path.join(project_dir, "model", "primus_model")
    logs_dir = os.path.join(project_dir, "logs")
    state_file = os.path.join(model_dir, "training_state.txt")
    log_file = os.path.join(logs_dir, "training_log.csv")
    
    # Check if training state file already exists
    if os.path.exists(state_file):
        with open(state_file, 'r') as f:
            current_epoch = int(f.read().strip())
        print(f"Found existing training state: {current_epoch}")
        
        # If we have a confirmed last checkpoint at epoch 750, update the state
        if current_epoch < 750:
            print(f"Updating training state to latest known checkpoint: 750")
            update_training_state(state_file, 750)
        return
    
    # No state file, try to find latest checkpoint
    latest_checkpoint_epoch = find_latest_checkpoint_epoch(model_dir)
    
    if latest_checkpoint_epoch is not None:
        update_training_state(state_file, latest_checkpoint_epoch)
        return
    
    # No checkpoint, try to find latest epoch from log
    latest_log_epoch = find_latest_log_epoch(log_file)
    
    if latest_log_epoch is not None:
        update_training_state(state_file, latest_log_epoch)
        return
    
    # If we get here, we have no information, set to 750 (the last known epoch)
    print("No epoch information found, setting to 750 (last known epoch)")
    update_training_state(state_file, 750)

if __name__ == "__main__":
    main() 