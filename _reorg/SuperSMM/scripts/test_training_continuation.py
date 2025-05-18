#!/usr/bin/env python
"""
Test script to verify that training continuation works properly.
This creates a mock training state file with a specified epoch number
and checks if we can read it correctly.
"""

import os
import sys
import argparse
from datetime import datetime, timedelta
import pandas as pd

# Parse command line arguments
parser = argparse.ArgumentParser(description="Test training continuation")
parser.add_argument("--epoch", type=int, default=25, help="Epoch number to set for testing")
parser.add_argument("--create-state", action="store_true", help="Create a mock training state file")
parser.add_argument("--verify", action="store_true", help="Verify training state and CSV files")
args = parser.parse_args()

# Get project directories
script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(script_dir)
tf_omr_dir = os.path.join(base_dir, "src", "tf-deep-omr")
checkpoint_dir = os.path.join(tf_omr_dir, "checkpoints")
logs_dir = os.path.join(base_dir, "logs")

# Ensure directories exist
os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(logs_dir, exist_ok=True)
os.makedirs(os.path.join(tf_omr_dir, "logs"), exist_ok=True)

# File paths
training_state_path = os.path.join(checkpoint_dir, "training_state.txt")
tf_csv_path = os.path.join(tf_omr_dir, "logs", "training_log.csv")
dashboard_csv_path = os.path.join(logs_dir, "training_run.csv")

def create_mock_state():
    """Create mock training state and CSV files"""
    print(f"Creating mock training state file with epoch {args.epoch}")
    
    # Create training state file
    with open(training_state_path, "w") as f:
        f.write(str(args.epoch))
    
    # Create a mock CSV file with the specified epoch
    total_epochs = 64000
    rows = []
    
    # Create rows for each epoch up to the specified one
    for epoch in range(1, args.epoch + 1):
        timestamp = (datetime.now() - timedelta(minutes=(args.epoch - epoch) * 10)).strftime('%Y-%m-%d %H:%M:%S')
        loss = 100.0 / (epoch + 1)  # Simulated decreasing loss
        
        row = {
            'timestamp': timestamp,
            'epoch': epoch,
            'loss': f"{loss:.4f}",
            'validation_error': "",
            'ser_percent': "",
            'epoch_time_sec': 30,
            'cumulative_time_sec': 30 * epoch,
            'memory_usage_mb': 1024,
            'batch_size': 32,
            'learning_rate': 0.001,
            'gpu_memory_mb': 2048,
            'checkpoint_path': f"checkpoints/model-{epoch}" if epoch % 10 == 0 else "",
            'validation_samples': "",
            'dataset_size': 10000,
            'remaining_epochs': total_epochs - epoch,
            'est_completion_time': (datetime.now() + timedelta(minutes=(total_epochs - epoch) * 10)).strftime('%Y-%m-%d %H:%M:%S')
        }
        rows.append(row)
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(rows)
    df.to_csv(tf_csv_path, index=False)
    df.to_csv(dashboard_csv_path, index=False)
    
    print(f"Created mock CSV files with {args.epoch} epochs")
    print(f"Training state file: {training_state_path}")
    print(f"TensorFlow CSV: {tf_csv_path}")
    print(f"Dashboard CSV: {dashboard_csv_path}")

def verify_state():
    """Verify that training state and CSV files exist and match"""
    problems = 0
    
    # Check training state file
    if os.path.exists(training_state_path):
        with open(training_state_path, "r") as f:
            state_epoch = int(f.read().strip())
        print(f"Training state file exists: Epoch {state_epoch}")
    else:
        print(f"ERROR: Training state file not found at {training_state_path}")
        problems += 1
    
    # Check TensorFlow CSV
    if os.path.exists(tf_csv_path):
        try:
            tf_df = pd.read_csv(tf_csv_path)
            tf_max_epoch = tf_df['epoch'].max()
            print(f"TensorFlow CSV exists: Max epoch {tf_max_epoch}")
        except Exception as e:
            print(f"ERROR: Could not read TensorFlow CSV: {e}")
            problems += 1
    else:
        print(f"ERROR: TensorFlow CSV not found at {tf_csv_path}")
        problems += 1
    
    # Check dashboard CSV
    if os.path.exists(dashboard_csv_path):
        try:
            dash_df = pd.read_csv(dashboard_csv_path)
            dash_max_epoch = dash_df['epoch'].max()
            print(f"Dashboard CSV exists: Max epoch {dash_max_epoch}")
        except Exception as e:
            print(f"ERROR: Could not read dashboard CSV: {e}")
            problems += 1
    else:
        print(f"ERROR: Dashboard CSV not found at {dashboard_csv_path}")
        problems += 1
    
    # Check if files are in sync
    if 'state_epoch' in locals() and 'tf_max_epoch' in locals() and 'dash_max_epoch' in locals():
        if state_epoch != tf_max_epoch:
            print(f"WARNING: Training state epoch ({state_epoch}) does not match TensorFlow CSV max epoch ({tf_max_epoch})")
            problems += 1
        
        if tf_max_epoch != dash_max_epoch:
            print(f"WARNING: TensorFlow CSV max epoch ({tf_max_epoch}) does not match dashboard CSV max epoch ({dash_max_epoch})")
            problems += 1
    
    # Summary
    if problems == 0:
        print("\nAll checks passed! Training continuation should work properly.")
    else:
        print(f"\nFound {problems} issues that might affect training continuation.")

# Main execution
if args.create_state:
    create_mock_state()

if args.verify:
    verify_state()

if not args.create_state and not args.verify:
    print("No action specified. Use --create-state to create mock files or --verify to check existing files.")
    print("Example: python test_training_continuation.py --create-state --epoch 25") 