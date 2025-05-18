#!/usr/bin/env python
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Path to the original CSV file
csv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs', 'training_run.csv')

# Check if the file exists
if not os.path.exists(csv_path):
    print(f"Error: File not found at {csv_path}")
    exit(1)

# Read the CSV file
try:
    df = pd.read_csv(csv_path)
    print(f"Original columns: {', '.join(df.columns)}")
except Exception as e:
    print(f"Error reading CSV file: {e}")
    exit(1)

# Add missing columns if they don't exist
required_columns = [
    'timestamp', 'epoch', 'loss', 'validation_error', 'ser_percent',
    'batch_size', 'dataset_size', 'memory_usage_mb', 'gpu_memory_mb', 'checkpoint_path',
    'epoch_time_sec', 'cumulative_time_sec', 'remaining_epochs', 'est_completion_time'
]

# Add missing columns with default values
for col in required_columns:
    if col not in df.columns:
        print(f"Adding missing column: {col}")
        if col == 'batch_size':
            df[col] = 32  # Default batch size
        elif col == 'dataset_size':
            df[col] = 10000  # Default dataset size
        elif col == 'memory_usage_mb':
            df[col] = 1024  # Default memory usage
        elif col == 'gpu_memory_mb':
            df[col] = 2048  # Default GPU memory
        elif col == 'checkpoint_path':
            df[col] = 'checkpoints/latest'  # Default checkpoint path
        elif col == 'epoch_time_sec':
            # Calculate epoch time from timestamps if possible
            if 'timestamp' in df.columns and len(df) > 1:
                timestamps = pd.to_datetime(df['timestamp'])
                df[col] = (timestamps.shift(-1) - timestamps).dt.total_seconds().fillna(10)
            else:
                df[col] = 10  # Default 10 seconds per epoch
        elif col == 'cumulative_time_sec':
            # Calculate cumulative time based on epoch times
            if 'epoch_time_sec' in df.columns:
                df[col] = df['epoch_time_sec'].cumsum()
            else:
                # Estimate based on epochs
                df[col] = df.index * 10  # 10 seconds per epoch
        elif col == 'remaining_epochs':
            # Calculate remaining epochs
            max_epoch = df['epoch'].max() if 'epoch' in df.columns else 0
            total_epochs = 64000  # Updated to match the full dataset size
            df[col] = total_epochs - df['epoch']
        elif col == 'est_completion_time':
            # Estimate completion time
            avg_epoch_time = df['epoch_time_sec'].tail(min(5, len(df))).mean()  # Use last 5 epochs for better estimate
            if pd.isna(avg_epoch_time) or avg_epoch_time <= 0:
                avg_epoch_time = 10  # Default to 10 seconds if no valid data
            now = datetime.now()
            completion_times = []
            for _, row in df.iterrows():
                remaining = row['remaining_epochs']
                try:
                    remaining = float(remaining)
                    if pd.isna(remaining) or remaining < 0:
                        remaining = 0
                    completion_time = (now + timedelta(seconds=remaining * avg_epoch_time)).strftime('%Y-%m-%d %H:%M:%S')
                except (ValueError, TypeError):
                    completion_time = now.strftime('%Y-%m-%d %H:%M:%S')
                completion_times.append(completion_time)
            df[col] = completion_times
        else:
            df[col] = ''  # Default empty string for other columns

# Ensure and fix data types
for col in df.columns:
    if col in ['epoch', 'batch_size', 'dataset_size']:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
    elif col in ['loss', 'validation_error', 'ser_percent', 'memory_usage_mb', 'gpu_memory_mb', 'epoch_time_sec', 'cumulative_time_sec', 'remaining_epochs']:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

# Save the updated CSV
try:
    # Create backup
    backup_path = csv_path + '.bak'
    if os.path.exists(csv_path):
        print(f"Creating backup at {backup_path}")
        with open(csv_path, 'r') as src, open(backup_path, 'w') as dst:
            dst.write(src.read())
    
    # Save the updated file
    df.to_csv(csv_path, index=False)
    print(f"Updated CSV saved to {csv_path}")
    print(f"New columns: {', '.join(df.columns)}")
except Exception as e:
    print(f"Error saving CSV file: {e}")
    exit(1) 