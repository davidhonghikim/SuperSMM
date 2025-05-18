#!/usr/bin/env python
"""
Training monitoring script for OMR model training.
This script monitors the progress of training, provides real-time statistics,
and can generate visualizations of the training process.
"""

import os
import sys
import argparse
import json
import glob
import time
import datetime
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
import psutil

def load_training_stats(stats_file):
    """
    Load training statistics from a CSV file.
    
    Args:
        stats_file: Path to the CSV file containing training statistics
    
    Returns:
        pandas.DataFrame: Training statistics
    """
    if not os.path.exists(stats_file):
        print(f"ERROR: Stats file '{stats_file}' does not exist!")
        return None
    
    try:
        stats = pd.read_csv(stats_file)
        print(f"Loaded training statistics with {len(stats)} entries")
        return stats
    except Exception as e:
        print(f"ERROR: Failed to load stats file: {e}")
        return None

def load_system_stats(stats_dir):
    """
    Load system statistics from JSON files.
    
    Args:
        stats_dir: Directory containing system statistics JSON files
    
    Returns:
        list: System statistics as a list of dictionaries
    """
    if not os.path.exists(stats_dir):
        print(f"ERROR: Stats directory '{stats_dir}' does not exist!")
        return None
    
    # Find all system stats files
    stats_files = glob.glob(os.path.join(stats_dir, "system_stats_*.json"))
    if not stats_files:
        print(f"No system stats files found in '{stats_dir}'")
        return None
    
    # Load and parse each file
    system_stats = []
    for file_path in sorted(stats_files):
        try:
            with open(file_path, 'r') as f:
                stats = json.load(f)
                system_stats.append(stats)
        except Exception as e:
            print(f"WARNING: Failed to load stats file '{file_path}': {e}")
    
    print(f"Loaded {len(system_stats)} system stats entries")
    return system_stats

def plot_training_progress(stats_df, output_dir=None):
    """
    Plot training progress metrics.
    
    Args:
        stats_df: DataFrame containing training statistics
        output_dir: Directory to save plots (if None, displays plots)
    """
    if stats_df is None or len(stats_df) == 0:
        print("No training statistics available for plotting")
        return
    
    # Create output directory if needed
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Set up the figure
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Loss over epochs
    plt.subplot(2, 2, 1)
    plt.plot(stats_df['epoch'], stats_df['loss'], 'b-', label='Loss')
    plt.title('Loss vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    
    # Plot 2: Time per epoch
    plt.subplot(2, 2, 2)
    plt.plot(stats_df['epoch'], stats_df['epoch_time_sec'], 'g-', label='Epoch Time')
    plt.title('Time per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Time (seconds)')
    plt.grid(True)
    plt.legend()
    
    # Plot 3: Cumulative training time
    plt.subplot(2, 2, 3)
    plt.plot(stats_df['epoch'], stats_df['cumulative_time_min'], 'r-', label='Cumulative Time')
    plt.title('Cumulative Training Time')
    plt.xlabel('Epoch')
    plt.ylabel('Time (minutes)')
    plt.grid(True)
    plt.legend()
    
    # Plot 4: Memory usage if available
    if 'memory_usage_mb' in stats_df.columns:
        plt.subplot(2, 2, 4)
        plt.plot(stats_df['epoch'], stats_df['memory_usage_mb'], 'm-', label='Memory Usage')
        plt.title('Memory Usage During Training')
        plt.xlabel('Epoch')
        plt.ylabel('Memory (MB)')
        plt.grid(True)
        plt.legend()
    
    plt.tight_layout()
    
    # Save or display the plot
    if output_dir:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = os.path.join(output_dir, f"training_progress_{timestamp}.png")
        plt.savefig(plot_path)
        print(f"Saved training progress plot to: {plot_path}")
    else:
        plt.show()

def plot_system_stats(system_stats, output_dir=None):
    """
    Plot system resource usage during training.
    
    Args:
        system_stats: List of dictionaries containing system stats
        output_dir: Directory to save plots (if None, displays plots)
    """
    if not system_stats or len(system_stats) == 0:
        print("No system statistics available for plotting")
        return
    
    # Create output directory if needed
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Extract timestamps and convert to datetime
    timestamps = [stat.get('timestamp') for stat in system_stats]
    datetimes = [datetime.datetime.fromisoformat(ts) if ts else None for ts in timestamps]
    
    # Filter out None values
    valid_indices = [i for i, dt in enumerate(datetimes) if dt is not None]
    datetimes = [datetimes[i] for i in valid_indices]
    system_stats = [system_stats[i] for i in valid_indices]
    
    if not datetimes:
        print("No valid timestamps found in system stats")
        return
    
    # Convert to relative time in minutes from the first timestamp
    start_time = min(datetimes)
    relative_times = [(dt - start_time).total_seconds() / 60 for dt in datetimes]
    
    # Extract metrics
    cpu_usage = [stat.get('cpu', {}).get('percent', 0) for stat in system_stats]
    memory_usage = [stat.get('memory', {}).get('percent', 0) for stat in system_stats]
    disk_usage = [stat.get('disk', {}).get('percent', 0) for stat in system_stats]
    
    # GPU metrics if available
    gpu_usage = []
    gpu_memory = []
    for stat in system_stats:
        gpu = stat.get('gpu', {})
        if gpu and isinstance(gpu, dict):
            gpu_usage.append(gpu.get('utilization', 0))
            gpu_memory.append(gpu.get('memory_used_percent', 0))
        else:
            gpu_usage.append(0)
            gpu_memory.append(0)
    
    # Set up the figure
    plt.figure(figsize=(15, 10))
    
    # Plot 1: CPU Usage
    plt.subplot(2, 2, 1)
    plt.plot(relative_times, cpu_usage, 'b-', label='CPU Usage')
    plt.title('CPU Usage During Training')
    plt.xlabel('Time (minutes)')
    plt.ylabel('Usage (%)')
    plt.ylim(0, 100)
    plt.grid(True)
    plt.legend()
    
    # Plot 2: Memory Usage
    plt.subplot(2, 2, 2)
    plt.plot(relative_times, memory_usage, 'g-', label='Memory Usage')
    plt.title('Memory Usage During Training')
    plt.xlabel('Time (minutes)')
    plt.ylabel('Usage (%)')
    plt.ylim(0, 100)
    plt.grid(True)
    plt.legend()
    
    # Plot 3: Disk Usage
    plt.subplot(2, 2, 3)
    plt.plot(relative_times, disk_usage, 'r-', label='Disk Usage')
    plt.title('Disk Usage During Training')
    plt.xlabel('Time (minutes)')
    plt.ylabel('Usage (%)')
    plt.ylim(0, 100)
    plt.grid(True)
    plt.legend()
    
    # Plot 4: GPU Usage if available
    if any(gpu_usage) or any(gpu_memory):
        plt.subplot(2, 2, 4)
        plt.plot(relative_times, gpu_usage, 'm-', label='GPU Utilization')
        plt.plot(relative_times, gpu_memory, 'c-', label='GPU Memory')
        plt.title('GPU Usage During Training')
        plt.xlabel('Time (minutes)')
        plt.ylabel('Usage (%)')
        plt.ylim(0, 100)
        plt.grid(True)
        plt.legend()
    
    plt.tight_layout()
    
    # Save or display the plot
    if output_dir:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = os.path.join(output_dir, f"system_stats_{timestamp}.png")
        plt.savefig(plot_path)
        print(f"Saved system stats plot to: {plot_path}")
    else:
        plt.show()

def estimate_completion(stats_df, total_epochs):
    """
    Estimate training completion time based on current progress.
    
    Args:
        stats_df: DataFrame containing training statistics
        total_epochs: Total number of epochs for training
    
    Returns:
        dict: Completion estimates
    """
    if stats_df is None or len(stats_df) < 2:
        print("Insufficient data to estimate completion time")
        return None
    
    # Get the latest epoch
    latest_epoch = stats_df['epoch'].max()
    
    # Calculate average time per epoch (from the last 5 epochs if available)
    recent_stats = stats_df.tail(min(5, len(stats_df)))
    avg_epoch_time = recent_stats['epoch_time_sec'].mean()
    
    # Estimate remaining time
    remaining_epochs = total_epochs - latest_epoch
    estimated_remaining_seconds = remaining_epochs * avg_epoch_time
    
    # Convert to human-readable format
    remaining_time = str(datetime.timedelta(seconds=int(estimated_remaining_seconds)))
    
    # Calculate progress percentage
    progress_percent = (latest_epoch / total_epochs) * 100
    
    # Estimate completion timestamp
    now = datetime.datetime.now()
    estimated_completion = now + datetime.timedelta(seconds=estimated_remaining_seconds)
    
    return {
        'current_epoch': latest_epoch,
        'total_epochs': total_epochs,
        'progress_percent': progress_percent,
        'avg_epoch_time_sec': avg_epoch_time,
        'remaining_time': remaining_time,
        'estimated_completion': estimated_completion.strftime("%Y-%m-%d %H:%M:%S")
    }

def get_gpu_memory():
    """Get GPU memory usage if available"""
    try:
        import tensorflow as tf
        if tf.test.is_built_with_cuda():
            return tf.config.experimental.get_memory_info('GPU:0')['current'] / (1024 * 1024)  # Convert to MB
    except:
        pass
    return 0

def monitor_training(log_file, refresh_interval=2):
    """
    Monitor training progress in real-time.
    
    Args:
        log_file: Path to the training log CSV file
        refresh_interval: How often to update stats (in seconds)
    """
    print(f"Starting training monitor...")
    print(f"Watching log file: {log_file}")
    
    last_modified = 0
    while True:
        try:
            # Check if log file exists and has been modified
            if os.path.exists(log_file):
                current_modified = os.path.getmtime(log_file)
                if current_modified > last_modified:
                    # Read and process log file
                    df = pd.read_csv(log_file)
                    if len(df) > 0:
                        last_row = df.iloc[-1]
                        
                        # Calculate stats
                        current_epoch = last_row['epoch']
                        total_epochs = current_epoch + last_row.get('remaining_epochs', 0)
                        progress = (current_epoch / total_epochs) * 100 if total_epochs > 0 else 0
                        
                        # Get system stats
                        process = psutil.Process()
                        cpu_percent = process.cpu_percent()
                        memory_mb = process.memory_info().rss / (1024 * 1024)
                        gpu_memory = get_gpu_memory()
                        
                        # Clear screen and print stats
                        os.system('clear' if os.name == 'posix' else 'cls')
                        print("=" * 50)
                        print(f"OMR Training Monitor - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                        print("=" * 50)
                        print(f"\nProgress: {progress:.1f}% (Epoch {current_epoch}/{total_epochs})")
                        print(f"Current Loss: {last_row['loss']:.4f}")
                        if 'validation_error' in last_row and pd.notna(last_row['validation_error']):
                            print(f"Validation Error: {last_row['validation_error']:.4f}")
                        if 'ser_percent' in last_row and pd.notna(last_row['ser_percent']):
                            print(f"Symbol Error Rate: {last_row['ser_percent']:.2f}%")
                        print(f"\nSystem Stats:")
                        print(f"CPU Usage: {cpu_percent:.1f}%")
                        print(f"Memory Usage: {memory_mb:.1f} MB")
                        print(f"GPU Memory: {gpu_memory:.1f} MB")
                        
                        # Update last modified time
                        last_modified = current_modified
                
            time.sleep(refresh_interval)
            
        except KeyboardInterrupt:
            print("\nMonitoring stopped by user")
            break
        except Exception as e:
            print(f"Error monitoring training: {e}")
            time.sleep(refresh_interval)

def main():
    parser = argparse.ArgumentParser(description="Monitor OMR training progress")
    parser.add_argument("--log-file", type=str, required=True, help="Path to training log CSV file")
    parser.add_argument("--refresh", type=int, default=2, help="Refresh interval in seconds")
    args = parser.parse_args()
    
    monitor_training(args.log_file, args.refresh)

if __name__ == "__main__":
    main()
