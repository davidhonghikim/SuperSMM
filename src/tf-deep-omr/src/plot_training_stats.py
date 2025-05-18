#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Plot training statistics from the CSV file generated during batch training.
"""

import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta

def plot_training_stats(stats_file, output_file=None, show=True):
    """
    Plot training statistics from CSV file.
    
    Args:
        stats_file: Path to the CSV file with training stats
        output_file: Path to save the plot (optional)
        show: Whether to show the plot
    """
    # Check if file exists
    if not os.path.exists(stats_file):
        print(f"Error: Stats file '{stats_file}' does not exist.")
        return
    
    # Read the CSV file
    try:
        df = pd.read_csv(stats_file)
    except Exception as e:
        print(f"Error reading stats file: {e}")
        return
    
    # Create figure with subplots
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training Statistics', fontsize=16)
    
    # Plot loss
    axs[0, 0].plot(df['epoch'], df['loss'], 'b-')
    axs[0, 0].set_title('Loss vs Epoch')
    axs[0, 0].set_xlabel('Epoch')
    axs[0, 0].set_ylabel('Loss')
    axs[0, 0].grid(True)
    
    # Plot epoch time
    axs[0, 1].plot(df['epoch'], df['epoch_time_sec'], 'g-')
    axs[0, 1].set_title('Epoch Time')
    axs[0, 1].set_xlabel('Epoch')
    axs[0, 1].set_ylabel('Time (seconds)')
    axs[0, 1].grid(True)
    
    # Plot cumulative time
    axs[1, 0].plot(df['epoch'], df['cumulative_time_sec'] / 60, 'r-')
    axs[1, 0].set_title('Cumulative Training Time')
    axs[1, 0].set_xlabel('Epoch')
    axs[1, 0].set_ylabel('Time (minutes)')
    axs[1, 0].grid(True)
    
    # Plot memory usage
    if 'memory_usage_mb' in df.columns:
        axs[1, 1].plot(df['epoch'], df['memory_usage_mb'], 'm-')
        axs[1, 1].set_title('Memory Usage')
        axs[1, 1].set_xlabel('Epoch')
        axs[1, 1].set_ylabel('Memory (MB)')
        axs[1, 1].grid(True)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Add training summary
    if len(df) > 0:
        total_epochs = len(df)
        total_time_min = df['cumulative_time_sec'].iloc[-1] / 60
        avg_epoch_time = df['epoch_time_sec'].mean()
        last_loss = df['loss'].iloc[-1]
        
        summary = (
            f"Total epochs: {total_epochs}\n"
            f"Total training time: {total_time_min:.2f} minutes\n"
            f"Average epoch time: {avg_epoch_time:.2f} seconds\n"
            f"Final loss: {last_loss:.6f}"
        )
        
        fig.text(0.5, 0.01, summary, ha='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
    
    # Save plot if output file is specified
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_file}")
    
    # Show plot if requested
    if show:
        plt.show()
    else:
        plt.close()

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Plot training statistics')
    parser.add_argument('-stats', required=True, help='Path to the CSV file with training stats')
    parser.add_argument('-output', help='Path to save the plot')
    parser.add_argument('-no_show', action='store_true', help='Do not show the plot')
    
    args = parser.parse_args()
    
    plot_training_stats(args.stats, args.output, not args.no_show)

if __name__ == "__main__":
    main()
