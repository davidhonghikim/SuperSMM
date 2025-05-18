#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Utility script to plot training progress from log files.
This helps visualize the training loss and validation metrics over time.
"""

import argparse
import re
import matplotlib.pyplot as plt
import numpy as np
import os

def parse_log_file(log_file):
    """
    Parse the training log file to extract epoch, loss, and validation metrics.
    
    Args:
        log_file (str): Path to the log file
        
    Returns:
        dict: Dictionary containing epochs, losses, and validation metrics
    """
    if not os.path.exists(log_file):
        print(f"Error: Log file '{log_file}' not found.")
        return None
        
    with open(log_file, 'r') as f:
        content = f.read()
    
    # Extract epoch and loss information
    epoch_pattern = r"Epoch (\d+)/\d+ - Loss: ([\d\.]+)"
    epoch_matches = re.findall(epoch_pattern, content)
    
    # Extract validation information
    val_pattern = r"\[Epoch (\d+)\] ([\d\.]+) \(([\d\.]+)% SER\) from (\d+) samples"
    val_matches = re.findall(val_pattern, content)
    
    # Process the data
    epochs = [int(match[0]) for match in epoch_matches]
    losses = [float(match[1]) for match in epoch_matches]
    
    val_epochs = [int(match[0]) for match in val_matches]
    val_metrics = [float(match[1]) for match in val_matches]
    val_ser = [float(match[2]) for match in val_matches]
    val_samples = [int(match[3]) for match in val_matches]
    
    return {
        'epochs': epochs,
        'losses': losses,
        'val_epochs': val_epochs,
        'val_metrics': val_metrics,
        'val_ser': val_ser,
        'val_samples': val_samples
    }

def plot_training_progress(data, output_file=None):
    """
    Plot the training progress.
    
    Args:
        data (dict): Dictionary containing training data
        output_file (str, optional): Path to save the plot
    """
    if not data:
        print("No data to plot.")
        return
    
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    
    # Plot training loss
    ax1.plot(data['epochs'], data['losses'], 'b-', label='Training Loss')
    ax1.set_title('Training Loss Over Time')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True)
    ax1.legend()
    
    # Plot validation metrics
    if data['val_epochs']:
        ax2.plot(data['val_epochs'], data['val_ser'], 'r-', label='Symbol Error Rate (%)')
        ax2.set_title('Validation Symbol Error Rate')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('SER (%)')
        ax2.grid(True)
        ax2.legend()
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file)
        print(f"Plot saved to {output_file}")
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description='Plot training progress from log file.')
    parser.add_argument('-log', dest='log_file', type=str, required=True, 
                        help='Path to the training log file.')
    parser.add_argument('-output', dest='output_file', type=str, default=None,
                        help='Path to save the output plot (optional).')
    args = parser.parse_args()
    
    data = parse_log_file(args.log_file)
    plot_training_progress(data, args.output_file)

if __name__ == "__main__":
    main()
