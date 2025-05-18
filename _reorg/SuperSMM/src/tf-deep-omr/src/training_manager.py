#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Training Manager for OMR model training.
This script helps manage training sessions, find checkpoints, and analyze training logs.
"""

import os
import json
import argparse
import glob
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

def find_latest_checkpoint(model_dir):
    """
    Find the latest checkpoint in the model directory.
    
    Args:
        model_dir (str): Path to the model directory
        
    Returns:
        str: Path to the latest checkpoint or None if not found
    """
    # Look for checkpoint files
    checkpoint_files = glob.glob(os.path.join(model_dir, "*.meta"))
    
    if not checkpoint_files:
        return None
    
    # Extract epoch numbers and find the latest
    latest_epoch = -1
    latest_checkpoint = None
    
    for checkpoint in checkpoint_files:
        # Extract epoch number from filename
        try:
            base = os.path.basename(checkpoint)
            if '-' in base:
                epoch = int(base.split('-')[-1].split('.')[0])
                if epoch > latest_epoch:
                    latest_epoch = epoch
                    latest_checkpoint = checkpoint[:-5]  # Remove .meta extension
        except:
            continue
    
    return latest_checkpoint

def analyze_training_log(log_file):
    """
    Analyze training log and print statistics.
    
    Args:
        log_file (str): Path to the training log file
        
    Returns:
        dict: Training statistics
    """
    if not os.path.exists(log_file):
        print(f"Error: Log file '{log_file}' not found.")
        return None
    
    try:
        with open(log_file, 'r') as f:
            training_data = json.load(f)
    except Exception as e:
        print(f"Error loading log file: {e}")
        return None
    
    # Extract data
    losses = training_data.get('losses', [])
    validation_metrics = training_data.get('validation_metrics', [])
    
    if not losses:
        print("No training data found in log file.")
        return None
    
    # Calculate statistics
    epochs_completed = len(losses)
    latest_loss = losses[-1]['loss'] if losses else None
    avg_loss = np.mean([l['loss'] for l in losses]) if losses else None
    
    # Get validation metrics
    latest_val = validation_metrics[-1] if validation_metrics else None
    latest_ser = latest_val['metrics'][1] if latest_val and len(latest_val['metrics']) > 1 else None
    
    stats = {
        'epochs_completed': epochs_completed,
        'latest_loss': latest_loss,
        'avg_loss': avg_loss,
        'latest_ser': latest_ser
    }
    
    # Print summary
    print("\n=== Training Analysis ===")
    print(f"Epochs completed: {epochs_completed}")
    print(f"Latest loss: {latest_loss:.4f}" if latest_loss is not None else "Latest loss: N/A")
    print(f"Average loss: {avg_loss:.4f}" if avg_loss is not None else "Average loss: N/A")
    print(f"Latest Symbol Error Rate: {latest_ser:.2f}%" if latest_ser is not None else "Latest SER: N/A")
    
    return stats

def plot_training_progress(log_file, output_file=None):
    """
    Plot training progress from log file.
    
    Args:
        log_file (str): Path to the training log file
        output_file (str, optional): Path to save the plot
    """
    if not os.path.exists(log_file):
        print(f"Error: Log file '{log_file}' not found.")
        return
    
    try:
        with open(log_file, 'r') as f:
            training_data = json.load(f)
    except Exception as e:
        print(f"Error loading log file: {e}")
        return
    
    # Extract data
    losses = training_data.get('losses', [])
    validation_metrics = training_data.get('validation_metrics', [])
    
    if not losses:
        print("No training data found in log file.")
        return
    
    # Prepare data for plotting
    epochs = [l['epoch'] for l in losses]
    loss_values = [l['loss'] for l in losses]
    
    val_epochs = [v['epoch'] for v in validation_metrics]
    val_ser = [v['metrics'][1] if len(v['metrics']) > 1 else None for v in validation_metrics]
    val_ser = [v for v in val_ser if v is not None]
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    
    # Plot loss
    ax1.plot(epochs, loss_values, 'b-')
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True)
    
    # Plot SER if available
    if val_epochs and val_ser:
        ax2.plot(val_epochs, val_ser, 'r-')
        ax2.set_title('Validation Symbol Error Rate')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('SER (%)')
        ax2.grid(True)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file)
        print(f"Plot saved to {output_file}")
    else:
        plt.show()

def generate_resume_command(model_dir, corpus, set_file, voc_file, semantic=False, batch_size=50, total_epochs=300):
    """
    Generate command to resume training.
    
    Args:
        model_dir (str): Path to the model directory
        corpus (str): Path to the corpus
        set_file (str): Path to the set file
        voc_file (str): Path to the vocabulary file
        semantic (bool): Whether to use semantic encoding
        batch_size (int): Number of epochs per batch
        total_epochs (int): Total number of epochs to train
        
    Returns:
        str: Command to resume training
    """
    # Find latest checkpoint
    checkpoint = find_latest_checkpoint(model_dir)
    
    if not checkpoint:
        print("No checkpoint found. Cannot generate resume command.")
        return None
    
    # Generate command
    semantic_flag = "-semantic" if semantic else ""
    command = (f"python batch_training.py {semantic_flag} "
               f"-corpus {corpus} -set {set_file} -vocabulary {voc_file} "
               f"-save_model {model_dir}/model -checkpoint {checkpoint} "
               f"-batch_size {batch_size} -total_epochs {total_epochs} "
               f"-log_file {model_dir}/training_log.json")
    
    return command

def main():
    """
    Main function.
    """
    parser = argparse.ArgumentParser(description='Training Manager for OMR model training')
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Find checkpoint command
    find_parser = subparsers.add_parser('find_checkpoint', help='Find latest checkpoint')
    find_parser.add_argument('-model_dir', required=True, help='Path to the model directory')
    find_parser.add_argument('-quiet', action='store_true', help='Only output the checkpoint path without additional text')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze training log')
    analyze_parser.add_argument('-log_file', required=True, help='Path to the training log file')
    
    # Plot command
    plot_parser = subparsers.add_parser('plot', help='Plot training progress')
    plot_parser.add_argument('-log_file', required=True, help='Path to the training log file')
    plot_parser.add_argument('-output', help='Path to save the plot')
    
    # Resume command
    resume_parser = subparsers.add_parser('resume', help='Generate command to resume training')
    resume_parser.add_argument('-model_dir', required=True, help='Path to the model directory')
    resume_parser.add_argument('-corpus', required=True, help='Path to the corpus')
    resume_parser.add_argument('-set', required=True, help='Path to the set file')
    resume_parser.add_argument('-vocabulary', required=True, help='Path to the vocabulary file')
    resume_parser.add_argument('-semantic', action='store_true', help='Use semantic encoding')
    resume_parser.add_argument('-batch_size', type=int, default=50, help='Number of epochs per batch')
    resume_parser.add_argument('-total_epochs', type=int, default=300, help='Total number of epochs to train')
    
    args = parser.parse_args()
    
    if args.command == 'find_checkpoint':
        checkpoint = find_latest_checkpoint(args.model_dir)
        if checkpoint:
            if args.quiet:
                print(checkpoint)
            else:
                print(f"Latest checkpoint: {checkpoint}")
        else:
            if not args.quiet:
                print("No checkpoint found.")
    
    elif args.command == 'analyze':
        analyze_training_log(args.log_file)
    
    elif args.command == 'plot':
        plot_training_progress(args.log_file, args.output)
    
    elif args.command == 'resume':
        command = generate_resume_command(
            args.model_dir,
            args.corpus,
            args.set,
            args.vocabulary,
            args.semantic,
            args.batch_size,
            args.total_epochs
        )
        
        if command:
            print("\n=== Resume Training Command ===")
            print(command)
            print("\nCopy and run this command to resume training.")

if __name__ == "__main__":
    main()
