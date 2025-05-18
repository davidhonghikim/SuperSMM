#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Auto-recovery training script for the OMR model.
This script provides a robust training system that automatically recovers from crashes
and continues training from the last saved checkpoint.
"""

import os
import sys
import time
import signal
import subprocess
import argparse
import json
import glob
import smtplib
import psutil
import platform
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta

def find_latest_checkpoint(model_dir):
    """
    Find the latest checkpoint in the model directory.
    
    Args:
        model_dir (str): Path to the model directory
        
    Returns:
        tuple: (checkpoint_path, epoch_number) or (None, 0) if not found
    """
    # Look for checkpoint files
    checkpoint_files = glob.glob(os.path.join(model_dir, "*.meta"))
    
    if not checkpoint_files:
        return None, 0
    
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
    
    return latest_checkpoint, latest_epoch

def create_recovery_file(args, model_dir):
    """
    Create a recovery file with training parameters.
    
    Args:
        args: Command line arguments
        model_dir: Model directory path
    """
    recovery_file = os.path.join(model_dir, "recovery.json")
    
    recovery_data = {
        "corpus": args.corpus,
        "set": args.set,
        "vocabulary": args.vocabulary,
        "save_model": args.save_model,
        "semantic": args.semantic,
        "batch_size": args.batch_size,
        "total_epochs": args.total_epochs,
        "last_update": datetime.now().isoformat()
    }
    
    with open(recovery_file, 'w') as f:
        json.dump(recovery_data, f, indent=2)
    
    print(f"Recovery file created: {recovery_file}")

def load_recovery_file(model_dir):
    """
    Load recovery file if it exists.
    
    Args:
        model_dir: Model directory path
        
    Returns:
        dict: Recovery data or None if not found
    """
    recovery_file = os.path.join(model_dir, "recovery.json")
    
    if not os.path.exists(recovery_file):
        return None
    
    try:
        with open(recovery_file, 'r') as f:
            recovery_data = json.load(f)
        
        print(f"Recovery file found: {recovery_file}")
        return recovery_data
    except Exception as e:
        print(f"Error loading recovery file: {e}")
        return None

def run_training_with_recovery(args):
    """
    Run training with automatic recovery.
    
    Args:
        args: Command line arguments
    """
    model_dir = os.path.dirname(args.save_model)
    os.makedirs(model_dir, exist_ok=True)
    
    # Create recovery file
    create_recovery_file(args, model_dir)
    
    # Find latest checkpoint
    checkpoint_path, latest_epoch = find_latest_checkpoint(model_dir)
    
    # Create stats directory for monitoring
    stats_dir = os.path.join(model_dir, "stats")
    os.makedirs(stats_dir, exist_ok=True)
    
    # Initialize monitoring variables
    start_time = time.time()
    last_notification_time = start_time
    last_checkpoint_time = start_time
    recovery_attempts = 0
    max_recovery_attempts = 10  # Maximum consecutive recovery attempts
    monitoring_interval = 300  # 5 minutes
    notification_interval = 3600  # 1 hour
    
    # Send start notification
    if hasattr(args, 'email_to') and args.email_to:
        send_notification(
            "Training Started", 
            f"OMR model training has started.\n\n"
            f"Model directory: {model_dir}\n"
            f"Starting from epoch: {latest_epoch}\n"
            f"Target epochs: {args.total_epochs}\n"
            f"Batch size: {args.batch_size}\n",
            args
        )
    
    # Main training loop with recovery
    while True:
        # Check if training is complete
        if latest_epoch >= args.total_epochs:
            print(f"Training completed ({latest_epoch}/{args.total_epochs} epochs)")
            
            # Send completion notification
            if hasattr(args, 'email_to') and args.email_to:
                total_time = time.time() - start_time
                hours = total_time // 3600
                minutes = (total_time % 3600) // 60
                
                send_notification(
                    "Training Completed", 
                    f"OMR model training has completed successfully!\n\n"
                    f"Total epochs: {latest_epoch}\n"
                    f"Total training time: {int(hours)} hours, {int(minutes)} minutes\n"
                    f"Model saved to: {args.save_model}\n",
                    args
                )
            break
        
        # Log current system state
        current_time = time.time()
        elapsed_time = current_time - start_time
        hours = elapsed_time // 3600
        minutes = (elapsed_time % 3600) // 60
        
        print(f"\n=== Starting/Resuming training from epoch {latest_epoch} ===")
        print(f"Target: {args.total_epochs} epochs")
        print(f"Progress: {latest_epoch}/{args.total_epochs} ({latest_epoch/args.total_epochs*100:.1f}%)")
        print(f"Elapsed time: {int(hours)} hours, {int(minutes)} minutes")
        print(f"Recovery attempts: {recovery_attempts}")
        
        # Check system resources
        system_info = get_system_info()
        print("\nSystem Status:")
        print(f"CPU Usage: {system_info['cpu']['percent']}%")
        print(f"Memory: {system_info['memory']['used_gb']:.1f}/{system_info['memory']['total_gb']:.1f} GB ({system_info['memory']['percent']}%)")
        print(f"Disk: {system_info['disk']['used_gb']:.1f}/{system_info['disk']['total_gb']:.1f} GB ({system_info['disk']['percent']}%)")
        
        # Save system info to stats directory
        stats_file = os.path.join(stats_dir, f"system_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(stats_file, 'w') as f:
            json.dump(system_info, f, indent=2)
        
        # Send periodic notification
        if hasattr(args, 'email_to') and args.email_to and (current_time - last_notification_time) > notification_interval:
            estimated_remaining = (args.total_epochs - latest_epoch) * (elapsed_time / max(1, latest_epoch))
            remaining_hours = estimated_remaining // 3600
            remaining_minutes = (estimated_remaining % 3600) // 60
            
            send_notification(
                "Training Progress Update", 
                f"OMR model training progress update:\n\n"
                f"Current epoch: {latest_epoch}/{args.total_epochs} ({latest_epoch/args.total_epochs*100:.1f}%)\n"
                f"Elapsed time: {int(hours)} hours, {int(minutes)} minutes\n"
                f"Estimated remaining time: {int(remaining_hours)} hours, {int(remaining_minutes)} minutes\n"
                f"Recovery attempts: {recovery_attempts}\n",
                args
            )
            last_notification_time = current_time
        
        # Prepare command
        cmd = [
            "python", "batch_training.py",
            "-corpus", args.corpus,
            "-set", args.set,
            "-vocabulary", args.vocabulary,
            "-save_model", args.save_model,
            "-total_epochs", str(args.total_epochs),
            "-batch_size", str(args.batch_size)
        ]
        
        if args.semantic:
            cmd.append("-semantic")
        
        if checkpoint_path:
            cmd.append("-auto_resume")
        
        # Run training process
        print("\nRunning command:", " ".join(cmd))
        print("\n" + "="*60)
        print("TRAINING OUTPUT")
        print("="*60)
        
        process_start_time = time.time()
        process_success = False
        
        try:
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
            
            # Monitor and display output
            while True:
                # Check if we need to collect system stats
                current_time = time.time()
                if (current_time - last_checkpoint_time) > monitoring_interval:
                    # Save system stats
                    system_info = get_system_info()
                    stats_file = os.path.join(stats_dir, f"system_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
                    with open(stats_file, 'w') as f:
                        json.dump(system_info, f, indent=2)
                    last_checkpoint_time = current_time
                
                # Process output
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    print(output.strip())
            
            # Check if process completed successfully
            if process.returncode == 0:
                print("\nTraining process completed successfully.")
                process_success = True
                recovery_attempts = 0  # Reset recovery attempts counter
            else:
                print(f"\nTraining process exited with code {process.returncode}")
                print("Will attempt to recover and continue...")
                recovery_attempts += 1
                
                # Send notification about failure
                if hasattr(args, 'email_to') and args.email_to:
                    send_notification(
                        "Training Process Failed", 
                        f"OMR model training process failed with exit code {process.returncode}.\n\n"
                        f"Current epoch: {latest_epoch}/{args.total_epochs}\n"
                        f"Recovery attempt: {recovery_attempts}/{max_recovery_attempts}\n"
                        f"The system will attempt to recover automatically.\n",
                        args
                    )
        
        except KeyboardInterrupt:
            print("\nTraining interrupted by user.")
            print("You can resume later by running this script again.")
            
            # Send notification about interruption
            if hasattr(args, 'email_to') and args.email_to:
                send_notification(
                    "Training Interrupted", 
                    f"OMR model training was interrupted by user.\n\n"
                    f"Current epoch: {latest_epoch}/{args.total_epochs}\n"
                    f"You can resume training by running the script again.\n",
                    args
                )
            break
            
        except Exception as e:
            print(f"\nError during training: {e}")
            print("Will attempt to recover and continue...")
            recovery_attempts += 1
            
            # Send notification about error
            if hasattr(args, 'email_to') and args.email_to:
                send_notification(
                    "Training Error", 
                    f"OMR model training encountered an error: {str(e)}\n\n"
                    f"Current epoch: {latest_epoch}/{args.total_epochs}\n"
                    f"Recovery attempt: {recovery_attempts}/{max_recovery_attempts}\n"
                    f"The system will attempt to recover automatically.\n",
                    args
                )
        
        # Check if we've exceeded maximum recovery attempts
        if recovery_attempts >= max_recovery_attempts:
            print(f"\nExceeded maximum recovery attempts ({max_recovery_attempts}). Stopping training.")
            
            # Send notification about max recovery attempts
            if hasattr(args, 'email_to') and args.email_to:
                send_notification(
                    "Training Recovery Failed", 
                    f"OMR model training has exceeded maximum recovery attempts ({max_recovery_attempts}).\n\n"
                    f"Current epoch: {latest_epoch}/{args.total_epochs}\n"
                    f"Training has been stopped. Manual intervention is required.\n",
                    args
                )
            break
        
        # Wait before attempting recovery
        wait_time = min(30, recovery_attempts * 5)  # Exponential backoff, max 30 seconds
        print(f"Waiting {wait_time} seconds before continuing...")
        time.sleep(wait_time)
        
        # Find latest checkpoint after training
        new_checkpoint_path, new_latest_epoch = find_latest_checkpoint(model_dir)
        
        # Check if we made progress
        if new_latest_epoch > latest_epoch:
            print(f"Made progress: {latest_epoch} -> {new_latest_epoch} epochs")
            recovery_attempts = 0  # Reset recovery attempts counter
        
        checkpoint_path = new_checkpoint_path
        latest_epoch = new_latest_epoch
        
        # Update recovery file
        recovery_data = load_recovery_file(model_dir)
        if recovery_data:
            recovery_data["last_update"] = datetime.now().isoformat()
            recovery_data["current_epoch"] = latest_epoch
            recovery_data["recovery_attempts"] = recovery_attempts
            with open(os.path.join(model_dir, "recovery.json"), 'w') as f:
                json.dump(recovery_data, f, indent=2)

def get_system_info():
    """
    Get system information for monitoring.
    
    Returns:
        dict: System information
    """
    try:
        # Get CPU info
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_count = psutil.cpu_count(logical=True)
        
        # Get memory info
        memory = psutil.virtual_memory()
        memory_total = memory.total / (1024 ** 3)  # GB
        memory_used = memory.used / (1024 ** 3)    # GB
        memory_percent = memory.percent
        
        # Get disk info
        disk = psutil.disk_usage('/')
        disk_total = disk.total / (1024 ** 3)      # GB
        disk_used = disk.used / (1024 ** 3)        # GB
        disk_percent = disk.percent
        
        # Get system uptime
        uptime = time.time() - psutil.boot_time()
        uptime_str = str(timedelta(seconds=int(uptime)))
        
        # Get GPU info if available
        gpu_info = "Not available"
        try:
            # This is a simple check - you might need to adjust based on your system
            gpu_process = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader"], 
                                        capture_output=True, text=True, timeout=5)
            if gpu_process.returncode == 0:
                gpu_info = gpu_process.stdout.strip()
        except:
            pass
        
        return {
            "timestamp": datetime.now().isoformat(),
            "hostname": platform.node(),
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "cpu": {
                "percent": cpu_percent,
                "count": cpu_count
            },
            "memory": {
                "total_gb": round(memory_total, 2),
                "used_gb": round(memory_used, 2),
                "percent": memory_percent
            },
            "disk": {
                "total_gb": round(disk_total, 2),
                "used_gb": round(disk_used, 2),
                "percent": disk_percent
            },
            "uptime": uptime_str,
            "gpu": gpu_info
        }
    except Exception as e:
        return {"error": str(e)}

def send_notification(subject, message, args):
    """
    Send email notification if email settings are provided.
    
    Args:
        subject (str): Email subject
        message (str): Email message
        args: Command line arguments with email settings
    """
    if not hasattr(args, 'email_to') or not args.email_to:
        return False
    
    try:
        msg = MIMEMultipart()
        msg['From'] = args.email_from if hasattr(args, 'email_from') and args.email_from else 'omr.training@example.com'
        msg['To'] = args.email_to
        msg['Subject'] = f"OMR Training: {subject}"
        
        # Add system info to the message
        system_info = get_system_info()
        system_info_text = "\n\nSystem Information:\n"
        system_info_text += f"Hostname: {system_info.get('hostname', 'Unknown')}\n"
        system_info_text += f"Platform: {system_info.get('platform', 'Unknown')}\n"
        system_info_text += f"CPU Usage: {system_info.get('cpu', {}).get('percent', 0)}% ({system_info.get('cpu', {}).get('count', 0)} cores)\n"
        system_info_text += f"Memory: {system_info.get('memory', {}).get('used_gb', 0)}/{system_info.get('memory', {}).get('total_gb', 0)} GB ({system_info.get('memory', {}).get('percent', 0)}%)\n"
        system_info_text += f"Disk: {system_info.get('disk', {}).get('used_gb', 0)}/{system_info.get('disk', {}).get('total_gb', 0)} GB ({system_info.get('disk', {}).get('percent', 0)}%)\n"
        system_info_text += f"Uptime: {system_info.get('uptime', 'Unknown')}\n"
        system_info_text += f"GPU: {system_info.get('gpu', 'Not available')}\n"
        
        msg.attach(MIMEText(message + system_info_text, 'plain'))
        
        if hasattr(args, 'smtp_server') and args.smtp_server:
            server = smtplib.SMTP(args.smtp_server, args.smtp_port if hasattr(args, 'smtp_port') else 587)
            server.starttls()
            
            if hasattr(args, 'smtp_user') and args.smtp_user and hasattr(args, 'smtp_password') and args.smtp_password:
                server.login(args.smtp_user, args.smtp_password)
                
            server.send_message(msg)
            server.quit()
            return True
    except Exception as e:
        print(f"Failed to send email notification: {e}")
    
    return False

def main():
    """
    Main function.
    """
    parser = argparse.ArgumentParser(description='Auto-recovery training for OMR model')
    
    # Training parameters
    parser.add_argument('-corpus', dest='corpus', required=True, help='Path to the corpus')
    parser.add_argument('-set', dest='set', required=True, help='Path to the set file')
    parser.add_argument('-vocabulary', dest='vocabulary', required=True, help='Path to the vocabulary file')
    parser.add_argument('-save_model', dest='save_model', required=True, help='Path to save the model')
    parser.add_argument('-semantic', dest='semantic', action='store_true', default=False, help='Use semantic encoding')
    parser.add_argument('-batch_size', dest='batch_size', type=int, default=10, help='Number of epochs per batch')
    parser.add_argument('-total_epochs', dest='total_epochs', type=int, default=100, help='Total number of epochs to train')
    parser.add_argument('-recover', dest='recover', action='store_true', default=False, 
                        help='Recover from existing recovery file (ignores other parameters)')
    
    # Recovery and monitoring parameters
    parser.add_argument('-max_recovery', dest='max_recovery_attempts', type=int, default=10, 
                        help='Maximum number of consecutive recovery attempts before giving up')
    parser.add_argument('-monitoring_interval', dest='monitoring_interval', type=int, default=300, 
                        help='Interval in seconds between system monitoring checks')
    
    # Email notification parameters
    parser.add_argument('-email_to', dest='email_to', type=str, default=None, 
                        help='Email address to send notifications to')
    parser.add_argument('-email_from', dest='email_from', type=str, default='omr.training@example.com', 
                        help='Email address to send notifications from')
    parser.add_argument('-smtp_server', dest='smtp_server', type=str, default=None, 
                        help='SMTP server for sending email notifications')
    parser.add_argument('-smtp_port', dest='smtp_port', type=int, default=587, 
                        help='SMTP port for sending email notifications')
    parser.add_argument('-smtp_user', dest='smtp_user', type=str, default=None, 
                        help='SMTP username for authentication')
    parser.add_argument('-smtp_password', dest='smtp_password', type=str, default=None, 
                        help='SMTP password for authentication')
    parser.add_argument('-notification_interval', dest='notification_interval', type=int, default=3600, 
                        help='Interval in seconds between email notifications')
    
    args = parser.parse_args()
    
    # Handle recovery mode
    if args.recover:
        model_dir = os.path.dirname(args.save_model)
        recovery_data = load_recovery_file(model_dir)
        
        if recovery_data:
            print("Recovering training from saved state...")
            
            # Update args with recovery data
            args.corpus = recovery_data.get("corpus", args.corpus)
            args.set = recovery_data.get("set", args.set)
            args.vocabulary = recovery_data.get("vocabulary", args.vocabulary)
            args.save_model = recovery_data.get("save_model", args.save_model)
            args.semantic = recovery_data.get("semantic", args.semantic)
            args.batch_size = recovery_data.get("batch_size", args.batch_size)
            args.total_epochs = recovery_data.get("total_epochs", args.total_epochs)
            
            print(f"Recovered parameters:")
            print(f"  Corpus: {args.corpus}")
            print(f"  Set: {args.set}")
            print(f"  Vocabulary: {args.vocabulary}")
            print(f"  Save model: {args.save_model}")
            print(f"  Semantic: {args.semantic}")
            print(f"  Batch size: {args.batch_size}")
            print(f"  Total epochs: {args.total_epochs}")
        else:
            print("No recovery file found. Please provide training parameters.")
            return
    
    # Run training with recovery
    run_training_with_recovery(args)

if __name__ == "__main__":
    main()
