#!/usr/bin/env python
# -*- coding: utf-8

"""
Batch training script for the OMR model.
This script allows training in batches with checkpoint saving and resuming.
"""

import os
import sys
import time
import glob
import json
import argparse
from datetime import datetime, timedelta

# Add the project root to the path for imports
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

import tensorflow as tf
from primus import CTC_PriMuS
import ctc_utils
import ctc_model

# Import our logging utilities
from utils.logger import setup_logger
from utils.csv_logger import get_csv_logger

# Set up logging
logger = setup_logger("training", log_type="training")

# Initialize CSV logger
metrics_logger = get_csv_logger("training_metrics")

def train_batch(args, start_epoch, end_epoch, checkpoint_path=None):
    """
    Train the model for a specific range of epochs.
    
    Args:
        args: Command line arguments
        start_epoch: Starting epoch number
        end_epoch: Ending epoch number
        checkpoint_path: Path to load checkpoint from (optional)
    
    Returns:
        Path to the saved model checkpoint
    """
    # Log training start
    logger.info(f"Starting training from epoch {start_epoch} to {end_epoch}")
    logger.info(f"Model will be saved to: {args.save_model}")
    logger.info(f"Using corpus: {args.corpus}")
    
    # Initialize metrics
    cumulative_time = 0
    start_time = time.time()
    
    # Log environment info
    logger.info(f"TensorFlow version: {tf.__version__}")
    logger.info(f"CUDA available: {tf.test.is_built_with_cuda()}")
    if tf.test.is_built_with_cuda():
        logger.info(f"GPU available: {tf.config.list_physical_devices('GPU')}")
    else:
        logger.warning("Running on CPU - training may be slow")
    
    # Record overall start time
    overall_start_time = time.time()
    avg_epoch_time = 0
    
    # Enable TF1.x compatibility mode
    tf.compat.v1.disable_eager_execution()

    # Configure GPU memory growth
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True

    # TensorFlow performance optimization
    config.gpu_options.allow_growth = True  # Allocate only as much GPU memory as needed
    config.intra_op_parallelism_threads = 4  # Use 4 threads for operations
    config.inter_op_parallelism_threads = 4  # Use 4 threads between operations

    # Create a new graph and session
    graph = tf.compat.v1.Graph()
    with graph.as_default():
        # Create TensorFlow session with optimized config
        sess = tf.compat.v1.Session(config=config)

    # Load primus dataset
    primus = CTC_PriMuS(args.corpus, args.set, args.voc, args.semantic, val_split=0.1)

    # Parameterization
    img_height = 128
    params = ctc_model.default_model_params(img_height, primus.vocabulary_size)
    dropout = 0.5

    # Create log directory if it doesn't exist
    os.makedirs(os.path.dirname(args.log_file), exist_ok=True)

    # Initialize training state
    training_state = {
        'current_epoch': start_epoch,
        'total_epochs': end_epoch,
        'losses': [],
        'validation_metrics': [],
        'start_time': time.time(),
        'epoch_times': []
    }

    # Load previous training state if available
    if os.path.exists(args.log_file):
        try:
            with open(args.log_file, 'r') as f:
                previous_state = json.load(f)
                if 'losses' in previous_state:
                    training_state['losses'] = previous_state['losses']
                if 'validation_metrics' in previous_state:
                    training_state['validation_metrics'] = previous_state['validation_metrics']
        except Exception as e:
            logger.warning(f"Could not load previous training state: {e}")

    with graph.as_default():
        # Model
        inputs, seq_len, targets, decoded, loss, rnn_keep_prob = ctc_model.ctc_crnn(params)
        train_opt = tf.compat.v1.train.AdamOptimizer().minimize(loss)
        
        # Initialize saver and variables
        saver = tf.compat.v1.train.Saver(max_to_keep=None)
        sess.run(tf.compat.v1.global_variables_initializer())
        
        # Load checkpoint if provided
        if checkpoint_path:
            logger.info(f"Loading checkpoint from {checkpoint_path}")
            saver.restore(sess, checkpoint_path)

    # Training loop
    start_time = time.time()
    epoch_times = []

    logger.info(f"Starting training from epoch {start_epoch} to {end_epoch}")
    logger.info(f"Model will be saved to {args.save_model}")
    logger.info(f"Training state will be logged to {args.log_file}")
    logger.info(f"Performance stats will be saved to {args.stats_file}")
    logger.info("--------------------------------------------------")
    
    # Get memory usage function
    try:
        import psutil
        def get_memory_usage():
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / 1024 / 1024  # Convert to MB
    except ImportError:
        def get_memory_usage():
            return 0  # Return 0 if psutil is not available
    
    # Performance monitoring begins

    for epoch in range(start_epoch, end_epoch):
        epoch_start = time.time()
        
        # Get a batch of data
        batch_data = primus.nextBatch(params)
        
        # Ensure batch_data is a tuple with at least 3 elements
        if not isinstance(batch_data, tuple) or len(batch_data) < 3:
            logger.error(f"Error: Invalid batch data format. Expected tuple with 3 elements, got {type(batch_data)}")
            continue
        
        with graph.as_default():
            try:
                _, loss_value = sess.run([train_opt, loss],
                                        feed_dict={
                                            inputs: batch_data[0],
                                            seq_len: batch_data[1],
                                            targets: batch_data[2],
                                            rnn_keep_prob: dropout
                                        })
            except Exception as e:
                logger.error(f"Error during training: {e}")
                logger.error(f"Batch shapes - inputs: {batch_data[0].shape}, seq_len: {batch_data[1].shape}, targets: {type(batch_data[2])}")
                continue
        
        # Calculate timing metrics
        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        avg_epoch_time = sum(epoch_times) / len(epoch_times)
        
        elapsed = time.time() - start_time
        total_elapsed = cumulative_time + (time.time() - overall_start_time)
        remaining_epochs = end_epoch - epoch - 1
        
        # Get memory usage
        memory_usage = get_memory_usage()
        
        # Calculate metrics
        est_completion_seconds = remaining_epochs * avg_epoch_time
        est_completion = datetime.now() + timedelta(seconds=est_completion_seconds)
        
        # Log metrics
        metrics = {
            "epoch": epoch,
            "loss": float(loss_value),
            "epoch_time_sec": epoch_time,
            "cumulative_time_sec": total_elapsed,
            "remaining_epochs": remaining_epochs,
            "avg_epoch_time_sec": avg_epoch_time,
            "estimated_remaining_time_sec": est_completion_seconds,
            "learning_rate": 0.0,  # Add actual learning rate if available
            "batch_size": args.batch_size if hasattr(args, 'batch_size') else 0,
            "step": epoch,
            "memory_usage_mb": memory_usage
        }
        
        # Log to CSV
        metrics_logger.log_metrics(metrics)
        
        # Log to console
        logger.info(
            f"Epoch {epoch:4d}/{end_epoch}: "
            f"loss={loss_value:.4f}, "
            f"epoch_time={epoch_time:.2f}s, "
            f"cumulative_time={total_elapsed:.2f}s, "
            f"remaining_epochs={remaining_epochs}, "
            f"avg_epoch_time={avg_epoch_time:.2f}s, "
            f"estimated_remaining_time={est_completion_seconds/60:.1f}min, "
            f"memory_usage={memory_usage:.2f} MB"
        )
        
        # Print progress
        print(f"Epoch {epoch}/{end_epoch-1} - Loss: {loss_value:.4f} - Time: {epoch_time:.2f}s - Avg: {avg_epoch_time:.2f}s")
        print(f"Elapsed: {time.time() - start_time:.2f}s - Total training time: {cumulative_time:.2f}s")
        print(f"Est. completion: {est_completion.strftime('%Y-%m-%d %H:%M:%S')} (in {est_completion_seconds/60:.1f} minutes)")
        print(f"Memory usage: {memory_usage:.2f} MB")
        
        # Validate every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == end_epoch - 1:
            print("--------------------------------------------------")
            print(f"Time elapsed: {elapsed:.2f}s")
            print(f"Estimated completion: {est_completion.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Average epoch time: {avg_epoch_time:.2f}s")
            print(f"Loss value at epoch {epoch}: {loss_value}")
            
            print("Validating...")
            with graph.as_default():
                val_metrics = primus.getValidation(params, sess, inputs, seq_len, rnn_keep_prob, decoded)
                
                # Store validation metrics
                training_state['validation_metrics'].append({
                    'epoch': epoch,
                    'metrics': val_metrics,
                    'time': time.time() - start_time
                })
                
                # Save model
                print("Saving the model...")
                saver.save(sess, args.save_model, global_step=epoch)
                
                # Save training state
                with open(args.log_file, 'w') as f:
                    json.dump(training_state, f, indent=2)
                
            print("--------------------------------------------------")
    
    # Final save
    with graph.as_default():
        final_checkpoint = saver.save(sess, args.save_model, global_step=end_epoch-1)
        
    # Save final training state
    with open(args.log_file, 'w') as f:
        json.dump(training_state, f, indent=2)
    
    return final_checkpoint

def find_latest_checkpoint(model_dir):
    """
    Find the latest checkpoint in the model directory.
    
    Args:
        model_dir (str): Path to the model directory
        
    Returns:
        tuple: (checkpoint_path, epoch_number) or (None, 0) if not found
    """
    # Look for checkpoint files
    checkpoint_files = glob.glob(os.path.join(os.path.dirname(model_dir), "*.meta"))
    
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

def main():
    """
    Main function.
    """
    parser = argparse.ArgumentParser(description='Batch training for OMR model')
    parser.add_argument('-corpus', dest='corpus', required=True, help='Path to the corpus')
    parser.add_argument('-set', dest='set', required=True, help='Path to the set file')
    parser.add_argument('-vocabulary', dest='voc', required=True, help='Path to the vocabulary file')
    parser.add_argument('-save_model', dest='save_model', required=True, help='Path to save the model')
    parser.add_argument('-semantic', dest='semantic', action='store_true', default=False, help='Use semantic encoding')
    parser.add_argument('-batch_size', dest='batch_size', type=int, default=10, help='Number of epochs per batch')
    parser.add_argument('-total_epochs', dest='total_epochs', type=int, default=50, help='Total number of epochs to train')
    parser.add_argument('-auto_resume', dest='auto_resume', action='store_true', default=False, help='Automatically resume training from the latest checkpoint')
    parser.add_argument('-auto_recover', dest='auto_recover', action='store_true', default=True, help='Enable automatic recovery in case of crash (default: True)')
    parser.add_argument('-checkpoint', dest='checkpoint', type=str, default=None, help='Path to specific checkpoint to resume from')
    parser.add_argument('-log_file', dest='log_file', type=str, default=None, help='Path to save training log')
    args = parser.parse_args()
    
    # Set default log file if not provided
    if args.log_file is None:
        model_dir = os.path.dirname(args.save_model)
        args.log_file = os.path.join(model_dir, 'training_log.json')
        os.makedirs(model_dir, exist_ok=True)
    
    # Determine starting epoch and checkpoint path
    start_epoch = 0
    checkpoint_path = args.checkpoint
    
    # Auto-resume from the latest checkpoint if requested
    if args.auto_resume and not checkpoint_path:
        model_dir = os.path.dirname(args.save_model)
        checkpoint_path, last_epoch = find_latest_checkpoint(args.save_model)
        if checkpoint_path:
            start_epoch = last_epoch + 1
            print(f"Auto-resuming from checkpoint: {checkpoint_path} (epoch {last_epoch})")
    elif checkpoint_path:
        # Extract epoch number from provided checkpoint path
        try:
            checkpoint_parts = checkpoint_path.split('-')
            if len(checkpoint_parts) > 1:
                start_epoch = int(checkpoint_parts[-1]) + 1
        except:
            print("Could not determine starting epoch from checkpoint path. Starting from epoch 0.")
    
    # Calculate number of batches
    remaining_epochs = args.total_epochs - start_epoch
    if remaining_epochs <= 0:
        print(f"Training already completed ({start_epoch} epochs). Set a higher total_epochs value to continue.")
        return
        
    num_batches = (remaining_epochs + args.batch_size - 1) // args.batch_size  # Ceiling division
    
    print(f"Training from epoch {start_epoch} to {args.total_epochs} ({remaining_epochs} epochs remaining)")
    print(f"Will train in {num_batches} batches of {args.batch_size} epochs each")
    
    for batch in range(num_batches):
        batch_start = start_epoch + batch * args.batch_size
        batch_end = min(start_epoch + (batch + 1) * args.batch_size, args.total_epochs)
        
        print(f"\n=== Starting batch {batch+1}/{num_batches} (epochs {batch_start}-{batch_end-1}) ===\n")
        
        checkpoint_path = train_batch(args, batch_start, batch_end, checkpoint_path)
        
        print(f"\n=== Completed batch {batch+1}/{num_batches} ===")
        print(f"Checkpoint saved to: {checkpoint_path}")
        print(f"Current progress: {batch_end}/{args.total_epochs} epochs ({batch_end/args.total_epochs*100:.1f}%)")
        print("You can resume training from this checkpoint if needed.")
        
        # Ask if user wants to continue to next batch
        if batch < num_batches - 1:
            response = input("\nContinue to next batch? [Y/n]: ")
            if response.lower() == 'n':
                print("Training stopped by user.")
                print(f"\nTo resume later, run:")
                print(f"python batch_training.py -auto_resume -corpus {args.corpus} -set {args.set} -vocabulary {args.voc} -save_model {args.save_model} {'-semantic' if args.semantic else ''}")
                break

if __name__ == "__main__":
    main()
