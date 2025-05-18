import tensorflow as tf
from primus import CTC_PriMuS
import ctc_utils
import ctc_model
import argparse
import os
import time
from datetime import datetime, timedelta
from typing import Dict, Any
import csv
import glob

# Enable TF1.x compatibility mode
tf.compat.v1.disable_eager_execution()

# Configure GPU memory growth
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True

# Create a new graph and session
graph = tf.compat.v1.Graph()
with graph.as_default():
    sess = tf.compat.v1.Session(config=config)

parser = argparse.ArgumentParser(description="Train model.")
parser.add_argument(
    "-corpus", dest="corpus", type=str, required=True, help="Path to the corpus."
)
parser.add_argument(
    "-set", dest="set", type=str, required=True, help="Path to the set file."
)
parser.add_argument(
    "-save_model",
    dest="save_model",
    type=str,
    required=True,
    help="Path to save the model.",
)
parser.add_argument(
    "-vocabulary",
    dest="voc",
    type=str,
    required=True,
    help="Path to the vocabulary file.",
)
parser.add_argument("-semantic", dest="semantic", action="store_true", default=False)
parser.add_argument("-resume", dest="resume", action="store_true", default=False,
                   help="Resume training from latest checkpoint")
parser.add_argument("-max_epochs", dest="max_epochs", type=int, default=0,
                   help="Maximum total epochs to train")
parser.add_argument("-epochs_per_run", dest="epochs_per_run", type=int, default=0,
                   help="Maximum epochs per run")
args = parser.parse_args()


def adjust_path(path: str) -> str:
    """Convert a path to an absolute path."""
    # If path is absolute, return as is
    if os.path.isabs(path):
        return path
        
    # If path exists as is, return it
    if os.path.exists(path):
        return os.path.abspath(path)
        
    # Try relative to current directory
    cwd = os.getcwd()
    abs_path = os.path.abspath(os.path.join(cwd, path))
    if os.path.exists(abs_path):
        return abs_path
        
    # Try relative to script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    abs_path = os.path.abspath(os.path.join(script_dir, path))
    if os.path.exists(abs_path):
        return abs_path
        
    # Try in Data directory
    data_dir = os.path.abspath(os.path.join(script_dir, '..', 'Data'))
    abs_path = os.path.abspath(os.path.join(data_dir, path))
    if os.path.exists(abs_path):
        return abs_path
        
    # If all else fails, return the original path and let the caller handle the error
    return os.path.abspath(path)


# Adjust paths
args.corpus = adjust_path(args.corpus)
args.set = adjust_path(args.set)
args.voc = adjust_path(args.voc)

# Create directories for save_model if they don't exist
os.makedirs(os.path.dirname(args.save_model), exist_ok=True)

# Initialize CSV logging
log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')
os.makedirs(log_dir, exist_ok=True)
csv_log_path = os.path.join(log_dir, 'training_log.csv')

# Check for existing checkpoints and CSV log
latest_checkpoint = None
start_epoch = 0
if args.resume:
    # Check for existing checkpoints
    checkpoint_pattern = args.save_model + "-*"
    checkpoints = glob.glob(checkpoint_pattern + ".meta")
    if checkpoints:
        # Extract epoch numbers and find the latest
        checkpoint_epochs = []
        for ckpt in checkpoints:
            try:
                # Extract epoch number from filename (e.g., model-42.meta)
                epoch_str = ckpt.replace(args.save_model + "-", "").replace(".meta", "")
                epoch_num = int(epoch_str)
                checkpoint_epochs.append((epoch_num, ckpt[:-5]))  # Remove .meta extension
            except ValueError:
                continue
        
        if checkpoint_epochs:
            # Sort by epoch number and get the latest
            checkpoint_epochs.sort(reverse=True)
            start_epoch, latest_checkpoint = checkpoint_epochs[0]
            print(f"Found checkpoint at epoch {start_epoch}: {latest_checkpoint}")
            start_epoch += 1  # Start from the next epoch
    
    # Check for existing CSV log - append to it instead of overwriting
    if os.path.exists(csv_log_path) and start_epoch > 0:
        print(f"Appending to existing log file: {csv_log_path}")
        # No need to create the file - we'll append to it
    else:
        # Create CSV file with headers if it doesn't exist or we're not actually resuming
        with open(csv_log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp',
                'epoch',
                'loss',
                'validation_error',
                'ser_percent',
                'epoch_time_sec',
                'cumulative_time_sec',
                'memory_usage_mb',
                'batch_size',
                'learning_rate',
                'gpu_memory_mb',
                'checkpoint_path',
                'validation_samples',
                'dataset_size',
                'remaining_epochs',
                'est_completion_time'
            ])
else:
    # If not resuming, create a new CSV file with headers
    with open(csv_log_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'timestamp',
            'epoch',
            'loss',
            'validation_error',
            'ser_percent',
            'epoch_time_sec',
            'cumulative_time_sec',
            'memory_usage_mb',
            'batch_size',
            'learning_rate',
            'gpu_memory_mb',
            'checkpoint_path',
            'validation_samples',
            'dataset_size',
            'remaining_epochs',
            'est_completion_time'
        ])

# Load primus
primus = CTC_PriMuS(args.corpus, args.set, args.voc, args.semantic, val_split=0.1)

# Parameterization
img_height = 128
params = ctc_model.default_model_params(img_height, primus.vocabulary_size)
TOTAL_EPOCHS=100000
BATCH_SIZE=50

# Use command line arguments for epoch limits if provided, otherwise use defaults
# that are appropriate for the dataset size
if hasattr(args, 'max_epochs') and args.max_epochs > 0:
    max_total_epochs = args.max_epochs
else:
    # Scale max_total_epochs based on dataset size with a reasonable default
    dataset_size = len(primus.training_list)
    max_total_epochs = max(50, min(dataset_size * 10, 100000))  # Between 500 and 10000 based on dataset size
    print(f"Setting max total epochs to {max_total_epochs} based on dataset size ({dataset_size} samples)")

if hasattr(args, 'epochs_per_run') and args.epochs_per_run > 0:
    max_epochs_per_run = args.epochs_per_run
else:
    # Scale epochs_per_run based on dataset size with a reasonable default
    max_epochs_per_run = max(50, min(dataset_size, 50))  # Between 50 and 500 based on dataset size
    print(f"Setting epochs per run to {max_epochs_per_run}")

dropout = 0.5  # Dropout rate for training
training_state_path = os.path.join(os.path.dirname(args.save_model), 'training_state.txt')

# Get the total number of epochs already completed (across all training runs)
total_completed_epochs = 0
if os.path.exists(training_state_path):
    try:
        with open(training_state_path, 'r') as f:
            total_completed_epochs = int(f.read().strip())
            print(f"Found training state: {total_completed_epochs} epochs already completed")
    except Exception as e:
        print(f"Error reading training state: {e}")

# If resuming, use the max of the checkpoint epoch and the stored total
if args.resume and start_epoch > 0:
    if total_completed_epochs > start_epoch:
        # If we have a larger number in our training state, use that
        print(f"Resuming training from epoch {total_completed_epochs} (training state) instead of {start_epoch} (checkpoint)")
        start_epoch = total_completed_epochs
    else:
        # Otherwise, update our training state with the checkpoint epoch
        total_completed_epochs = start_epoch
        with open(training_state_path, 'w') as f:
            f.write(str(total_completed_epochs))
        print(f"Updated training state to match checkpoint: {total_completed_epochs}")
    
    print(f"Resuming training from epoch {start_epoch}")
else:
    print(f"Starting new training run from epoch {start_epoch}")

# Calculate the end epoch for this run
end_epoch = min(start_epoch + max_epochs_per_run, max_total_epochs)

print(f"Training from epoch {start_epoch} to {end_epoch} ({end_epoch - start_epoch} epochs this run)")
print(f"Total training plan: {max_total_epochs} epochs")

with graph.as_default():
    # Model
    inputs, seq_len, targets, decoded, loss, rnn_keep_prob = ctc_model.ctc_crnn(params)
    train_opt = tf.compat.v1.train.AdamOptimizer().minimize(loss)

    # Initialize saver and variables
    saver = tf.compat.v1.train.Saver(max_to_keep=None)
    sess.run(tf.compat.v1.global_variables_initializer())
    
    # Restore from checkpoint if resuming
    if latest_checkpoint:
        print(f"Restoring model from checkpoint: {latest_checkpoint}")
        saver.restore(sess, latest_checkpoint)

# Training loop
start_time = time.time()
epoch_times = []

for epoch in range(start_epoch, end_epoch):
    epoch_start = time.time()
    batch_images, seq_lengths, batch_labels = primus.nextBatch(params)

    with graph.as_default():
        _, loss_value = sess.run(
            [train_opt, loss],
            feed_dict={
                inputs: batch_images,
                seq_len: seq_lengths,
                targets: ctc_utils.sparse_tuple_from(batch_labels),
                rnn_keep_prob: dropout,
            },
        )

    epoch_time = time.time() - epoch_start
    epoch_times.append(epoch_time)

    # Calculate average epoch time and estimate completion
    avg_epoch_time = sum(epoch_times[-100:]) / min(len(epoch_times), 100)
    remaining_epochs = max_total_epochs - epoch - 1  # Remaining in the entire training
    current_run_remaining = end_epoch - epoch - 1  # Remaining in this run
    est_remaining_time = remaining_epochs * avg_epoch_time
    est_completion = datetime.now() + timedelta(seconds=est_remaining_time)
    est_run_completion = datetime.now() + timedelta(seconds=current_run_remaining * avg_epoch_time)

    # Print detailed progress for every epoch
    elapsed = time.time() - start_time
    print(
        f"Epoch {epoch}/{max_total_epochs} (run: {epoch-start_epoch+1}/{end_epoch-start_epoch}) - Loss: {loss_value:.4f} - Time: {epoch_time:.2f}s - Avg: {avg_epoch_time:.2f}s"
    )
    print(
        f"Elapsed: {elapsed:.2f}s - Est. run completion: {est_run_completion.strftime('%Y-%m-%d %H:%M:%S')}"
    )
    print(
        f"Est. total completion: {est_completion.strftime('%Y-%m-%d %H:%M:%S')} (remaining: {remaining_epochs} epochs)"
    )

    # Update training state after each epoch
    with open(training_state_path, 'w') as f:
        f.write(str(epoch + 1))
    
    # Get memory usage
    try:
        import psutil
        process = psutil.Process(os.getpid())
        memory_usage = process.memory_info().rss / (1024 * 1024)  # Convert to MB
    except:
        memory_usage = 0

    # Log progress to CSV
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    gpu_memory = 0
    try:
        if tf.test.is_built_with_cuda():
            gpu_memory = tf.config.experimental.get_memory_info('GPU:0')['current'] / (1024 * 1024)  # Convert to MB
    except:
        pass
    
    # Get memory usage
    try:
        import psutil
        process = psutil.Process(os.getpid())
        memory_usage = process.memory_info().rss / (1024 * 1024)  # Convert to MB
    except:
        memory_usage = 0
        
    checkpoint_path = args.save_model + f"-{epoch}" if epoch % 10 == 0 else ""
    with open(csv_log_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            timestamp,
            epoch + 1,
            f"{loss_value:.4f}",
            "",
            "",
            f"{epoch_time:.2f}",
            f"{elapsed:.2f}",
            f"{memory_usage:.2f}",
            params["batch_size"],
            0.001,  # Default learning rate
            gpu_memory,
            checkpoint_path,
            "",
            len(primus.training_list),
            max_total_epochs - epoch - 1,
            est_completion.strftime('%Y-%m-%d %H:%M:%S')
        ])

    # Validate and save every 10 epochs as requested
    if epoch % 10 == 0:
        # VALIDATION
        elapsed = time.time() - start_time
        print("-" * 50)
        print(f"Time elapsed: {elapsed:.2f}s")
        print(f'Estimated completion: {est_completion.strftime("%Y-%m-%d %H:%M:%S")}')
        print(f"Average epoch time: {avg_epoch_time:.2f}s")
        print(f"Loss value at epoch {epoch}: {loss_value}")
        print("Validating...")

        validation_images, validation_seq_lengths, validation_labels = primus.getValidation(params)
        validation_size = len(validation_labels)

        # Use the returned validation data directly
        validation_inputs = validation_images
        validation_targets = validation_labels

        with graph.as_default():
            val_idx = 0
            val_ed = 0
            val_len = 0
            val_count = 0

            while val_idx < validation_size:
                # Ensure we have data to process
                if val_idx >= len(validation_inputs):
                    print(
                        f"Warning: val_idx {val_idx} exceeds validation batch size {len(validation_inputs)}"
                    )
                    break

                # Get the actual batch size for this iteration
                current_batch_size = min(
                    params["batch_size"], len(validation_inputs) - val_idx
                )
                if current_batch_size <= 0:
                    break

                try:
                    mini_batch_feed_dict = {
                        inputs: validation_inputs[
                            val_idx : val_idx + current_batch_size
                        ],
                        seq_len: validation_seq_lengths[
                            val_idx : val_idx + current_batch_size
                        ],
                        rnn_keep_prob: 1.0,
                    }

                    # Print shape information for debugging
                    print(
                        f"Validation batch shapes - inputs: {validation_inputs[val_idx:val_idx+current_batch_size].shape}, "
                        f"seq_len: {validation_seq_lengths[val_idx:val_idx+current_batch_size].shape}"
                    )

                    prediction = sess.run(decoded, mini_batch_feed_dict)
                except Exception as e:
                    print(f"Error during validation: {e}")
                    val_idx = val_idx + current_batch_size
                    continue

                str_predictions = ctc_utils.sparse_tensor_to_strs(prediction)

                for i in range(len(str_predictions)):
                    if val_idx + i < len(validation_targets):
                        ed = ctc_utils.edit_distance(
                            str_predictions[i], validation_targets[val_idx + i]
                        )
                        val_ed = val_ed + ed
                        val_len = val_len + len(validation_targets[val_idx + i])
                        val_count = val_count + 1

                val_idx = val_idx + params["batch_size"]

            # Calculate and print validation metrics
            if val_count > 0:
                val_error = 1. * val_ed / val_count
                ser_percent = 100. * val_ed / val_len
                # Log validation metrics
                with open(csv_log_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([timestamp, epoch + 1, f"{loss_value:.4f}", f"{val_error:.4f}", f"{ser_percent:.2f}"])
                print(
                    f"[Epoch {epoch}] {val_error:.4f} ({ser_percent:.2f}% SER) from {val_count} samples"
                )
            else:
                print(f"[Epoch {epoch}] No validation samples processed")

            print("Saving the model...")
            saver.save(sess, args.save_model, global_step=epoch)
            print("-" * 50)


def run_training(config: Dict[str, Any]):
    """
    Run the training process with the provided configuration.

    Args:
        config: A dictionary containing training configuration parameters.
    """
    # Extract configuration parameters
    corpus_path = config.get("corpus")
    set_file = config.get("set_file")
    vocabulary_path = config.get("vocabulary")
    save_model_path = config.get("save_model")
    semantic = config.get("semantic", False)
    agnostic = config.get("agnostic", False)
    validation_split = config.get("validation_split", 0.1)
    total_epochs = config.get("epochs", 50)
    epochs_per_run = config.get("epochs_per_run", 50)
    batch_size = config.get("batch_size", 16)
    learning_rate = config.get("learning_rate", 0.001)
    resume = config.get("resume", False)
    
    # Training state path
    training_state_path = os.path.join(os.path.dirname(save_model_path), 'training_state.txt')
    
    # Get the total number of epochs already completed (across all training runs)
    total_completed_epochs = 0
    if os.path.exists(training_state_path) and resume:
        try:
            with open(training_state_path, 'r') as f:
                total_completed_epochs = int(f.read().strip())
                print(f"Found training state: {total_completed_epochs} epochs already completed")
        except Exception as e:
            print(f"Error reading training state: {e}")
    
    # Initialize CSV logging
    log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    csv_log_path = os.path.join(log_dir, 'training_log.csv')
    
    # Check if we should append to existing log
    if resume and os.path.exists(csv_log_path) and total_completed_epochs > 0:
        print(f"Appending to existing log file: {csv_log_path}")
    else:
        # Create/overwrite CSV file with headers
        with open(csv_log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp', 
                'epoch', 
                'loss', 
                'validation_error', 
                'ser_percent',
                'epoch_time_sec',
                'cumulative_time_sec',
                'memory_usage_mb',
                'batch_size',
                'learning_rate',
                'gpu_memory_mb',
                'checkpoint_path',
                'validation_samples',
                'dataset_size',
                'remaining_epochs',
                'est_completion_time'
            ])

    # Adjust paths if needed
    def adjust_path(path):
        # If the path is relative and doesn't exist, try to find it in the Data directory
        if not os.path.isabs(path) and not os.path.exists(path):
            data_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "Data",
                os.path.basename(path),
            )
            if os.path.exists(data_path):
                return data_path
        return path

    # Adjust paths
    corpus_path = adjust_path(corpus_path)
    set_file = adjust_path(set_file)
    vocabulary_path = adjust_path(vocabulary_path)

    print(f"Starting training with the following configuration:")
    print(f"Corpus: {corpus_path}")
    print(f"Set file: {set_file}")
    print(f"Total epochs: {total_epochs}")
    print(f"Epochs per run: {epochs_per_run}")
    print(f"Starting from epoch: {total_completed_epochs}")
    
    # Initialize PRIMUS dataset
    primus = CTC_PriMuS(corpus_path, set_file, vocabulary_path, semantic, val_split=validation_split)
    
    # Parameterization
    img_height = 128
    params = ctc_model.default_model_params(img_height, primus.vocabulary_size)
    params["batch_size"] = batch_size
    dropout = 0.5
    
    # Calculate the end epoch for this run
    start_epoch = total_completed_epochs
    end_epoch = min(start_epoch + epochs_per_run, total_epochs)
    
    print(f"Training from epoch {start_epoch} to {end_epoch} ({end_epoch - start_epoch} epochs this run)")
    
    # Rest of your training code...
    
    with graph.as_default():
        # Input placeholders
        inputs = tf.compat.v1.placeholder(tf.float32, [None, img_height, None, 1], name="inputs")
        seq_len = tf.compat.v1.placeholder(tf.int32, [None], name="seq_len")
        targets = tf.compat.v1.sparse_placeholder(tf.int32, name="targets")
        rnn_keep_prob = tf.compat.v1.placeholder(tf.float32, name="rnn_keep_prob")
        
        # Model
        logits, _ = ctc_model.ctc_crnn(inputs, params, is_training=True)
        loss = tf.reduce_mean(ctc_utils.ctc_loss(logits, targets, seq_len))
        train_op = tf.compat.v1.train.AdamOptimizer(learning_rate).minimize(loss)
        
        # Decoder
        decoded, _ = tf.nn.ctc_beam_search_decoder(
            logits, seq_len, beam_width=50, merge_repeated=True
        )
        
        # Saver
        saver = tf.compat.v1.train.Saver(max_to_keep=10)

    # Check for existing checkpoints if resuming
    latest_checkpoint = None
    if resume and start_epoch > 0:
        # Check for checkpoint with the exact epoch number first
        checkpoint_path = f"{save_model_path}_epoch_{start_epoch}"
        if os.path.exists(f"{checkpoint_path}.meta"):
            latest_checkpoint = checkpoint_path
            print(f"Found exact checkpoint for epoch {start_epoch}: {latest_checkpoint}")
        else:
            # Look for any checkpoints
            checkpoint_pattern = f"{save_model_path}*"
            checkpoints = glob.glob(checkpoint_pattern + ".meta")
            if checkpoints:
                # Sort by modification time to get the latest
                latest_checkpoint = max(checkpoints, key=os.path.getmtime)[:-5]  # Remove .meta extension
                print(f"Using latest checkpoint: {latest_checkpoint}")

    # Initialize variables and restore from checkpoint if resuming
    with graph.as_default():
        sess.run(tf.compat.v1.global_variables_initializer())
        
        if latest_checkpoint and resume:
            print(f"Restoring from checkpoint: {latest_checkpoint}")
            saver.restore(sess, latest_checkpoint)

    # Create save directory
    os.makedirs(os.path.dirname(save_model_path), exist_ok=True)

    # Training loop
    print("\nStarting training...")
    start_time = time.time()
    epoch_times = []

    for epoch in range(start_epoch, end_epoch):
        try:
            epoch_start = time.time()
            
            # Get next batch
            batch = primus.nextBatch(params)
            if not batch or len(batch) < 3:
                print("No more training data")
                break
                
            # Prepare feed dict
            feed_dict = {
                inputs: batch[0],
                seq_len: batch[1],
                targets: ctc_utils.sparse_tuple_from(batch[2]),
                rnn_keep_prob: dropout
            }
            
            # Run training step
            _, loss_val = sess.run([train_op, loss], feed_dict=feed_dict)
            
            # Calculate epoch time
            epoch_time = time.time() - epoch_start
            epoch_times.append(epoch_time)
            
            # Calculate average epoch time and estimate completion
            avg_epoch_time = sum(epoch_times[-100:]) / min(len(epoch_times), 100)
            total_elapsed = time.time() - start_time
            remaining_epochs = total_epochs - epoch - 1
            est_remaining_time = remaining_epochs * avg_epoch_time
            est_completion = datetime.now() + timedelta(seconds=est_remaining_time)
            
            # Get memory usage
            try:
                import psutil
                process = psutil.Process(os.getpid())
                memory_usage = process.memory_info().rss / (1024 * 1024)  # Convert to MB
            except:
                memory_usage = 0
                
            # Get GPU memory if available
            gpu_memory = 0
            try:
                if tf.test.is_built_with_cuda():
                    gpu_memory = tf.config.experimental.get_memory_info('GPU:0')['current'] / (1024 * 1024)  # Convert to MB
            except:
                pass
                
            # Update training state
            with open(training_state_path, 'w') as f:
                f.write(str(epoch + 1))
            
            # Log progress to CSV
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            checkpoint_path = ""
            if (epoch + 1) % 10 == 0:
                checkpoint_path = f"{save_model_path}_epoch_{epoch+1}"
                
            with open(csv_log_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    timestamp,
                    epoch + 1,
                    f"{loss_val:.4f}",
                    "",
                    "",
                    f"{epoch_time:.2f}",
                    f"{total_elapsed:.2f}",
                    f"{memory_usage:.2f}",
                    batch_size,
                    learning_rate,
                    gpu_memory,
                    checkpoint_path,
                    "",
                    len(primus.training_list),
                    total_epochs - epoch - 1,
                    est_completion.strftime('%Y-%m-%d %H:%M:%S')
                ])
            
            # Print progress
            print(f"Epoch {epoch+1}/{total_epochs} (run: {epoch-start_epoch+1}/{end_epoch-start_epoch}) - Loss: {loss_val:.4f} - Time: {epoch_time:.2f}s - Avg: {avg_epoch_time:.2f}s")
            print(f"Elapsed: {total_elapsed:.2f}s - Est. total completion: {est_completion.strftime('%Y-%m-%d %H:%M:%S')} (remaining: {remaining_epochs} epochs)")
            
            # Save model every 10 epochs
            if (epoch + 1) % 10 == 0:
                save_file = f"{save_model_path}_epoch_{epoch+1}"
                saver.save(sess, save_file)
                print(f"Model saved to {save_file}")
                
                # Validation code would go here...
                
        except Exception as e:
            print(f"Error during epoch {epoch + 1}: {str(e)}")
            break

    # Save final model
    final_save_path = f"{save_model_path}_epoch_{end_epoch}"
    saver.save(sess, final_save_path)
    print(f"\nTraining run completed ({start_epoch} to {end_epoch}) in {(time.time() - start_time)/60:.2f} minutes")
    print(f"Final model saved to {final_save_path}")
    
    if end_epoch < total_epochs:
        print(f"\nTo continue training from epoch {end_epoch}, run with the resume option")
        print(f"Training progress: {end_epoch}/{total_epochs} epochs ({end_epoch/total_epochs*100:.1f}%)")
    else:
        print(f"\nTraining completed! All {total_epochs} epochs finished.")

    return True
