#!/bin/bash

# Optimized training script for the OMR model
# Features:
# - Performance optimized TensorFlow configuration
# - Detailed statistics tracking
# - Automatic checkpoint management
# - Training visualization

# Set paths
CORPUS_PATH="Data/primus"
SET_FILE="Data/train.txt"
VOC_FILE="Data/vocabulary_semantic.txt"
MODEL_DIR="Models/optimized_model"
TOTAL_EPOCHS=50
BATCH_SIZE=10

# Create model directory if it doesn't exist
mkdir -p $MODEL_DIR

# Display training configuration
echo "======================================================="
echo "           OPTIMIZED OMR MODEL TRAINING"
echo "======================================================="
echo "Corpus path:       $CORPUS_PATH"
echo "Training set:      $SET_FILE"
echo "Vocabulary file:   $VOC_FILE"
echo "Model directory:   $MODEL_DIR"
echo "Total epochs:      $TOTAL_EPOCHS"
echo "Batch size:        $BATCH_SIZE epochs"
echo "======================================================="
echo ""

# Check if there's an existing checkpoint
CHECKPOINT=$(python training_manager.py find_checkpoint -model_dir $MODEL_DIR -quiet)
if [ -n "$CHECKPOINT" ]; then
    echo "Found existing checkpoint: $CHECKPOINT"
    echo "Resuming training from checkpoint..."
    
    # Get the current epoch from the checkpoint filename
    CURRENT_EPOCH=$(echo $CHECKPOINT | grep -o '[0-9]*$')
    echo "Current epoch: $CURRENT_EPOCH"
    
    # Resume training
    python batch_training.py -auto_resume \
        -semantic \
        -corpus $CORPUS_PATH \
        -set $SET_FILE \
        -vocabulary $VOC_FILE \
        -save_model $MODEL_DIR/model \
        -total_epochs $TOTAL_EPOCHS
else
    echo "Starting new training session..."
    
    # Start new training
    python batch_training.py \
        -semantic \
        -corpus $CORPUS_PATH \
        -set $SET_FILE \
        -vocabulary $VOC_FILE \
        -save_model $MODEL_DIR/model \
        -batch_size $BATCH_SIZE \
        -total_epochs $TOTAL_EPOCHS
fi

# Plot training statistics
echo ""
echo "Generating training visualizations..."
python plot_training_stats.py -stats $MODEL_DIR/training_stats.csv -output $MODEL_DIR/training_stats_plot.png -no_show
python plot_training.py -log $MODEL_DIR/training_log.json -output $MODEL_DIR/training_loss_plot.png

echo ""
echo "Training completed!"
echo "Model saved to: $MODEL_DIR"
echo "Statistics plot: $MODEL_DIR/training_stats_plot.png"
echo "Loss plot: $MODEL_DIR/training_loss_plot.png"
echo ""

# Display training summary
echo "======================================================="
echo "                 TRAINING SUMMARY"
echo "======================================================="
python training_manager.py analyze -log_file $MODEL_DIR/training_log.json
echo "======================================================="
