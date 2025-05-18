#!/bin/bash

# This script tests the batch training system with a small number of epochs
# It will train the model for just 2 epochs to verify functionality

# Set paths
CORPUS_PATH="Data/primus"
SET_FILE="Data/train.txt"
VOC_FILE="Data/vocabulary_semantic.txt"
MODEL_DIR="Models/test_batch_model"
TOTAL_EPOCHS=2

# Create model directory if it doesn't exist
mkdir -p $MODEL_DIR

# Run test training
echo "Starting test training (2 epochs)..."
python batch_training.py -semantic \
    -corpus $CORPUS_PATH \
    -set $SET_FILE \
    -vocabulary $VOC_FILE \
    -save_model $MODEL_DIR/model \
    -batch_size 1 \
    -total_epochs $TOTAL_EPOCHS

echo ""
echo "Test completed. Check the output above for any errors."

# Plot the training statistics if the training completed successfully
if [ -f "$MODEL_DIR/training_stats.csv" ]; then
    echo ""
    echo "Plotting training statistics..."
    python plot_training_stats.py -stats $MODEL_DIR/training_stats.csv -output $MODEL_DIR/training_stats_plot.png
    echo "Statistics plot saved to $MODEL_DIR/training_stats_plot.png"
fi
