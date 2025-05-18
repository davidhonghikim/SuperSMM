#!/bin/bash

# This script demonstrates how to use the batch training system
# It will train the model in batches of 10 epochs each

# Set paths
CORPUS_PATH="Data/primus"
SET_FILE="Data/train.txt"
VOC_FILE="Data/vocabulary_semantic.txt"
MODEL_DIR="Models/batch_trained_model"
TOTAL_EPOCHS=50

# Create model directory if it doesn't exist
mkdir -p $MODEL_DIR

# Initial training run - first 10 epochs
echo "Starting initial training batch (epochs 0-9)..."
python batch_training.py -semantic \
    -corpus $CORPUS_PATH \
    -set $SET_FILE \
    -vocabulary $VOC_FILE \
    -save_model $MODEL_DIR/model \
    -batch_size 10 \
    -total_epochs $TOTAL_EPOCHS

# To resume training automatically (this will be used if the training is interrupted)
echo ""
echo "To resume training automatically, run:"
echo "python batch_training.py -auto_resume -semantic -corpus $CORPUS_PATH -set $SET_FILE -vocabulary $VOC_FILE -save_model $MODEL_DIR/model -total_epochs $TOTAL_EPOCHS"
