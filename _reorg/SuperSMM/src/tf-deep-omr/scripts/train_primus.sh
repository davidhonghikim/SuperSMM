#!/bin/bash

# Training script for real data using the primus dataset

# Configuration
CORPUS="/Users/danger/CascadeProjects/LOO/SuperSMM/src/tf-deep-omr/src/Data/primus"
SET="/Users/danger/CascadeProjects/LOO/SuperSMM/src/tf-deep-omr/data/primus_set.txt"
VOCABULARY="/Users/danger/CascadeProjects/LOO/SuperSMM/src/tf-deep-omr/data/vocabulary_semantic.txt"
MODEL_DIR="/Users/danger/CascadeProjects/LOO/SuperSMM/src/tf-deep-omr/model/primus_model"
TOTAL_EPOCHS=500
BATCH_SIZE=10

# Allow memory growth instead of pre-allocating all GPU memory
export TF_FORCE_GPU_ALLOW_GROWTH=true

# Create model directory if it doesn't exist
mkdir -p "$MODEL_DIR"
mkdir -p "$MODEL_DIR/logs"

echo "================================================================="
echo "Starting OMR model training with primus dataset"
echo "=================================================================="
echo "Dataset: $CORPUS"
echo "Set file: $SET"
echo "Vocabulary: $VOCABULARY"
echo "Model directory: $MODEL_DIR"
echo "Total epochs: $TOTAL_EPOCHS"
echo "=================================================================="

# Run the training
cd "$(dirname "$0")/.." || exit
python src/ctc_training.py \
  -corpus "$CORPUS" \
  -set "$SET" \
  -save_model "$MODEL_DIR/model" \
  -vocabulary "$VOCABULARY" \
  -semantic 2>&1 | tee -a "$MODEL_DIR/logs/training_output.log"

echo "=================================================================="
echo "Training complete!"
echo "=================================================================="
