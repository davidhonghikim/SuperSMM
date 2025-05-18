#!/bin/bash

# Script to continue training from the latest checkpoint

# Configuration
CORPUS="/Users/danger/CascadeProjects/LOO/SuperSMM/src/tf-deep-omr/src/Data/primus"
SET="/Users/danger/CascadeProjects/LOO/SuperSMM/src/tf-deep-omr/src/Data/train.txt"
VOCABULARY="/Users/danger/CascadeProjects/LOO/SuperSMM/src/tf-deep-omr/src/Data/vocabulary_semantic.txt"
MODEL_DIR="/Users/danger/CascadeProjects/LOO/SuperSMM/src/tf-deep-omr/model/primus_model"
TOTAL_EPOCHS=100000
BATCH_SIZE=50

# Allow memory growth instead of pre-allocating all GPU memory
export TF_FORCE_GPU_ALLOW_GROWTH=true

echo "================================================================="
echo "Resuming OMR model training with primus dataset"
echo "=================================================================="
echo "Dataset: $CORPUS"
echo "Set file: $SET"
echo "Model directory: $MODEL_DIR"
echo "=================================================================="

# Run the training with resume flag
cd "$(dirname "$0")/.." || exit
python src/ctc_training.py \
  -corpus "$CORPUS" \
  -set "$SET" \
  -save_model "$MODEL_DIR/model" \
  -vocabulary "$VOCABULARY" \
  -semantic \
  -resume 2>&1 | tee -a "$MODEL_DIR/logs/training_output.log"

echo "=================================================================="
echo "Training complete!"
echo "=================================================================="
