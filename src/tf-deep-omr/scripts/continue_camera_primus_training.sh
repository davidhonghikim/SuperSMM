#!/bin/bash

# Script to continue training from the latest checkpoint

# Configuration
CORPUS="/Users/danger/CascadeProjects/LOO/SuperSMM/src/tf-deep-omr/src/Data/camera_primus/Corpus"
SET="/Users/danger/CascadeProjects/LOO/SuperSMM/src/tf-deep-omr/data/camera_primus_set.txt"
VOCABULARY="/Users/danger/CascadeProjects/LOO/SuperSMM/src/tf-deep-omr/data/vocabulary_semantic.txt"
MODEL_DIR="/Users/danger/CascadeProjects/LOO/SuperSMM/src/tf-deep-omr/model/camera_primus_model"
TOTAL_EPOCHS=500
BATCH_SIZE=10

# Allow memory growth instead of pre-allocating all GPU memory
export TF_FORCE_GPU_ALLOW_GROWTH=true

echo "================================================================="
echo "Resuming OMR model training with camera_primus dataset"
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
