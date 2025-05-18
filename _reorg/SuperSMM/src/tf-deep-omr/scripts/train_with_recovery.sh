#!/bin/bash

# Robust training script with automatic crash recovery and TensorFlow optimizations
# This script will automatically resume training from the last checkpoint if it crashes

# Configuration
# Use sample dataset for faster testing
CORPUS="./sample_data/corpus"
SET="./sample_data/sample_set.txt"

# If sample dataset doesn't exist, create it from the fixed set file
if [ ! -d "$CORPUS" ] || [ ! -f "$SET" ]; then
    echo "Sample dataset not found. Creating from fixed set file..."
    python prepare_corpus.py -corpus "./data" -set "./data/train_fixed.txt" -sample 100 -output "./sample_data"
    
    # Verify sample data was created
    if [ ! -d "$CORPUS" ] || [ ! -f "$SET" ]; then
        echo "ERROR: Failed to create sample dataset!"
        exit 1
    fi
fi
VOCABULARY="./data/vocabulary_semantic.txt"
MODEL_DIR="./model/semantic_model"
# Reduced epochs for testing
TOTAL_EPOCHS=20
BATCH_SIZE=5

# TensorFlow Optimizations
export TF_ENABLE_AUTO_MIXED_PRECISION=1
export TF_GPU_THREAD_MODE=gpu_private
export TF_GPU_THREAD_COUNT=2
export TF_USE_CUDNN_BATCHNORM_SPATIAL_PERSISTENT=1
export TF_ENABLE_CUBLAS_TENSOR_OP_MATH_FP32=1
export TF_ENABLE_CUDNN_TENSOR_OP_MATH_FP32=1

# Allow memory growth instead of pre-allocating all GPU memory
export TF_FORCE_GPU_ALLOW_GROWTH=true

# Monitoring settings
MONITORING_INTERVAL=300  # 5 minutes
MAX_RECOVERY_ATTEMPTS=10

# Create model directory if it doesn't exist
mkdir -p "$MODEL_DIR"
mkdir -p "$MODEL_DIR/stats"
mkdir -p "$MODEL_DIR/logs"

# Make the script executable
chmod +x auto_recover_training.py

echo "=================================================================="
echo "Starting OMR model training with automatic crash recovery..."
echo "=================================================================="
echo "Total epochs: $TOTAL_EPOCHS"
echo "Model will be saved to: $MODEL_DIR"
echo "TensorFlow optimizations: ENABLED"
echo "GPU memory growth: ENABLED"
echo "System monitoring interval: $MONITORING_INTERVAL seconds"
echo "Maximum recovery attempts: $MAX_RECOVERY_ATTEMPTS"
echo "This training will automatically resume if it crashes."
echo "=================================================================="

# Log start time
START_TIME=$(date +"%Y-%m-%d %H:%M:%S")
echo "Training started at: $START_TIME" > "$MODEL_DIR/logs/training_session.log"

# Run the auto-recovery training script
python auto_recover_training.py \
  -corpus "$CORPUS" \
  -set "$SET" \
  -vocabulary "$VOCABULARY" \
  -save_model "$MODEL_DIR/model" \
  -semantic \
  -batch_size "$BATCH_SIZE" \
  -total_epochs "$TOTAL_EPOCHS" \
  -monitoring_interval "$MONITORING_INTERVAL" \
  -max_recovery "$MAX_RECOVERY_ATTEMPTS" 2>&1 | tee -a "$MODEL_DIR/logs/training_output.log"

# Log end time
END_TIME=$(date +"%Y-%m-%d %H:%M:%S")
echo "Training ended at: $END_TIME" >> "$MODEL_DIR/logs/training_session.log"

echo "=================================================================="
echo "Training complete or stopped."
echo "To resume training after a manual stop, simply run this script again."
echo "The training will automatically continue from the last checkpoint."
echo "=================================================================="
echo "Training logs saved to: $MODEL_DIR/logs/"
echo "System statistics saved to: $MODEL_DIR/stats/"
