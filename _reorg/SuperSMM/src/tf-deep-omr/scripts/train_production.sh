#!/bin/bash

# Production training script with automatic crash recovery and TensorFlow optimizations
# This script will automatically resume training from the last checkpoint if it crashes

# Configuration
CORPUS="./Data"
SET="./data/train_fixed.txt"
VOCABULARY="./data/vocabulary_semantic.txt"
MODEL_DIR="./model/production_model"
TOTAL_EPOCHS=300
BATCH_SIZE=10

# Email notification configuration
# Uncomment and fill in these values to enable email notifications
#EMAIL_TO="your.email@example.com"
#SMTP_SERVER="smtp.example.com"
#SMTP_PORT=587
#SMTP_USER="username"
#SMTP_PASSWORD="password"

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
NOTIFICATION_INTERVAL=3600  # 1 hour

# Create model directory if it doesn't exist
mkdir -p "$MODEL_DIR"
mkdir -p "$MODEL_DIR/stats"
mkdir -p "$MODEL_DIR/logs"

# Make the script executable
chmod +x auto_recover_training.py

echo "=================================================================="
echo "Starting PRODUCTION OMR model training with automatic crash recovery..."
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

# Build email notification parameters
EMAIL_PARAMS=""
if [ ! -z "$EMAIL_TO" ]; then
  EMAIL_PARAMS="$EMAIL_PARAMS -email_to $EMAIL_TO"
  
  if [ ! -z "$SMTP_SERVER" ]; then
    EMAIL_PARAMS="$EMAIL_PARAMS -smtp_server $SMTP_SERVER"
  fi
  
  if [ ! -z "$SMTP_PORT" ]; then
    EMAIL_PARAMS="$EMAIL_PARAMS -smtp_port $SMTP_PORT"
  fi
  
  if [ ! -z "$SMTP_USER" ]; then
    EMAIL_PARAMS="$EMAIL_PARAMS -smtp_user $SMTP_USER"
  fi
  
  if [ ! -z "$SMTP_PASSWORD" ]; then
    EMAIL_PARAMS="$EMAIL_PARAMS -smtp_password $SMTP_PASSWORD"
  fi
  
  if [ ! -z "$NOTIFICATION_INTERVAL" ]; then
    EMAIL_PARAMS="$EMAIL_PARAMS -notification_interval $NOTIFICATION_INTERVAL"
  fi
  
  echo "Email notifications: ENABLED"
else
  echo "Email notifications: DISABLED"
fi

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
  -max_recovery "$MAX_RECOVERY_ATTEMPTS" \
  $EMAIL_PARAMS 2>&1 | tee -a "$MODEL_DIR/logs/training_output.log"

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
