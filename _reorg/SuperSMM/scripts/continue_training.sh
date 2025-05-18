#!/bin/bash

# Script to continue training OMR model from the last checkpoint

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
BASE_DIR="$(dirname "$SCRIPT_DIR")"
TF_OMR_DIR="$BASE_DIR/src/tf-deep-omr"

# Make sure checkpoint directory exists
mkdir -p "$TF_OMR_DIR/checkpoints"
mkdir -p "$TF_OMR_DIR/logs"
mkdir -p "$BASE_DIR/logs"

# Get training state to check progress
TRAINING_STATE_FILE="$TF_OMR_DIR/Users/danger/CascadeProjects/LOO/SuperSMM/src/tf-deep-omr/src/Data/primus/training_state.txt"
MODEL_DIR="$TF_OMR_DIR/checkpoints"
CURRENT_EPOCH=0

if [ -f "$TRAINING_STATE_FILE" ]; then
    CURRENT_EPOCH=$(cat "$TRAINING_STATE_FILE")
    echo "Current training progress: Epoch $CURRENT_EPOCH"
else
    echo "No training state found. Starting from the beginning."
fi

# Default parameters 
CORPUS="$TF_OMR_DIR/Users/danger/CascadeProjects/LOO/SuperSMM/src/tf-deep-omr/src/Data/primus"
SET_FILE="$TF_OMR_DIR/Users/danger/CascadeProjects/LOO/SuperSMM/src/tf-deep-omr/src/Data/train.txt"  
VOC_FILE="$TF_OMR_DIR/Users/danger/CascadeProjects/LOO/SuperSMM/src/tf-deep-omr/src/Data/vocabulary_semantic.txt"
SAVE_MODEL="$TF_OMR_DIR/Users/danger/CascadeProjects/LOO/SuperSMM/src/tf-deep-omr/model/primus_model"
TOTAL_EPOCHS=100000
BATCH_SIZE=50

# Check for custom parameters
if [ ! -z "$1" ]; then
    CORPUS="$1"
fi

if [ ! -z "$2" ]; then
    SET_FILE="$2"
fi

if [ ! -z "$3" ]; then
    VOC_FILE="$3"
fi

if [ ! -z "$4" ]; then
    SAVE_MODEL="$4"
fi

echo "========================================"
echo "Continuing OMR model training"
echo "========================================"
echo "Corpus: $CORPUS"
echo "Set file: $SET_FILE"
echo "Vocabulary: $VOC_FILE"
echo "Model: $SAVE_MODEL"
echo "Current epoch: $CURRENT_EPOCH"
echo "========================================"
echo 

# Run the training script with resume flag
cd "$TF_OMR_DIR" || exit 1

# Determine log file
LOG_FILE="$TF_OMR_DIR/logs/training_$(date +%Y%m%d_%H%M%S).log"

echo "Starting training... (Logging to $LOG_FILE)"
python src/ctc_training.py -corpus "$CORPUS" -set "$SET_FILE" -vocabulary "$VOC_FILE" -save_model "$SAVE_MODEL" -resume 2>&1 | tee "$LOG_FILE"

# Copy training log to the main logs directory 
if [ -f "$TF_OMR_DIR/logs/training_log.csv" ]; then
    cp "$TF_OMR_DIR/logs/training_log.csv" "$BASE_DIR/logs/training_run.csv"
    echo "Training log copied to $BASE_DIR/logs/training_run.csv"
    
    # Update dashboard
    echo "Updating dashboard data..."
    python "$BASE_DIR/scripts/fix_csv.py" || {
        echo "Warning: Error updating CSV data for dashboard. Dashboard may show incomplete information."
    }
else
    echo "Warning: Training log file not found at $TF_OMR_DIR/logs/training_log.csv"
    if [ -f "$BASE_DIR/logs/training_run.csv" ]; then
        echo "Previous training log exists in $BASE_DIR/logs/training_run.csv"
    else
        echo "No training logs found. Dashboard may not show any data."
    fi
fi

echo 
echo "Training completed. To continue training, run this script again."
echo "To view the training dashboard, run: cd $BASE_DIR/src/dashboard && ./start_dashboard_fixed.sh" 