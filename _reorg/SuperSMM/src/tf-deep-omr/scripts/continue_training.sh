#!/bin/bash
# Script to continue training from the last checkpoint

# Navigate to the project root
cd "$(dirname "$0")/.." || exit

# Check if training is already running
if pgrep -f "python src/ctc_training.py" > /dev/null; then
  echo "Error: Training appears to be already running."
  echo "If you're sure it's not running, delete any stale PID files and try again."
  exit 1
fi

# Configuration variables
SCRIPT_DIR="$(pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
MODEL_DIR="$SCRIPT_DIR/model/primus_model"
LOG_DIR="$SCRIPT_DIR/logs"
TRAINING_STATE="$MODEL_DIR/training_state.txt"
OUTPUT_LOG="$MODEL_DIR/logs/training_output.log"

# Create directories if they don't exist
mkdir -p "$MODEL_DIR/logs"
mkdir -p "$LOG_DIR"

# Find the latest checkpoint epoch
if [ -f "$TRAINING_STATE" ]; then
  LAST_EPOCH=$(cat "$TRAINING_STATE")
  echo "Found training state: Epoch $LAST_EPOCH"
else
  echo "No training state found. Looking for checkpoints..."
  
  # Try to find checkpoints
  LATEST_CHECKPOINT=$(find "$MODEL_DIR" -name "model-*.meta" | sort -V | tail -1)
  
  if [ -n "$LATEST_CHECKPOINT" ]; then
    LAST_EPOCH=$(echo "$LATEST_CHECKPOINT" | grep -o 'model-[0-9]*' | cut -d'-' -f2)
    echo "Found checkpoint at epoch $LAST_EPOCH"
  else
    echo "No checkpoints found. Starting from epoch 0."
    LAST_EPOCH=0
  fi
  
  # Write to training state file
  echo "$LAST_EPOCH" > "$TRAINING_STATE"
fi

# Determine paths for the training dataset
if [ -d "$SCRIPT_DIR/src/Data/primus" ]; then
  CORPUS="$SCRIPT_DIR/src/Data/primus"
elif [ -d "$SCRIPT_DIR/src/Data/camera_primus/Corpus" ]; then
  CORPUS="$SCRIPT_DIR/src/Data/camera_primus/Corpus"
else
  echo "Error: Could not find dataset directory."
  echo "Please make sure one of these directories exists:"
  echo "  - $SCRIPT_DIR/src/Data/primus"
  echo "  - $SCRIPT_DIR/src/Data/camera_primus/Corpus"
  exit 1
fi

# Find the set file
if [ -f "$SCRIPT_DIR/data/primus_set.txt" ]; then
  SET_FILE="$SCRIPT_DIR/data/primus_set.txt"
elif [ -f "$SCRIPT_DIR/data/camera_primus_set.txt" ]; then
  SET_FILE="$SCRIPT_DIR/data/camera_primus_set.txt"
else
  echo "Error: Could not find set file."
  echo "Please make sure one of these files exists:"
  echo "  - $SCRIPT_DIR/data/primus_set.txt"
  echo "  - $SCRIPT_DIR/data/camera_primus_set.txt"
  exit 1
fi

# Find vocabulary file
VOCABULARY="$SCRIPT_DIR/data/vocabulary_semantic.txt"
if [ ! -f "$VOCABULARY" ]; then
  echo "Error: Could not find vocabulary file at $VOCABULARY"
  exit 1
fi

# Determine model name based on dataset
MODEL_NAME="$(basename "$CORPUS")_model_semantic"
if [[ "$SET_FILE" == *"agnostic"* ]]; then
  MODEL_NAME="$(basename "$CORPUS")_model_agnostic"
fi

echo "==================================================================="
echo "Resuming OMR model training from epoch $LAST_EPOCH"
echo "==================================================================="
echo "Dataset: $CORPUS"
echo "Set file: $SET_FILE"
echo "Vocabulary: $VOCABULARY"
echo "Model directory: $MODEL_DIR"
echo "Model name: $MODEL_NAME"
echo "==================================================================="

# Allow memory growth instead of pre-allocating all GPU memory
export TF_FORCE_GPU_ALLOW_GROWTH=true

# Run the training with resume flag
echo "Starting training process..."
python src/ctc_training.py \
  -corpus "$CORPUS" \
  -set "$SET_FILE" \
  -save_model "$MODEL_DIR/model" \
  -vocabulary "$VOCABULARY" \
  -semantic \
  -resume 2>&1 | tee -a "$OUTPUT_LOG"

echo "==================================================================="
echo "Training complete!"
echo "===================================================================" 