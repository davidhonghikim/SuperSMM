#!/bin/bash

# Quick test script for OMR model training
# This script runs a minimal training session to verify the setup works correctly

# Set the base directory
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/.."

# Configuration
CORPUS="$BASE_DIR/sample_data/corpus"
SET="$BASE_DIR/sample_data/sample_set.txt"
VOCABULARY="$BASE_DIR/data/vocabulary_semantic.txt"
MODEL_DIR="$BASE_DIR/model/test_model"
CONFIG_FILE="$BASE_DIR/config/test_config.yaml"
EPOCHS=2
BATCH_SIZE=2

echo "==================================================================="
echo "Starting OMR test training with minimal configuration"
echo "==================================================================="
echo "Corpus: $CORPUS"
echo "Set file: $SET"
echo "Model directory: $MODEL_DIR"
echo "Epochs: $EPOCHS"
echo "==================================================================="

# Create model directory if it doesn't exist
mkdir -p "$MODEL_DIR"
mkdir -p "$MODEL_DIR/logs"

# First, check if the sample data exists
if [ ! -d "$CORPUS" ] || [ ! -f "$SET" ]; then
    echo "Sample data not found. Creating sample dataset..."
    ./create_sample_dataset.sh
    
    # Verify sample data was created
    if [ ! -d "$CORPUS" ] || [ ! -f "$SET" ]; then
        echo "ERROR: Failed to create sample dataset!"
        exit 1
    fi
fi

# Create test config file if it doesn't exist
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Creating test configuration file..."
    cat > "$CONFIG_FILE" << EOF
# Test training configuration
corpus: $CORPUS
set_file: $SET
vocabulary: $VOCABULARY
save_model: $MODEL_DIR/model
semantic: true
epochs: $EPOCHS
batch_size: $BATCH_SIZE
validation_split: 0.1
img_height: 128
dropout: 0.5
learning_rate: 0.001
EOF
fi

# Check if the deep-omr command is available
if command -v deep-omr &> /dev/null; then
    # Run using the CLI
    echo "Running test training using the deep-omr CLI..."
    deep-omr train -config "$CONFIG_FILE" 2>&1 | tee -a "$MODEL_DIR/logs/test_output.log"
else
    # Run using the Python script
    echo "Running test training using the Python script..."
    python "$BASE_DIR/train.py" -config "$CONFIG_FILE" 2>&1 | tee -a "$MODEL_DIR/logs/test_output.log"
fi

# Check if training completed successfully
if [ $? -eq 0 ]; then
    echo "==================================================================="
    echo "Test training completed successfully!"
    echo "==================================================================="
    echo "You can now run the full training with:"
    echo "./scripts/run_training.sh"
    echo ""
    echo "Or using the configuration file:"
    echo "./scripts/run_training.sh -config ./config/training_config.yaml"
    echo "==================================================================="
    exit 0
else
    echo "==================================================================="
    echo "Test training failed! Check the logs for errors."
    echo "Log file: $MODEL_DIR/logs/test_output.log"
    echo "==================================================================="
    exit 1
fi
