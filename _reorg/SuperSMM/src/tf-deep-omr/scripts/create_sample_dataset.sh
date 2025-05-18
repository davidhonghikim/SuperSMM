#!/bin/bash

# Create a small sample dataset for testing the training pipeline
# This script creates a smaller version of the dataset for quick testing

# Configuration
CORPUS="./Data"
ORIGINAL_SET="./data/train_fixed.txt"
SAMPLE_SIZE=100
OUTPUT_DIR="./sample_data"

echo "==================================================================="
echo "Creating sample dataset with $SAMPLE_SIZE samples"
echo "==================================================================="
echo "Source corpus: $CORPUS"
echo "Source set file: $ORIGINAL_SET"
echo "Output directory: $OUTPUT_DIR"
echo "==================================================================="

# Make the script executable
chmod +x prepare_corpus.py

# Create the sample dataset
python prepare_corpus.py \
  -corpus "$CORPUS" \
  -set "$ORIGINAL_SET" \
  -sample "$SAMPLE_SIZE" \
  -output "$OUTPUT_DIR"

echo "==================================================================="
echo "Sample dataset created successfully!"
echo "To use the sample dataset for training, update train_with_recovery.sh:"
echo "CORPUS=\"$OUTPUT_DIR/corpus\""
echo "SET=\"$OUTPUT_DIR/sample_set.txt\""
echo "==================================================================="
