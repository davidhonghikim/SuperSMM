#!/bin/bash

# Script to process the example OMR image

# Define Variables
OMR_CLI_PATH="./omr_cli.py"
INPUT_IMAGE_DIR="Data/Example"
INPUT_IMAGE_NAME="000051652-1_2_1.png"
INPUT_IMAGE_PATH="$INPUT_IMAGE_DIR/$INPUT_IMAGE_NAME"

OUTPUT_DIR_BASE="output/example"

AGNOSTIC_MODEL_PATH="Data/Models/Agnostic_Model"
AGNOSTIC_VOCAB_PATH="Data/vocabulary_agnostic.txt"

SEMANTIC_MODEL_PATH="Data/Models/Semantic-Model"
SEMANTIC_VOCAB_PATH="Data/vocabulary_semantic.txt"

# Create output directories
mkdir -p "$OUTPUT_DIR_BASE"

# --- Process with Agnostic Model ---
echo "Processing with Agnostic Model: $INPUT_IMAGE_NAME"
OUTPUT_PREFIX_AGNOSTIC="${OUTPUT_DIR_BASE}/Agnostic_${INPUT_IMAGE_NAME%.*}"

python3 "$OMR_CLI_PATH" test \
    -image "$INPUT_IMAGE_PATH" \
    -model Agnostic


echo "Agnostic model processing complete. Outputs at ${OUTPUT_PREFIX_AGNOSTIC}.*"

# --- Process with Semantic Model ---
echo "Processing with Semantic Model: $INPUT_IMAGE_NAME"
OUTPUT_PREFIX_SEMANTIC="${OUTPUT_DIR_BASE}/Semantic_${INPUT_IMAGE_NAME%.*}"

python3 "$OMR_CLI_PATH" test \
    -image "$INPUT_IMAGE_PATH" \
    -model Semantic


echo "Semantic model processing complete. Outputs at ${OUTPUT_PREFIX_SEMANTIC}.*"

# Copy original image and ground truth files to output for comparison
echo "Copying original image and ground truth files to $OUTPUT_DIR_BASE"
cp "$INPUT_IMAGE_PATH" "$OUTPUT_DIR_BASE/"
cp "$INPUT_IMAGE_DIR/${INPUT_IMAGE_NAME%.*}.agnostic" "$OUTPUT_DIR_BASE/"
cp "$INPUT_IMAGE_DIR/${INPUT_IMAGE_NAME%.*}.semantic" "$OUTPUT_DIR_BASE/"
cp "$INPUT_IMAGE_DIR/${INPUT_IMAGE_NAME%.*}.mei" "$OUTPUT_DIR_BASE/"

echo "Example processing finished."
