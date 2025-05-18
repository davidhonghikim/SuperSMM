#!/bin/bash

# Exit on error
set -e

# Set up environment
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
# Add both project root and src directory to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:${PROJECT_ROOT}:${PROJECT_ROOT}/src"

# Create necessary directories
mkdir -p "${PROJECT_ROOT}/models" "${PROJECT_ROOT}/logs" "${PROJECT_ROOT}/checkpoints"

# Set default values
EPOCHS=50
BATCH_SIZE=16
CONFIG_FILE="${PROJECT_ROOT}/src/tf-deep-omr/config/training_config.yaml"
CORPUS="${PROJECT_ROOT}/src/tf-deep-omr/src/Data/primus"
TRAIN_SET="${PROJECT_ROOT}/src/tf-deep-omr/src/Data/train.txt"
VOCABULARY="${PROJECT_ROOT}/src/tf-deep-omr/src/Data/vocabulary_semantic.txt"
MODEL_DIR="${PROJECT_ROOT}/models/Semantic-Model"

# Print configuration
echo "=== Training Configuration ==="
echo "Project Root: ${PROJECT_ROOT}"
echo "Config File: ${CONFIG_FILE}"
echo "Corpus: ${CORPUS}"
echo "Train Set: ${TRAIN_SET}"
echo "Vocabulary: ${VOCABULARY}"
echo "Model Dir: ${MODEL_DIR}"
echo "Epochs: ${EPOCHS}"
echo "Batch Size: ${BATCH_SIZE}"
echo "============================"

# Check if required files exist
for file in "${CONFIG_FILE}" "${TRAIN_SET}" "${VOCABULARY}"; do
    if [ ! -f "${file}" ]; then
        echo "Error: Required file not found: ${file}" >&2
        exit 1
    fi
done

if [ ! -d "${CORPUS}" ]; then
    echo "Error: Corpus directory not found: ${CORPUS}" >&2
    exit 1
fi

# Run the training script
echo "Running training script..."
cd "${PROJECT_ROOT}/src/tf-deep-omr/src"

# First run with -h to check if module is found
python -c "import sys; print('Python path:', sys.path)" > "${PROJECT_ROOT}/logs/python_path.log" 2>&1

# Run the training script with the correct argument format for ctc_training.py
python ctc_training.py \
    -corpus "${CORPUS}" \
    -set "${TRAIN_SET}" \
    -vocabulary "${VOCABULARY}" \
    -save_model "${MODEL_DIR}" \
    -semantic 2>&1 | tee "${PROJECT_ROOT}/logs/training.log"

# Print completion message
echo "Training completed. Check ${PROJECT_ROOT}/logs/ for training logs and ${PROJECT_ROOT}/models/ for saved models."
