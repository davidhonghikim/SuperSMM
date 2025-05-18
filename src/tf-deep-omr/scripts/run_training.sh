#!/bin/bash

# Script to run the training process using the new modular structure
# This script uses the new train.py entry point with the YAML configuration file

# Set the base directory
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/.."

# Default config file
CONFIG_FILE="$BASE_DIR/config/training_config.yaml"

# Function to display usage information
usage() {
  echo "Usage: $0 [options]"
  echo "Options:"
  echo "  -config <file>    Path to the configuration file (default: $CONFIG_FILE)"
  echo "  -corpus <dir>     Path to the corpus directory"
  echo "  -set <file>       Path to the set file"
  echo "  -save_model <dir> Path to save the model"
  echo "  -vocabulary <file> Path to the vocabulary file"
  echo "  -epochs <num>     Number of epochs to train for"
  echo "  -batch_size <num> Batch size for training"
  echo "  -semantic         Use semantic vocabulary"
  echo "  -agnostic         Use agnostic vocabulary"
  echo "  -h, --help        Display this help message"
  exit 1
}

# Parse command line arguments
ARGS=""
while [[ $# -gt 0 ]]; do
  case $1 in
    -h|--help)
      usage
      ;;
    -config)
      CONFIG_FILE="$2"
      ARGS="$ARGS -config $CONFIG_FILE"
      shift 2
      ;;
    -corpus|-set|-save_model|-vocabulary|-epochs|-batch_size)
      ARGS="$ARGS $1 $2"
      shift 2
      ;;
    -semantic|-agnostic)
      ARGS="$ARGS $1"
      shift
      ;;
    *)
      echo "Unknown option: $1"
      usage
      ;;
  esac
done

# Check if the deep-omr command is available
if command -v deep-omr &> /dev/null; then
    # Construct the command using the CLI
    CMD="deep-omr train $ARGS"
    
    # Print the command
    echo "Running: $CMD"
    
    # Execute the command
    $CMD
else
    # Construct the command using the Python script
    CMD="python $BASE_DIR/train.py $ARGS"
    
    # Print the command
    echo "Running: $CMD"
    
    # Execute the command
    $CMD
fi
