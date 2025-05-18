#!/bin/bash

# Setup script for OMR model testing
echo "Setting up OMR model testing environment..."

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install required packages
echo "Installing required packages..."
pip install tensorflow opencv-python numpy flask tqdm

# Make scripts executable
echo "Making scripts executable..."
chmod +x test_model.py batch_test_model.py omr_visualizer.py

# Create necessary directories
echo "Creating necessary directories..."
mkdir -p static/uploads static/results templates logs

echo "Setup complete!"
echo "You can now run the following commands:"
echo "1. Test a single image with a model:"
echo "   ./test_model.py -model <model_path> -vocabulary <vocabulary_path> -image <image_path>"
echo ""
echo "2. Batch test multiple images:"
echo "   ./batch_test_model.py -directory <directory_path>"
echo ""
echo "3. Launch the web visualizer:"
echo "   ./omr_visualizer.py"
