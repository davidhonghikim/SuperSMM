#!/bin/bash

# SuperSMM Launch Script

# Navigate to project root
cd "$(dirname "$0")/.."

# Activate virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    
    # Install dependencies
    echo "Installing dependencies..."
    pip install -r DEPENDENCIES.md
fi

# Set PYTHONPATH to include the project root
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Run the application
python3 src/main.py
