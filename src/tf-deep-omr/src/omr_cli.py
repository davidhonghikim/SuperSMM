#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Command-line interface for the OMR system
"""

import argparse
import os
import subprocess
import sys
import glob

def print_header():
    """Print a fancy header for the CLI"""
    print("\n" + "=" * 80)
    print("                     OMR (Optical Music Recognition) CLI                     ")
    print("=" * 80 + "\n")

def print_section(title):
    """Print a section title"""
    print("\n" + "-" * 80)
    print(f" {title}")
    print("-" * 80 + "\n")

def run_command(command, cwd=None):
    """Run a command and print its output"""
    print(f"Running: {' '.join(command)}\n")
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        cwd=cwd
    )
    
    # Print output in real-time
    for line in iter(process.stdout.readline, ''):
        print(line, end='')
    
    process.stdout.close()
    return_code = process.wait()
    
    if return_code != 0:
        print(f"\nCommand failed with return code {return_code}")
    else:
        print("\nCommand completed successfully")
    
    return return_code

def find_models():
    """Find available models in the Data/Models directory"""
    models = []
    
    # Look for Agnostic Model
    agnostic_meta = glob.glob("Data/Models/Agnostic*/agnostic_model.meta")
    if agnostic_meta:
        models.append({
            'name': 'Agnostic',
            'meta': agnostic_meta[0],
            'vocabulary': 'Data/vocabulary_agnostic.txt'
        })
    
    # Look for Semantic Model
    semantic_meta = glob.glob("Data/Models/Semantic*/semantic_model.meta")
    if semantic_meta:
        models.append({
            'name': 'Semantic',
            'meta': semantic_meta[0],
            'vocabulary': 'Data/vocabulary_semantic.txt'
        })
    
    return models

def test_single_image(args):
    """Test a single image with a model"""
    print_section("Testing Single Image")
    
    # Find available models
    models = find_models()
    if not models:
        print("No models found. Please check the Data/Models directory.")
        return 1
    
    # Select model
    model_name = getattr(args, 'model', None)
    if model_name:
        selected_model = next((m for m in models if m['name'].lower() == model_name.lower()), None)
        if not selected_model:
            print(f"Model '{model_name}' not found. Available models: {', '.join(m['name'] for m in models)}")
            return 1
    else:
        print("Available models:")
        for i, model in enumerate(models):
            print(f"{i+1}. {model['name']}")
        
        choice = input("\nSelect a model (number): ")
        try:
            selected_model = models[int(choice) - 1]
        except (ValueError, IndexError):
            print("Invalid choice. Exiting.")
            return 1
    
    # Get image path
    image_path = getattr(args, 'image', None)
    if not image_path:
        image_path = input("Enter path to the image file: ")
    
    if not os.path.isfile(image_path):
        print(f"Image file not found: {image_path}")
        return 1
    
    # Run test_model.py
    command = [
        "python", "test_model.py",
        "-model", selected_model['meta'],
        "-vocabulary", selected_model['vocabulary'],
        "-image", image_path
    ]
    
    return run_command(command)

def batch_test(args):
    """Batch test multiple images"""
    print_section("Batch Testing")
    
    # Find available models
    models = find_models()
    if not models:
        print("No models found. Please check the Data/Models directory.")
        return 1
    
    # Get directory path
    directory = getattr(args, 'directory', None)
    if not directory:
        directory = input("Enter path to the directory containing images: ")
    
    if not os.path.isdir(directory):
        print(f"Directory not found: {directory}")
        return 1
    
    # Get output file
    output_file = getattr(args, 'output', None)
    if not output_file:
        output_file = input("Enter path for the output JSON file (or leave empty for no output file): ")
    
    # Get comparison file
    comparison_file = getattr(args, 'comparison', None)
    if not comparison_file and len(models) > 1:
        comparison_file = input("Enter path for the comparison JSON file (or leave empty for no comparison file): ")
    
    # Build command
    command = ["python", "batch_test_model.py", "-directory", directory]
    
    # Add models and vocabularies
    use_all_models = getattr(args, 'all_models', False)
    if use_all_models or len(models) > 1 and input("Use all available models? (y/n): ").lower() == 'y':
        command.extend(["-models"] + [m['meta'] for m in models])
        command.extend(["-vocabularies"] + [m['vocabulary'] for m in models])
    else:
        # Select a single model
        model_name = getattr(args, 'model', None)
        if model_name:
            selected_model = next((m for m in models if m['name'].lower() == model_name.lower()), None)
            if not selected_model:
                print(f"Model '{model_name}' not found. Available models: {', '.join(m['name'] for m in models)}")
                return 1
        else:
            print("Available models:")
            for i, model in enumerate(models):
                print(f"{i+1}. {model['name']}")
            
            choice = input("\nSelect a model (number): ")
            try:
                selected_model = models[int(choice) - 1]
            except (ValueError, IndexError):
                print("Invalid choice. Exiting.")
                return 1
        
        command.extend(["-models", selected_model['meta']])
        command.extend(["-vocabularies", selected_model['vocabulary']])
    
    # Add output and comparison files
    if output_file:
        command.extend(["-output", output_file])
    
    if comparison_file:
        command.extend(["-comparison", comparison_file])
    
    # Add workers
    workers = getattr(args, 'workers', None)
    if workers:
        command.extend(["-workers", str(workers)])
    
    return run_command(command)

def start_visualizer(args):
    """Start the web visualizer"""
    print_section("Starting Web Visualizer")
    
    # Find available models
    models = find_models()
    if not models:
        print("No models found. Please check the Data/Models directory.")
        return 1
    
    # Build command
    command = ["python", "omr_visualizer.py"]
    
    # Add models and vocabularies
    command.extend(["-models"] + [m['meta'] for m in models])
    command.extend(["-vocabularies"] + [m['vocabulary'] for m in models])
    
    # Add port
    port = getattr(args, 'port', None)
    if port:
        command.extend(["-port", str(port)])
    
    # Add debug mode
    debug = getattr(args, 'debug', False)
    if debug:
        command.append("-debug")
    
    return run_command(command)

def setup_environment(args):
    """Run the setup script"""
    print_section("Setting Up Environment")
    
    # Check if setup.sh exists
    if not os.path.isfile("setup.sh"):
        print("Setup script not found. Creating it...")
        
        # Create setup.sh
        with open("setup.sh", "w") as f:
            f.write("""#!/bin/bash

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
chmod +x test_model.py batch_test_model.py omr_visualizer.py omr_cli.py

# Create necessary directories
echo "Creating necessary directories..."
mkdir -p static/uploads static/results templates logs

echo "Setup complete!"
""")
        
        # Make it executable
        os.chmod("setup.sh", 0o755)
    
    # Run setup.sh
    return run_command(["bash", "setup.sh"])

def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Command-line interface for the OMR system')
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Test single image
    test_parser = subparsers.add_parser('test', help='Test a single image')
    test_parser.add_argument('-model', help='Model to use (Agnostic or Semantic)')
    test_parser.add_argument('-image', help='Path to the image file')
    
    # Batch test
    batch_parser = subparsers.add_parser('batch', help='Batch test multiple images')
    batch_parser.add_argument('-model', help='Model to use (Agnostic or Semantic)')
    batch_parser.add_argument('-all-models', action='store_true', help='Use all available models')
    batch_parser.add_argument('-directory', help='Path to the directory containing images')
    batch_parser.add_argument('-output', help='Path for the output JSON file')
    batch_parser.add_argument('-comparison', help='Path for the comparison JSON file')
    batch_parser.add_argument('-workers', type=int, default=4, help='Number of worker threads')
    
    # Visualizer
    visualizer_parser = subparsers.add_parser('visualize', help='Start the web visualizer')
    visualizer_parser.add_argument('-port', type=int, default=5000, help='Port to run the server on')
    visualizer_parser.add_argument('-debug', action='store_true', help='Run the server in debug mode')
    
    # Setup
    setup_parser = subparsers.add_parser('setup', help='Setup the environment')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Print header
    print_header()
    
    # Run command
    if args.command == 'test':
        return test_single_image(args)
    elif args.command == 'batch':
        return batch_test(args)
    elif args.command == 'visualize':
        return start_visualizer(args)
    elif args.command == 'setup':
        return setup_environment(args)
    else:
        # No command specified, show help
        parser.print_help()
        
        # If we have models, show a menu
        models = find_models()
        if models:
            print("\nAvailable models:")
            for model in models:
                print(f"- {model['name']}")
            
            print("\nAvailable commands:")
            print("1. Test a single image")
            print("2. Batch test multiple images")
            print("3. Start the web visualizer")
            print("4. Setup the environment")
            print("5. Exit")
            
            choice = input("\nEnter your choice (1-5): ")
            
            if choice == '1':
                return test_single_image(args)
            elif choice == '2':
                return batch_test(args)
            elif choice == '3':
                return start_visualizer(args)
            elif choice == '4':
                return setup_environment(args)
            else:
                print("Exiting.")
                return 0
        
        return 0

if __name__ == '__main__':
    sys.exit(main())
