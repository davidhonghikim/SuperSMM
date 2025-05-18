#!/bin/bash

# Simple script to test the OMR system with the example image
echo "Testing OMR system with example image..."

# Test with Agnostic model
echo "Testing with Agnostic model..."
python test_model.py -model Data/Models/Agnostic\ Model/agnostic_model.meta -vocabulary Data/vocabulary_agnostic.txt -image Data/Example/000051652-1_2_1.png

# Test with Semantic model
echo -e "\nTesting with Semantic model..."
python test_model.py -model Data/Models/Semantic-Model/semantic_model.meta -vocabulary Data/vocabulary_semantic.txt -image Data/Example/000051652-1_2_1.png

echo -e "\nTests completed successfully!"
