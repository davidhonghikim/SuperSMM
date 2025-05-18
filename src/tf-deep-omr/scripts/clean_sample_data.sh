#!/bin/bash

# Script to clean up the sample data directory and regenerate it correctly

echo "Cleaning up sample data directory..."
rm -rf ./sample_data

echo "Regenerating sample data from fixed set file..."
python prepare_corpus.py -corpus "./data" -set "./data/train_fixed.txt" -sample 100 -output "./sample_data"

echo "Verifying sample data was created correctly..."
if [ -d "./sample_data/corpus" ] && [ -f "./sample_data/sample_set.txt" ]; then
    echo "Sample data created successfully!"
    echo "Number of samples in set file: $(wc -l < ./sample_data/sample_set.txt)"
    echo "Number of image files in corpus: $(find ./sample_data/corpus -name "*.png" | wc -l)"
else
    echo "ERROR: Failed to create sample dataset!"
    exit 1
fi

echo "Done!"
