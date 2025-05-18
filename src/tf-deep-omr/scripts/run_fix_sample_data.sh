#!/bin/bash
# Script to run fix_sample_data.py with the correct parameters

echo "Running fix_sample_data.py to ensure all semantic and agnostic files exist..."
python fix_sample_data.py ./sample_data/corpus ./sample_data/sample_set.txt

echo "Done!"
