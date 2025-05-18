#!/bin/bash
# Script to run the CTC training with the fixed sample data

# First, ensure all semantic and agnostic files exist
echo "Step 1: Fixing sample data to ensure all semantic and agnostic files exist..."
python fix_sample_data.py ./sample_data/corpus ./sample_data/sample_set.txt

# Then, restore the original primus.py file if needed
echo "Step 2: Ensuring original primus.py file is in place..."
if [ -f "primus.py.backup" ]; then
  cp primus.py.backup primus.py
  echo "Restored original primus.py from backup."
fi

# Finally, run the training with recovery
echo "Step 3: Running CTC training with automatic recovery..."
./train_with_recovery.sh

echo "Training process complete!"
