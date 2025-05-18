#!/bin/bash
# Script to simulate a training continuation run

# Get current epoch
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
BASE_DIR="$(dirname "$SCRIPT_DIR")"
TF_OMR_DIR="$BASE_DIR/src/tf-deep-omr"
TRAINING_STATE_FILE="$TF_OMR_DIR/checkpoints/training_state.txt"

# Default increment 
INCREMENT=10
if [ ! -z "$1" ]; then
    INCREMENT=$1
fi

# Check if training state exists
if [ -f "$TRAINING_STATE_FILE" ]; then
    CURRENT_EPOCH=$(cat "$TRAINING_STATE_FILE")
    echo "Current epoch: $CURRENT_EPOCH"
else
    echo "No training state found. Creating initial state."
    mkdir -p "$(dirname "$TRAINING_STATE_FILE")"
    CURRENT_EPOCH=0
fi

# Calculate new epoch
NEW_EPOCH=$((CURRENT_EPOCH + INCREMENT))
echo "Simulating training run, advancing from epoch $CURRENT_EPOCH to $NEW_EPOCH"

# Run the test script to create the new state
python "$SCRIPT_DIR/test_training_continuation.py" --create-state --epoch $NEW_EPOCH

# Verify the state
python "$SCRIPT_DIR/test_training_continuation.py" --verify

# Update the dashboard
echo "Updating dashboard data..."
python "$SCRIPT_DIR/fix_csv.py"

echo ""
echo "Simulation complete! Training has advanced to epoch $NEW_EPOCH"
echo "To continue training, you would run: $SCRIPT_DIR/continue_training.sh"
echo "To view the dashboard: cd $BASE_DIR/src/dashboard && ./start_dashboard_fixed.sh" 