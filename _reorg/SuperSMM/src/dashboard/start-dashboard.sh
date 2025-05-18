#!/bin/bash

# Stop any existing servers
echo "Stopping existing servers..."
pkill -f "node.*LogProxy" || true
pkill -f "node.*simple-server" || true
pkill -f "npm.*start" || true
pkill -f "python.*monitor_training" || true
pkill -f "node.*serve" || true

# Wait for processes to terminate
sleep 2

# Create log directory if it doesn't exist
mkdir -p logs
mkdir -p ../../logs

# Check if training log exists
if [ ! -f "../../logs/training_run.csv" ]; then
  echo "Warning: Training log file not found. Create an empty one."
  echo "timestamp,epoch,loss,validation_error,ser_percent,batch_size,dataset_size,memory_usage_mb,gpu_memory_mb,checkpoint_path" > ../../logs/training_run.csv
fi

# Start the simple web server for dashboard
echo "Starting dashboard server..."
node simple-server.js > logs/dashboard.log 2>&1 &
echo $! > logs/dashboard.pid
echo "Dashboard server started on http://localhost:8000"

# Start training monitor in a separate terminal (if needed)
# Commented out as we're not using monitor_training.py anymore
# echo "Starting training monitor..."
# cd ..
# python tf-deep-omr/src/monitor_training.py --log-file logs/training_run.csv > monitor.log 2>&1 &
# echo $! > monitor.pid
# echo "Training monitor started"

echo ""
echo "Dashboard is now available at: http://localhost:8000"
echo "Use the following command to follow logs:"
echo "  tail -f src/dashboard/logs/dashboard.log"
echo ""
echo "To stop all servers, run: bash stop-dashboard.sh" 