#!/bin/bash

# Stop any existing servers
echo "Stopping existing servers..."
pkill -f "node.*simple-server" || true

# Wait for processes to terminate
sleep 2

# Create log directory if it doesn't exist
mkdir -p logs

# Check if training log exists, if not use the one in the root logs dir
if [ ! -f "../../logs/training_run.csv" ]; then
  echo "Warning: Training log file not found at ../../logs/training_run.csv"
  exit 1
fi

# Start the simple web server for dashboard
echo "Starting dashboard server..."
node simple-server.js > server.log 2>&1 &
SERVER_PID=$!
echo $SERVER_PID > server.pid
echo "Dashboard server started on http://localhost:8000 (PID: $SERVER_PID)"

echo ""
echo "Dashboard is now available at: http://localhost:8000"
echo "Use the following command to follow logs:"
echo "  tail -f server.log"
echo ""
echo "To stop the server, run: kill $(cat server.pid)" 