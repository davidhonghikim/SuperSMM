#!/bin/bash
# Script to stop the dashboard servers (log proxy, React frontend, and simple server)

cd "$(dirname "$0")" || exit

PID_DIR=".pids"
mkdir -p "$PID_DIR"

# Define PID files
PROXY_PID_FILE="$PID_DIR/proxy.pid"
REACT_PID_FILE="$PID_DIR/react.pid"
SERVER_PID_FILE="$PID_DIR/server.pid"

# Function to stop process with PID file
stop_process() {
  local pid_file=$1
  local process_name=$2
  
  if [ -f "$pid_file" ]; then
    PID=$(cat "$pid_file")
    if ps -p "$PID" > /dev/null 2>&1; then
      echo "Stopping $process_name (PID: $PID)..."
      kill "$PID" > /dev/null 2>&1
      sleep 1
      
      # Force kill if still running
      if ps -p "$PID" > /dev/null 2>&1; then
        echo "Force stopping $process_name (PID: $PID)..."
        kill -9 "$PID" > /dev/null 2>&1
      fi
      
      echo "$process_name stopped"
    else
      echo "$process_name not running (PID: $PID)"
    fi
    
    rm "$pid_file"
  else
    echo "No PID file found for $process_name"
  fi
}

# Stop all services
echo "Stopping dashboard services..."

stop_process "$PROXY_PID_FILE" "Log proxy server"
stop_process "$REACT_PID_FILE" "React dashboard"
stop_process "$SERVER_PID_FILE" "Simple server"

# Force kill any remaining dashboard processes
echo "Ensuring all dashboard processes are stopped..."
pkill -f "node LogProxy.js" > /dev/null 2>&1 || true
pkill -f "node simple-server.js" > /dev/null 2>&1 || true
pkill -f "react-scripts start" > /dev/null 2>&1 || true

echo "Dashboard services stopped"

# Optional: Clean up log files
# echo "Cleaning up log files (logproxy.out, react_dashboard.out)..."
# rm -f logproxy.out react_dashboard.out

echo "Done."
