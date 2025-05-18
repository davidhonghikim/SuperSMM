#!/bin/bash

# Stop all servers
echo "Stopping all servers..."

# Stop dashboard server
if [ -f "logs/dashboard.pid" ]; then
  DASHBOARD_PID=$(cat logs/dashboard.pid)
  if ps -p $DASHBOARD_PID > /dev/null; then
    kill $DASHBOARD_PID
    echo "Dashboard server stopped"
  else
    echo "Dashboard server already stopped"
  fi
  rm logs/dashboard.pid
fi

# Stop training monitor
cd ..
if [ -f "monitor.pid" ]; then
  MONITOR_PID=$(cat monitor.pid)
  if ps -p $MONITOR_PID > /dev/null; then
    kill $MONITOR_PID
    echo "Training monitor stopped"
  else
    echo "Training monitor already stopped"
  fi
  rm monitor.pid
fi

# Force kill any remaining processes
pkill -f "node.*LogProxy" || true
pkill -f "node.*simple-server" || true
pkill -f "npm.*start" || true
pkill -f "python.*monitor_training" || true
pkill -f "node.*serve" || true

echo "All servers stopped" 