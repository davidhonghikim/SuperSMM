#!/bin/bash
# Script to auto-run backend log proxy, frontend React app, and open dashboard in browser

cd "$(dirname "$0")" || exit

PID_DIR=".pids"
mkdir -p "$PID_DIR"

# First, sync logs to ensure they're available in all needed locations
echo "Syncing logs..."
./sync_logs.sh

# Stop any existing servers
echo "Stopping existing servers..."
bash ./stop_dashboard.sh

# Create log directory if it doesn't exist
mkdir -p logs

# 1. Start the Node.js log proxy (background)
echo "Starting log proxy server..."
nohup node LogProxy.js > logs/logproxy.out 2>&1 &
PROXY_PID=$!
echo "Log proxy PID: $PROXY_PID"
echo "$PROXY_PID" > "$PID_DIR/proxy.pid"

# 2. Start the React dashboard (background)
echo "Starting React dashboard (npm start)..."
nohup npm start > logs/react_dashboard.out 2>&1 &
FRONTEND_PID=$!
echo "React dashboard PID: $FRONTEND_PID"
echo "$FRONTEND_PID" > "$PID_DIR/react.pid"

# 3. Also start simple non-React server for basic dashboard
echo "Starting simple dashboard server..."
nohup node simple-server.js > logs/simple_server.out 2>&1 &
SIMPLE_PID=$!
echo "Simple server PID: $SIMPLE_PID"
echo "$SIMPLE_PID" > "$PID_DIR/server.pid"

# 4. Wait for servers to initialize
echo "Waiting for servers to initialize (8s)..."
sleep 8

# 5. Display information about the current model
TRAINING_STATE="../tf-deep-omr/model/primus_model/training_state.txt"
if [ -f "$TRAINING_STATE" ]; then
  CURRENT_EPOCH=$(cat "$TRAINING_STATE")
  echo "Current training state: Epoch $CURRENT_EPOCH"
else
  echo "No training state found."
fi

# 6. Print dashboard URLs
cat <<EOM

---
Dashboard servers are running!
- Log proxy: PID $PROXY_PID
- React dashboard: PID $FRONTEND_PID
- Simple HTML dashboard: PID $SIMPLE_PID

Available dashboards:
- React Dashboard: http://localhost:3000/
- Simple HTML Dashboard: http://localhost:8000/

To stop the dashboard servers, run:
  ./stop_dashboard.sh
--- 
EOM

# 7. Open dashboard in browser (optional, uncomment if desired)
# URL="http://localhost:3000/"
# echo "Attempting to open dashboard at $URL ..."
# if which xdg-open > /dev/null; then
#   xdg-open "$URL"
# elif which open > /dev/null; then
#   open "$URL"
# fi
