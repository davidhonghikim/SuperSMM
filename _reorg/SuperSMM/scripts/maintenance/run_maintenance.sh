#!/bin/bash
# run_maintenance.sh
# Runs all maintenance tasks in the correct order with proper logging

set -e

# Setup logging
TIMESTAMP=$(date +%Y-%m-%d)
LOG_DIR="../../logs/maintenance/$TIMESTAMP"
mkdir -p "$LOG_DIR"

# Python script for structured logging
cat > "$(dirname "$0")/maint_log_helper.py" << 'EOF'
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent / 'src'))
from utils.logging_utils import get_logger
logger = get_logger('maintenance', 'maintenance', structured=True)
logger.info(sys.stdin.read())
EOF

# Log function that uses structured logging
log() {
    echo "$1" | python3 "$(dirname "$0")/maint_log_helper.py"
}

# Change to project root
cd ../..

# 1. Rotate and archive logs
log "Starting log rotation and archival"
python3 scripts/maintenance/log_manager.py --rotate
python3 scripts/maintenance/log_manager.py --stats

# 2. Run linting and fixes
log "Running code quality checks"
bash scripts/dev/lint_and_fix.sh

# 3. Run tests with auto-fixes
log "Running tests with auto-fixes"
bash scripts/dev/batch_test_and_fix.sh

# 4. Update asset index
log "Updating asset index"
python3 scripts/maintenance/asset_index_cli.py update

# 5. Run optimization analysis
log "Running optimization analysis"
python3 scripts/optimization/pipeline_optimizer.py

# 6. Validate project structure
log "Validating project structure"
python3 scripts/validation/validate_project.py

# Cleanup
rm "$(dirname "$0")/maint_log_helper.py"

log "Maintenance tasks completed successfully"

# Generate final stats
python3 scripts/maintenance/log_manager.py --stats
