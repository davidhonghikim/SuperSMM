#!/bin/bash
# batch_test_and_fix.sh
# Runs all test files with proper PYTHONPATH and attempts auto-fix for import errors.
# Uses structured logging for better tracking and analysis.

set -e

# Setup logging
TIMESTAMP=$(date +%Y-%m-%d)
LOG_DIR="../../logs/testing/$TIMESTAMP"
mkdir -p "$LOG_DIR"

# Python script for structured logging
cat > "$(dirname "$0")/test_log_helper.py" << 'EOF'
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent / 'src'))
from utils.logging_utils import get_logger
logger = get_logger('batch_test', 'testing', structured=True)
logger.info(sys.stdin.read())
EOF

# Log function that uses structured logging
log() {
    echo "$1" | python3 "$(dirname "$0")/test_log_helper.py"
}

# Set up environment
cd ../..
export PYTHONPATH="$(pwd)/src"

# Find all test files
TEST_FILES=$(find tests -type f -name "test_*.py" -not -path "*/venv/*")

# Process each test file
for testfile in $TEST_FILES; do
    log "Running test file: $testfile"
    
    # Run test and capture output
    TEST_OUTPUT=$(python3 "$testfile" 2>&1) || {
        ERROR_MSG="$TEST_OUTPUT"
        log "Test failed: $testfile\nError: $ERROR_MSG"
        
        if echo "$ERROR_MSG" | grep -q "ModuleNotFoundError"; then
            log "Attempting to fix import errors in $testfile"
            
            # Backup the file
            cp "$testfile" "${testfile}.bak"
            
            # Auto-fix common import patterns
            sed -i '' \
                -e 's/from \([a-z_]*\)\./from src.\1./g' \
                -e 's/import \([a-z_]*\)$/import src.\1/g' \
                "$testfile"
            
            # Retry test
            RETRY_OUTPUT=$(python3 "$testfile" 2>&1) || {
                log "Fix failed. Restoring backup of $testfile"
                mv "${testfile}.bak" "$testfile"
                continue
            }
            
            rm "${testfile}.bak"
            log "Successfully fixed imports in $testfile"
            log "Test output after fix:\n$RETRY_OUTPUT"
        fi
        continue
    }
    
    log "Test passed: $testfile\nOutput:\n$TEST_OUTPUT"
done

# Cleanup
rm "$(dirname "$0")/test_log_helper.py"

log "Batch test and auto-fix process completed"
