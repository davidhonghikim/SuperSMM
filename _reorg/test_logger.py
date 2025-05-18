"""
Test script for the enhanced logger utility.
"""
import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.utils.logger import setup_logger

def test_logger():
    """Test different logging scenarios."""
    # Test default logger (app logs)
    app_logger = setup_logger("test_app")
    app_logger.info("This is an app info message")
    app_logger.error("This is an app error message")
    
    # Test ML logger
    ml_logger = setup_logger("test_ml", log_type="ml", log_level=logging.DEBUG)
    ml_logger.debug("Debug message for ML")
    ml_logger.info("ML training started")
    
    # Test file-only logger
    file_logger = setup_logger(
        "test_file_only", 
        log_to_console=False,
        log_to_file=True
    )
    file_logger.info("This will only appear in the log file")
    
    print("\nLogger test completed. Check the logs/ directory for output files.")

if __name__ == "__main__":
    test_logger()
