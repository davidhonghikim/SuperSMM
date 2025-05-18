"""Preprocessing utilities package.

This package provides common utilities used across the preprocessing module:
- Performance monitoring
- File handling utilities
- Image validation helpers
"""

from ...utils.logger import setup_logger
import logging

# Configure logger for the preprocessing utils
logger = setup_logger(
    name="preprocessing.utils",
    log_type="app",
    log_level=logging.INFO,
    log_to_console=True,
    log_to_file=True
)
