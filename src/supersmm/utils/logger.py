"""
Centralized logging utility for the SuperSMM project.

This module provides a consistent way to handle logging across the entire application,
with support for different log levels and output destinations.

Example usage:
    from utils.logger import setup_logger
    
    # Basic usage
    logger = setup_logger(__name__)
    logger.info("This is an info message")
    
    # With custom log level
    logger = setup_logger(__name__, log_level=logging.DEBUG)
    
    # For ML training logs
    logger = setup_logger("training", log_type="ml")
"""

import logging
import os
import sys
from pathlib import Path
from typing import Optional, Literal

# Define log types and their corresponding directories
LOG_TYPES = {
    "app": "app",
    "debug": "debug",
    "linting": "linting",
    "maintenance": "maintenance",
    "ml": "ml",  # Machine learning logs
    "system": "system"
}

def setup_logger(
    name: str = "supersmm",
    log_type: Literal["app", "debug", "linting", "maintenance", "ml", "system"] = "app",
    log_level: int = logging.INFO,
    log_to_console: bool = True,
    log_to_file: bool = True,
    log_dir: Optional[str] = None
) -> logging.Logger:
    """
    Configure and return a logger with the specified settings.
    
    Args:
        name: Name of the logger (usually __name__)
        log_type: Type of log (determines subdirectory in logs/)
        log_level: Logging level (e.g., logging.DEBUG, logging.INFO)
        log_to_console: Whether to log to console
        log_to_file: Whether to log to file
        log_dir: Custom log directory (defaults to project_root/logs/type/)
        
    Returns:
        Configured logger instance
    """
    # Get project root (assuming this file is in src/utils/)
    project_root = Path(__file__).resolve().parent.parent
    
    # Set up log directory
    if log_dir is None:
        log_dir = project_root.parent / "logs" / LOG_TYPES.get(log_type, "app")
    else:
        log_dir = Path(log_dir)
    
    # Ensure log directory exists
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    
    # Clear any existing handlers to prevent duplicate logs
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # File Handler
    if log_to_file:
        log_file = log_dir / f"{name}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Console Handler
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    return logger

# Create a default logger for easy imports
logger = setup_logger("supersmm", log_type="app")
