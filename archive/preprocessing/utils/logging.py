"""Logging utilities for preprocessing module.

This module provides consistent logging configuration across the preprocessing package.
All loggers use the same format and can optionally have their level set at creation.

Typical usage:
    logger = get_logger(__name__)  # For normal logging
    perf_logger = get_logger("performance")  # For performance metrics
"""

import logging
from typing import Optional


def get_logger(name: str, level: Optional[int] = None) -> logging.Logger:
    """Get a configured logger instance with consistent formatting.

    Args:
        name: Logger name (typically __name__ or a specific category like 'performance')
        level: Optional logging level to set (e.g., logging.DEBUG)

    Returns:
        Logger instance with consistent formatting and optional level setting

    Example:
        >>> logger = get_logger(__name__, logging.DEBUG)
        >>> logger.debug("Processing started")
        2025-05-13 22:46:00 - mymodule - DEBUG - Processing started
    """
    logger = logging.getLogger(name)

    if level is not None:
        logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger
