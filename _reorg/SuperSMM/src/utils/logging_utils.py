"""Enhanced logging utilities with file rotation and categorization.

This module provides an enhanced logging system that:
1. Organizes logs by category and date
2. Implements file rotation based on line count
3. Uses structured logging format
4. Supports multiple output handlers
"""

import os
import logging
import datetime
from pathlib import Path
from typing import Optional, Dict, Any
import json


class RotatingFileLineHandler(logging.Handler):
    """A log handler that rotates files based on line count."""

    def __init__(
        self, base_path: str, category: str, prefix: str = "", max_lines: int = 200
    ):
        """Initialize the handler.

        Args:
            base_path: Base log directory path
            category: Log category (e.g., 'linting', 'testing')
            prefix: Optional prefix for log files
            max_lines: Maximum lines per log file
        """
        super().__init__()
        self.base_path = Path(base_path)
        self.category = category
        self.prefix = prefix
        self.max_lines = max_lines
        self.line_count = 0
        self.current_file = None
        self._setup_new_file()

    def _get_date_dir(self) -> Path:
        """Get the date-based directory path."""
        today = datetime.datetime.now().strftime("%Y-%m-%d")
        path = self.base_path / self.category / today
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _get_next_sequence(self, date_dir: Path) -> int:
        """Get the next sequence number for log files."""
        existing = list(date_dir.glob(f"{self.prefix}*_*.log"))
        if not existing:
            return 1
        sequences = [int(f.stem.split("_")[-1]) for f in existing]
        return max(sequences) + 1

    def _setup_new_file(self):
        """Set up a new log file."""
        date_dir = self._get_date_dir()
        seq = self._get_next_sequence(date_dir)
        filename = f"{self.prefix}report_{seq:03d}.log"
        self.current_file = date_dir / filename
        self.line_count = 0

    def emit(self, record: logging.LogRecord):
        """Emit a log record."""
        if self.line_count >= self.max_lines:
            self._setup_new_file()

        msg = self.format(record)
        with open(self.current_file, "a") as f:
            f.write(msg + "\n")
        self.line_count += 1


def get_logger(
    name: str,
    category: str,
    level: Optional[int] = None,
    base_log_dir: str = "logs",
    include_console: bool = True,
    structured: bool = True,
    **kwargs: Any,
) -> logging.Logger:
    """Get a logger with enhanced features.

    Args:
        name: Logger name
        category: Log category (e.g., 'linting', 'testing')
        level: Optional logging level
        base_log_dir: Base directory for logs
        include_console: Whether to include console output
        structured: Whether to use structured logging format
        **kwargs: Additional fields for structured logging

    Returns:
        Configured logger instance

    Example:
        >>> logger = get_logger('lint_check', 'linting',
        ...                    script_name='lint_and_fix.sh',
        ...                    target_file='main.py')
        >>> logger.info('Starting lint check')
        2025-05-13 23:30:15 - lint_check - INFO - {
            "message": "Starting lint check",
            "script_name": "lint_and_fix.sh",
            "target_file": "main.py"
        }
    """
    logger = logging.getLogger(name)

    if level is not None:
        logger.setLevel(level)

    if not logger.handlers:
        # File handler with rotation
        file_handler = RotatingFileLineHandler(
            base_log_dir, category, prefix=f"{category}_"
        )

        # Console handler if requested
        if include_console:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                )
            )
            logger.addHandler(console_handler)

        # Use structured logging if requested
        if structured:

            def structured_format(record):
                data = {
                    "timestamp": datetime.datetime.fromtimestamp(
                        record.created
                    ).isoformat(),
                    "logger": record.name,
                    "level": record.levelname,
                    "message": record.getMessage(),
                    **kwargs,
                }
                return json.dumps(data, indent=2)

            file_handler.setFormatter(logging.Formatter(structured_format))
        else:
            file_handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                )
            )

        logger.addHandler(file_handler)

    return logger
