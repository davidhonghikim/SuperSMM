"""Preprocessing utilities package.

This package provides common utilities used across the preprocessing module:
- Logging configuration and management
- (Future) Performance monitoring
- (Future) File handling utilities
- (Future) Image validation helpers
"""

from .logging import get_logger

__all__ = ["get_logger"]

# Configure root logger for the preprocessing package
logger = get_logger("preprocessing")
