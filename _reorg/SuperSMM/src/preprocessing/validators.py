"""Validation utilities for preprocessing operations.

This module contains validation functions used across the preprocessing package
to ensure data integrity and proper parameter values.
"""

from typing import Any, Tuple
import numpy as np
import logging

logger = logging.getLogger(__name__)


def validate_image(image: Any, allow_color: bool = True) -> Tuple[bool, str]:
    """Validate an input image array.

    Args:
        image: Input to validate
        allow_color: Whether to allow 3-channel color images

    Returns:
        Tuple of (is_valid, error_message)
    """
    if image is None:
        return False, "Input image is None"

    if not isinstance(image, np.ndarray):
        return False, f"Expected numpy.ndarray, got {type(image)}"

    if len(image.shape) < 2 or (not allow_color and len(image.shape) != 2):
        return False, f"Invalid image dimensions: {image.shape}"

    if len(image.shape) > 3:
        return False, f"Too many dimensions: {image.shape}"

    if len(image.shape) == 3 and image.shape[2] not in (1, 3, 4):
        return False, f"Invalid number of channels: {image.shape[2]}"

    return True, ""


def validate_size_range(
    size: int, min_size: int, max_size: int, param_name: str
) -> Tuple[bool, str]:
    """Validate that a size parameter falls within an acceptable range.

    Args:
        size: Size value to check
        min_size: Minimum acceptable size
        max_size: Maximum acceptable size
        param_name: Name of parameter for error messages

    Returns:
        Tuple of (is_valid, error_message)
    """
    if size < min_size:
        return False, f"{param_name} ({size}) is below minimum ({min_size})"
    if size > max_size:
        return False, f"{param_name} ({size}) exceeds maximum ({max_size})"
    return True, ""


def ensure_odd(value: int, param_name: str) -> int:
    """Ensure a parameter value is odd, adjusting if necessary.

    Args:
        value: Input value
        param_name: Name of parameter for logging

    Returns:
        Odd-numbered value (input value + 1 if even)
    """
    if value % 2 == 0:
        logger.warning(
            f"{param_name} must be odd, got {value}. "
            f"Automatically adjusting to {value + 1}."
        )
        return value + 1
    return value
