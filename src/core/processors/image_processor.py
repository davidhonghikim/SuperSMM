"""
Image processing utilities for OMR.

This module provides image preprocessing functions specifically designed
for sheet music recognition, including noise reduction and binarization.
"""

import cv2
import logging
import numpy as np
from typing import Dict, Any

logger = logging.getLogger("image_processor")


def preprocess_image(image: np.ndarray) -> Dict[str, Any]:
    """
    Advanced image preprocessing for OMR.

    Performs:
    - Grayscale conversion
    - Noise reduction
    - Adaptive thresholding

    Args:
        image (np.ndarray): Input image

    Returns:
        Dict[str, Any]: Preprocessed images and metadata
            - 'grayscale': Grayscale image
            - 'denoised': Denoised image
            - 'binary': Final binary image
            - 'threshold_params': Parameters used for thresholding
    """
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Denoise
        denoised = cv2.fastNlMeansDenoising(gray)

        # Adaptive thresholding parameters
        block_size = 11
        c_value = 2

        # Adaptive thresholding
        binary = cv2.adaptiveThreshold(
            denoised,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            block_size,
            c_value,
        )

        result = {
            "grayscale": gray,
            "denoised": denoised,
            "binary": binary,
            "threshold_params": {"block_size": block_size, "c_value": c_value},
        }

        logger.debug("Image preprocessing completed successfully")
        return result

    except Exception as e:
        logger.error(f"Image preprocessing error: {e}")
        # Handle invalid inputs more gracefully
        if image is None or image.size == 0:
            return {
                "grayscale": None,
                "denoised": None,
                "binary": None,
                "error": "Invalid input image",
            }

        # If image is already grayscale, return as is
        if len(image.shape) == 2:
            return {
                "grayscale": image,
                "denoised": None,
                "binary": None,
                "error": str(e),
            }

        try:
            # Try one more time to convert to grayscale
            return {
                "grayscale": cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),
                "denoised": None,
                "binary": None,
                "error": str(e),
            }
        except Exception:
            # If all else fails, return None
            return {
                "grayscale": None,
                "denoised": None,
                "binary": None,
                "error": "Failed to process image",
            }
