"""
Image Preprocessor Module

Handles basic image preprocessing operations for the OMR system.
"""

# Standard library imports
import logging
from typing import Dict, Any

# Third-party imports
import cv2
import numpy as np


class ImagePreprocessor:
    """Basic image preprocessing operations.

    Handles common preprocessing tasks like grayscale conversion,
    noise reduction, and thresholding.

    Attributes:
        logger (logging.Logger): Logger instance
    """

    def __init__(self):
        """Initialize the preprocessor."""
        self.logger = logging.getLogger(__name__)

    def to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """Convert image to grayscale.

        Args:
            image (np.ndarray): Input image

        Returns:
            np.ndarray: Grayscale image
        """
        try:
            print(
                f"[to_grayscale] dtype: {image.dtype}, shape: {image.shape}, min: {image.min() if hasattr(image, 'min') else 'n/a'}, max: {image.max() if hasattr(image, 'max') else 'n/a'}"
            )
            # Ensure uint8
            if image.dtype != np.uint8:
                print(f"[to_grayscale] Casting image from {image.dtype} to uint8")
                image = image.astype(np.uint8)
            # If already grayscale
            if len(image.shape) == 2:
                return image
            # If shape is (H, W, 1), squeeze
            if len(image.shape) == 3 and image.shape[2] == 1:
                print(
                    "[to_grayscale] Squeezing last dimension from (H, W, 1) to (H, W)"
                )
                image = np.squeeze(image, axis=2)
                return image
            # If RGBA, convert to RGB first
            if len(image.shape) == 3 and image.shape[2] == 4:
                print("[to_grayscale] Converting RGBA to RGB before grayscale")
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            # Standard BGR/RGB to grayscale
            if len(image.shape) == 3 and image.shape[2] == 3:
                return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            print(
                f"[to_grayscale] Unsupported image shape for grayscale: {image.shape}"
            )
            return image
        except Exception as e:
            print(f"[to_grayscale] Grayscale conversion failed: {e}")
            self.logger.error("Grayscale conversion failed: %s", e)
            return image

    def denoise(self, image: np.ndarray) -> np.ndarray:
        """Apply noise reduction.

        Args:
            image (np.ndarray): Input image

        Returns:
            np.ndarray: Denoised image
        """
        try:
            print(f"[denoise] image dtype: {image.dtype}, shape: {image.shape}")
            return cv2.fastNlMeansDenoising(image)
        except Exception as e:
            print(f"[denoise] Exception: {e}")
            self.logger.error("Denoising failed: %s", e)
            return image

    def threshold(self, image: np.ndarray) -> np.ndarray:
        """Apply adaptive thresholding.

        Args:
            image (np.ndarray): Input grayscale image

        Returns:
            np.ndarray: Binary image
        """
        try:
            print(f"[threshold] image dtype: {image.dtype}, shape: {image.shape}")
            return cv2.adaptiveThreshold(
                image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
            )
        except Exception as e:
            print(f"[threshold] Exception: {e}")
            self.logger.error("Thresholding failed: %s", e)
            return image

    def preprocess(self, image: np.ndarray) -> Dict[str, Any]:
        """Apply full preprocessing pipeline.

        Args:
            image (np.ndarray): Input image

        Returns:
            Dict[str, Any]: Preprocessed images and metadata
        """
        try:
            print(
                f"[preprocess] input dtype: {getattr(image, 'dtype', None)}, shape: {getattr(image, 'shape', None)}"
            )
            # Handle invalid inputs
            if image is None or image.size == 0:
                print("[preprocess] Invalid input image")
                return {
                    "grayscale": None,
                    "denoised": None,
                    "binary": None,
                    "error": "Invalid input image",
                }
            # Convert to grayscale
            gray = self.to_grayscale(image)
            print(
                f"[preprocess] grayscale dtype: {getattr(gray, 'dtype', None)}, shape: {getattr(gray, 'shape', None)}"
            )
            # Apply denoising
            denoised = self.denoise(gray)
            print(
                f"[preprocess] denoised dtype: {getattr(denoised, 'dtype', None)}, shape: {getattr(denoised, 'shape', None)}"
            )
            # Apply thresholding
            binary = self.threshold(denoised)
            print(
                f"[preprocess] binary dtype: {getattr(binary, 'dtype', None)}, shape: {getattr(binary, 'shape', None)}"
            )
            return {
                "grayscale": gray,
                "denoised": denoised,
                "binary": binary,
                "threshold_params": {"block_size": 11, "c_value": 2},
            }
        except Exception as e:
            print(f"[preprocess] Exception: {e}")
            self.logger.error("Preprocessing pipeline failed: %s", e)
            return {
                "grayscale": None,
                "denoised": None,
                "binary": None,
                "error": str(e),
            }
