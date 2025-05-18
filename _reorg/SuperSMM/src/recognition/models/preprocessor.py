"""
Symbol Preprocessor Module

Handles preprocessing of symbol images for recognition.
"""

import numpy as np
import tensorflow as tf
import cv2
from typing import List


class SymbolPreprocessor:
    """Preprocesses symbol images for recognition."""

    def __init__(self, input_shape=(64, 64, 1)):
        """Initialize the preprocessor with target input shape.

        Args:
            input_shape: Target shape for preprocessed images (height, width, channels)
        """
        self.input_shape = input_shape

    def preprocess_symbol(self, symbol_image: np.ndarray) -> np.ndarray:
        """Preprocess a single symbol image for recognition.

        Args:
            symbol_image: Raw symbol image

        Returns:
            Preprocessed image ready for model input
        """
        # Handle empty images
        if symbol_image is None or symbol_image.size == 0:
            return np.zeros(self.input_shape)

        # Resize to standard input size
        if symbol_image.shape[:2] != self.input_shape[:2]:
            resized = cv2.resize(
                symbol_image, (self.input_shape[1], self.input_shape[0])
            )
        else:
            resized = symbol_image.copy()

        # Ensure grayscale
        if len(resized.shape) > 2 and resized.shape[2] > 1:
            resized = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

        # Ensure correct dimensions
        if len(resized.shape) == 2:
            resized = resized.reshape(*resized.shape, 1)

        # Normalize to [0,1]
        normalized = resized.astype(np.float32) / 255.0

        return normalized

    def preprocess_batch(self, symbol_images: List[np.ndarray]) -> np.ndarray:
        """Preprocess a batch of symbol images.

        Args:
            symbol_images: List of raw symbol images

        Returns:
            Batch of preprocessed images
        """
        processed_symbols = [self.preprocess_symbol(img) for img in symbol_images]
        return np.array(processed_symbols)
