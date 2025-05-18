"""
Model Loader Module

Handles loading and validation of TensorFlow models for symbol recognition.
"""

import os
import logging
from pathlib import Path
import tensorflow as tf


class ModelLoader:
    """Loads and validates TensorFlow models for symbol recognition."""

    def __init__(self, logger=None):
        """Initialize the model loader.

        Args:
            logger: Optional logger instance
        """
        self.logger = logger or logging.getLogger(__name__)

    def load_model(self, model_path: str) -> tf.keras.Model:
        """Load a TensorFlow model from the specified path.

        Args:
            model_path: Path to the model file

        Returns:
            Loaded TensorFlow model

        Raises:
            FileNotFoundError: If model file does not exist
            ValueError: If model loading fails
        """
        path = Path(model_path)
        if not path.exists():
            self.logger.error(f"Model file not found at {path}")
            raise FileNotFoundError(f"Model file not found at {path}")

        try:
            model = tf.keras.models.load_model(str(path))
            self.logger.info(f"Successfully loaded model from {path}")
            return model
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise ValueError(f"Failed to load model: {e}")

    def load_vocabulary(self, vocab_path: str) -> list:
        """Load class labels from vocabulary file.

        Args:
            vocab_path: Path to vocabulary file

        Returns:
            List of class labels

        Raises:
            FileNotFoundError: If vocabulary file does not exist
        """
        path = Path(vocab_path)
        if not path.exists():
            self.logger.error(f"Vocabulary file not found at {path}")
            raise FileNotFoundError(f"Vocabulary file not found at {path}")

        try:
            with open(path, "r") as f:
                class_labels = [line.strip() for line in f.readlines()]
            self.logger.info(f"Loaded {len(class_labels)} class labels from {path}")
            return class_labels
        except Exception as e:
            self.logger.error(f"Failed to load vocabulary: {e}")
            raise ValueError(f"Failed to load vocabulary: {e}")

    def configure_tensorflow(self):
        """Configure TensorFlow for optimal performance."""
        # Memory growth configuration for GPUs
        gpus = tf.config.experimental.list_physical_devices("GPU")
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                self.logger.info(f"Configured memory growth for {len(gpus)} GPUs")
            except RuntimeError as e:
                self.logger.warning(f"Error configuring GPU memory: {e}")

        # Set random seed for reproducibility
        tf.random.set_seed(42)
