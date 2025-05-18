"""
Base Model Module

Provides the base class for all machine learning models in the system.
"""

# Standard library imports
import logging
from typing import Optional
from pathlib import Path

# Third-party imports
import tensorflow as tf


class BaseModel:
    """Base class for all ML models.

    Provides common functionality for model loading, initialization,
    and basic error handling.

    Attributes:
        model_path (Path): Path to the model file
        model (tf.keras.Model): The loaded TensorFlow model
        logger (logging.Logger): Logger instance
    """

    def __init__(self, model_path: str):
        """Initialize the base model.

        Args:
            model_path (str): Path to the model file
        """
        self.model_path = Path(model_path)
        self.model: Optional[tf.keras.Model] = None
        self.logger = logging.getLogger(__name__)

    def _load_model(self) -> tf.keras.Model:
        """Load model from disk.

        Returns:
            tf.keras.Model: The loaded model

        Raises:
            FileNotFoundError: If model file doesn't exist
            ValueError: If model loading fails
        """
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found at {self.model_path}")

        try:
            model = tf.keras.models.load_model(str(self.model_path))
            self.logger.info("Model loaded successfully from %s", self.model_path)
            return model
        except Exception as e:
            self.logger.error("Failed to load model: %s", e)
            raise ValueError(f"Model loading failed: {e}")
