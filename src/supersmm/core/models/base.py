"""Base model implementation for all machine learning models in the system.

This module provides the abstract base class that all ML models should inherit from.
It includes common functionality for model loading, saving, and basic operations.
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional

import tensorflow as tf


class BaseModel(ABC):
    """Abstract base class for all ML models in the system.

    Provides common functionality for model loading, initialization,
    and basic error handling. All concrete model implementations should
    inherit from this class.

    Attributes:
        model_path: Path to the model file or directory
        model: The loaded TensorFlow model
        logger: Logger instance for the model
        input_shape: Expected input shape of the model
        num_classes: Number of output classes
    """

    def __init__(self, model_path: Optional[str] = None):
        """Initialize the base model.

        Args:
            model_path: Optional path to a pre-trained model file or directory.
                       If provided, the model will be loaded during initialization.
        """
        self.model_path = Path(model_path) if model_path else None
        self.model: Optional[tf.keras.Model] = None
        self.logger = logging.getLogger(self.__class__.__name__)
        self.input_shape: Optional[tuple] = None
        self.num_classes: Optional[int] = None

        if self.model_path and self.model_path.exists():
            self.load()

    @abstractmethod
    def build(self, *args, **kwargs) -> tf.keras.Model:
        """Build the model architecture.

        This method must be implemented by subclasses to define the model's architecture.

        Returns:
            A compiled Keras model.
        """
        pass

    def load(self, model_path: Optional[str] = None) -> None:
        """Load a pre-trained model from disk.

        Args:
            model_path: Path to the model file or directory.
                      If None, uses the path provided during initialization.

        Raises:
            FileNotFoundError: If the specified model file doesn't exist.
            ValueError: If model loading fails.
        """
        path = Path(model_path) if model_path else self.model_path
        if path is None:
            raise ValueError("No model path provided")

        if not path.exists():
            raise FileNotFoundError(f"Model not found at {path}")

        try:
            self.logger.info("Loading model from %s", path)
            self.model = tf.keras.models.load_model(path)
            self.model_path = path
            self._update_model_properties()
            self.logger.info("Successfully loaded model from %s", path)
        except Exception as e:
            self.logger.error("Failed to load model from %s: %s", path, str(e))
            raise ValueError(f"Failed to load model: {str(e)}") from e

    def save(self, save_path: str) -> None:
        """Save the model to disk.

        Args:
            save_path: Path where the model should be saved.

        Raises:
            ValueError: If model is not built or saving fails.
        """
        if self.model is None:
            raise ValueError("Model has not been built or loaded")

        try:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            self.model.save(save_path)
            self.logger.info("Model saved to %s", save_path)
            self.model_path = save_path
        except Exception as e:
            self.logger.error("Failed to save model to %s: %s", save_path, str(e))
            raise ValueError(f"Failed to save model: {str(e)}") from e

    def predict(self, inputs: Any, **kwargs) -> Any:
        """Generate predictions for the given inputs.

        Args:
            inputs: Input data for prediction.
            **kwargs: Additional arguments to pass to the model's predict method.

        Returns:
            Model predictions.

        Raises:
            ValueError: If model is not built or loaded.
        """
        if self.model is None:
            raise ValueError("Model has not been built or loaded")
        return self.model.predict(inputs, **kwargs)

    def summary(self) -> None:
        """Print a summary of the model architecture.

        Raises:
            ValueError: If model is not built or loaded.
        """
        if self.model is None:
            raise ValueError("Model has not been built or loaded")
        self.model.summary()

    def _update_model_properties(self) -> None:
        """Update model properties based on the loaded model.

        This method is called after loading a model to update properties
        like input_shape and num_classes based on the loaded model.
        """
        if self.model is not None:
            if hasattr(self.model, "input_shape"):
                self.input_shape = self.model.input_shape[1:]  # Remove batch dimension
            if hasattr(self.model, "output_shape"):
                self.num_classes = self.model.output_shape[
                    -1
                ]  # Last dimension is num_classes
