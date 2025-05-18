"""
Symbol Classifier Module

Implements the CNN-based classifier for musical symbol recognition.
"""

# Standard library imports
from typing import List, Dict, Any
import logging

# Third-party imports
import numpy as np
import tensorflow as tf

# Local imports
from .base_model import BaseModel


class SymbolClassifier(BaseModel):
    """CNN-based musical symbol classifier.

    Implements symbol classification using a convolutional neural network.
    Inherits basic model functionality from BaseModel.

    Attributes:
        input_shape (tuple): Expected input shape for the model
        num_classes (int): Number of symbol classes
        class_labels (List[str]): Names of symbol classes
    """

    def __init__(self, model_path: str = "resources/ml_models/symbol_classifier"):
        """Initialize the symbol classifier.

        Args:
            model_path (str): Path to the trained model
        """
        super().__init__(model_path)
        self.logger = logging.getLogger(__name__)

        # Load model and initialize
        self.model = self._load_model()
        self.input_shape = (64, 64, 1)  # Standard input size
        self._initialize_classes()

    def _initialize_classes(self):
        """Initialize the symbol classes."""
        self.class_labels = [
            "whole_note",
            "half_note",
            "quarter_note",
            "eighth_note",
            "sixteenth_note",
            "treble_clef",
            "bass_clef",
            "sharp",
            "flat",
            "natural",
            "bar_line",
        ]
        self.num_classes = len(self.class_labels)

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess an image for classification.

        Args:
            image (np.ndarray): Input image

        Returns:
            np.ndarray: Preprocessed image
        """
        # Resize to standard input size
        resized = tf.image.resize(image, (64, 64))

        # Ensure grayscale
        if len(resized.shape) > 2:
            resized = tf.image.rgb_to_grayscale(resized)

        # Normalize to [0,1]
        normalized = resized / 255.0

        # Add batch dimension
        return tf.expand_dims(normalized, 0)

    def predict(self, images: List[np.ndarray]) -> List[Dict[str, Any]]:
        """Predict symbol classes for a batch of images.

        Args:
            images (List[np.ndarray]): List of symbol images

        Returns:
            List[Dict[str, Any]]: Predictions with class labels and confidence
        """
        try:
            # Preprocess all images
            processed = [self.preprocess_image(img) for img in images]
            batch = tf.concat(processed, axis=0)

            # Get predictions
            predictions = self.model.predict(batch)

            # Format results
            results = []
            for pred in predictions:
                class_idx = np.argmax(pred)
                confidence = float(pred[class_idx])
                results.append(
                    {
                        "label": self.class_labels[class_idx],
                        "confidence": confidence,
                        "class_index": int(class_idx),
                    }
                )

            return results

        except Exception as e:
            self.logger.error("Prediction failed: %s", e)
            return [{"error": str(e)} for _ in images]
