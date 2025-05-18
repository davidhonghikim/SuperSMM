"""
Symbol Recognizer Core Module

Core implementation of the symbol recognition model.
"""

import numpy as np
import tensorflow as tf
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

from ..config.recognizer_config import SymbolRecognizerConfig
from .preprocessor import SymbolPreprocessor
from .model_loader import ModelLoader


class SymbolRecognizerCore:
    """Core implementation of the symbol recognition model."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the symbol recognizer.

        Args:
            config: Optional configuration dictionary
        """
        # Setup logging
        self.logger = logging.getLogger(__name__)

        # Initialize configuration
        default_config = SymbolRecognizerConfig()
        self.config = {k: v for k, v in default_config.__dict__.items()}
        if config:
            self.config.update(config)

        # Initialize components
        self.model_loader = ModelLoader(logger=self.logger)
        self.preprocessor = SymbolPreprocessor(input_shape=self.config["input_shape"])

        # Configure TensorFlow
        self.model_loader.configure_tensorflow()

        # Load model and vocabulary
        self.model = self.model_loader.load_model(self.config["model_path"])
        self.class_labels = self.model_loader.load_vocabulary(self.config["vocab_path"])
        self.config["num_classes"] = len(self.class_labels)

    def predict(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Predict symbol class for a single image.

        Args:
            image: Input image

        Returns:
            List of top-k predictions with confidence scores
        """
        # Preprocess image
        processed_image = self.preprocessor.preprocess_symbol(image)
        processed_image = processed_image.reshape(1, *processed_image.shape)

        # Predict probabilities
        symbol_pred = self.model.predict(processed_image)[0]

        # Get top-k predictions
        top_k_indices = np.argsort(symbol_pred)[-self.config["top_k_predictions"] :][
            ::-1
        ]

        # Create predictions list with class labels and confidence scores
        predictions = []
        for idx in top_k_indices:
            if symbol_pred[idx] >= self.config["confidence_threshold"]:
                predictions.append(
                    {
                        "label": self.class_labels[idx],
                        "confidence": float(symbol_pred[idx]),
                        "class_index": int(idx),
                    }
                )

        return predictions

    def predict_batch(self, images: List[np.ndarray]) -> List[List[Dict[str, Any]]]:
        """Predict symbol classes for a batch of images.

        Args:
            images: List of input images

        Returns:
            List of prediction lists, one for each input image
        """
        if not images:
            return []

        # Preprocess batch
        processed_batch = self.preprocessor.preprocess_batch(images)

        # Predict probabilities
        batch_predictions = self.model.predict(processed_batch)

        # Process each prediction
        results = []
        for symbol_pred in batch_predictions:
            # Get top-k predictions
            top_k_indices = np.argsort(symbol_pred)[
                -self.config["top_k_predictions"] :
            ][::-1]

            # Create predictions list with class labels and confidence scores
            predictions = []
            for idx in top_k_indices:
                if symbol_pred[idx] >= self.config["confidence_threshold"]:
                    predictions.append(
                        {
                            "label": self.class_labels[idx],
                            "confidence": float(symbol_pred[idx]),
                            "class_index": int(idx),
                        }
                    )

            results.append(predictions)

        return results
