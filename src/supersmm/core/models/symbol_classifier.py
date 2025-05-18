"""Symbol classifier implementation for musical symbol recognition.

This module provides a CNN-based classifier for recognizing musical symbols
in sheet music. It extends the BaseModel class and provides additional
functionality specific to symbol classification.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

from .base import BaseModel


class SymbolClassifier(BaseModel):
    """CNN-based classifier for musical symbol recognition.

    This class implements a convolutional neural network for classifying
    musical symbols in sheet music. It supports both training and inference
    modes and can be used as part of a larger OMR pipeline.

    Attributes:
        input_shape: Expected input shape (height, width, channels)
        num_classes: Number of symbol classes
        class_labels: List of class names
        class_indices: Mapping from class names to indices
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        input_shape: Tuple[int, int, int] = (64, 64, 1),
        num_classes: int = 10,
        class_labels: Optional[List[str]] = None,
    ):
        """Initialize the symbol classifier.

        Args:
            model_path: Path to a pre-trained model file or directory
            input_shape: Expected input shape (height, width, channels)
            num_classes: Number of symbol classes
            class_labels: Optional list of class names. If not provided,
                        default class names will be used.
        """
        super().__init__(model_path)
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.class_labels = class_labels or self._get_default_class_labels()
        self.class_indices = {label: i for i, label in enumerate(self.class_labels)}

        if self.model is None:
            self.model = self.build()

    def _get_default_class_labels(self) -> List[str]:
        """Get default class labels for musical symbols.

        Returns:
            List of default class names.
        """
        return [
            "whole_note",
            "half_note",
            "quarter_note",
            "eighth_note",
            "sixteenth_note",
            "whole_rest",
            "half_rest",
            "quarter_rest",
            "eighth_rest",
            "sixteenth_rest",
            "sharp",
            "flat",
            "natural",
            "time_signature",
            "barline",
            "dot",
            "tie",
            "slur",
            "fermata",
            "other",
        ]

    def build(self) -> tf.keras.Model:
        """Build the symbol classifier model architecture.

        Returns:
            A compiled Keras model.
        """
        self.logger.info("Building symbol classifier model")

        inputs = layers.Input(shape=self.input_shape, name="input_layer")

        # Convolutional base
        x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)

        x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)

        x = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)

        # Dense classifier
        x = layers.Flatten()(x)
        x = layers.Dense(256, activation="relu")(x)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(self.num_classes, activation="softmax")(x)

        # Create model
        model = models.Model(inputs=inputs, outputs=outputs, name="symbol_classifier")

        # Compile model
        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        return model

    def preprocess_input(self, image: np.ndarray) -> np.ndarray:
        """Preprocess input image for prediction.

        Args:
            image: Input image as numpy array

        Returns:
            Preprocessed image
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Resize to model input shape
        image = cv2.resize(image, (self.input_shape[1], self.input_shape[0]))

        # Normalize pixel values
        image = image.astype("float32") / 255.0

        # Add batch dimension
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=-1)

        return image

    def predict_symbols(
        self, images: Union[np.ndarray, List[np.ndarray]]
    ) -> List[Dict[str, float]]:
        """Predict symbols from input images.

        Args:
            images: Batch of input images or single image

        Returns:
            List of prediction results, each containing class probabilities
        """
        if self.model is None:
            raise ValueError("Model has not been loaded or built")

        # Handle single image input
        if isinstance(images, np.ndarray) and len(images.shape) == 3:
            images = [images]

        # Preprocess images
        processed_images = np.array([self.preprocess_input(img) for img in images])

        # Make predictions
        predictions = self.model.predict(processed_images)

        # Convert to list of class probabilities
        results = []
        for pred in predictions:
            results.append(
                {label: float(prob) for label, prob in zip(self.class_labels, pred)}
            )

        return results
