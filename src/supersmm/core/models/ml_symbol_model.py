"""Machine learning model for musical symbol recognition.

This module provides a TensorFlow-based model for recognizing musical symbols
in sheet music. It's designed to work as part of the OMR pipeline and supports
both training and inference modes.
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, applications

from .base import BaseModel
from .symbol_classifier import SymbolClassifier


class MLSymbolModel(BaseModel):
    """Machine learning model for musical symbol recognition.

    This class implements a deep learning model for recognizing musical symbols
    in sheet music. It can be used for both training and inference and supports
    transfer learning from pre-trained models.

    Attributes:
        input_shape: Expected input shape (height, width, channels)
        num_classes: Number of symbol classes
        class_names: List of class names
        use_transfer_learning: Whether to use transfer learning
        base_model: Base model for transfer learning
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        input_shape: Tuple[int, int, int] = (256, 256, 3),
        num_classes: int = 20,
        class_names: Optional[List[str]] = None,
        use_transfer_learning: bool = True,
        freeze_base: bool = True,
    ):
        """Initialize the ML symbol model.

        Args:
            model_path: Path to a pre-trained model file or directory
            input_shape: Expected input shape (height, width, channels)
            num_classes: Number of symbol classes
            class_names: Optional list of class names
            use_transfer_learning: Whether to use transfer learning
            freeze_base: Whether to freeze the base model layers (for transfer learning)
        """
        super().__init__(model_path)
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.class_names = class_names or [f"class_{i}" for i in range(num_classes)]
        self.use_transfer_learning = use_transfer_learning
        self.freeze_base = freeze_base
        self.base_model = None

        if self.model is None:
            self.model = self.build()

    def build(self) -> tf.keras.Model:
        """Build the model architecture.

        Returns:
            A compiled Keras model.
        """
        self.logger.info("Building ML symbol model")

        inputs = layers.Input(shape=self.input_shape, name="input_layer")

        if self.use_transfer_learning:
            # Use EfficientNetB0 as base model
            self.base_model = applications.EfficientNetB0(
                include_top=False,
                weights="imagenet",
                input_tensor=inputs,
                input_shape=self.input_shape,
                pooling="avg",
            )

            # Freeze base model layers if needed
            if self.freeze_base:
                self.base_model.trainable = False

            x = self.base_model.output
            x = layers.Dense(512, activation="relu")(x)
            x = layers.Dropout(0.5)(x)
        else:
            # Build custom CNN
            x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(inputs)
            x = layers.BatchNormalization()(x)
            x = layers.MaxPooling2D((2, 2))(x)

            x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
            x = layers.BatchNormalization()(x)
            x = layers.MaxPooling2D((2, 2))(x)

            x = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)
            x = layers.BatchNormalization()(x)
            x = layers.MaxPooling2D((2, 2))(x)

            x = layers.Flatten()(x)
            x = layers.Dense(256, activation="relu")(x)
            x = layers.Dropout(0.5)(x)

        # Output layer
        outputs = layers.Dense(self.num_classes, activation="softmax")(x)

        # Create model
        model = models.Model(inputs=inputs, outputs=outputs, name="ml_symbol_model")

        # Compile model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
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
        # Convert to RGB if needed
        if len(image.shape) == 2:  # Grayscale
            image = np.stack([image] * 3, axis=-1)
        elif image.shape[2] == 1:  # Single channel
            image = np.concatenate([image] * 3, axis=-1)
        elif image.shape[2] == 4:  # RGBA
            image = image[..., :3]  # Remove alpha channel

        # Resize to model input shape
        image = tf.image.resize(
            image, (self.input_shape[0], self.input_shape[1])
        ).numpy()

        # Normalize pixel values
        if image.dtype == np.uint8:
            image = image.astype("float32") / 255.0

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
                {label: float(prob) for label, prob in zip(self.class_names, pred)}
            )

        return results

    def train(
        self,
        train_data: tf.data.Dataset,
        val_data: Optional[tf.data.Dataset] = None,
        epochs: int = 10,
        batch_size: int = 32,
        callbacks: Optional[List[tf.keras.callbacks.Callback]] = None,
    ) -> tf.keras.callbacks.History:
        """Train the model.

        Args:
            train_data: Training dataset
            val_data: Optional validation dataset
            epochs: Number of training epochs
            batch_size: Batch size
            callbacks: List of Keras callbacks

        Returns:
            Training history
        """
        if self.model is None:
            raise ValueError("Model has not been built")

        callbacks = callbacks or []

        # Add default callbacks if not provided
        if not any(
            isinstance(cb, tf.keras.callbacks.ModelCheckpoint) for cb in callbacks
        ):
            callbacks.append(
                tf.keras.callbacks.ModelCheckpoint(
                    filepath="best_model.keras",
                    save_best_only=True,
                    monitor="val_loss",
                    mode="min",
                    verbose=1,
                )
            )

        if not any(
            isinstance(cb, tf.keras.callbacks.EarlyStopping) for cb in callbacks
        ):
            callbacks.append(
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss", patience=5, restore_best_weights=True
                )
            )

        # Train the model
        history = self.model.fit(
            train_data,
            validation_data=val_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1,
        )

        return history
