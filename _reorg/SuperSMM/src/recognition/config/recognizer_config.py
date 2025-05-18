"""
Symbol Recognizer Configuration Module

Defines configuration dataclasses for the symbol recognition pipeline.
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass
class SymbolRecognizerConfig:
    """Configuration for the SymbolRecognizer.

    Attributes:
        input_shape: Expected input shape for the model (height, width, channels)
        num_classes: Number of symbol classes to recognize
        top_k_predictions: Number of top predictions to return
        confidence_threshold: Minimum confidence threshold for valid predictions
        batch_size: Batch size for batch processing
        model_path: Path to the trained model file
        vocab_path: Path to the vocabulary file with class labels
    """

    input_shape: Tuple[int, int, int] = (64, 64, 1)
    num_classes: int = 10
    top_k_predictions: int = 5
    confidence_threshold: float = 0.5
    batch_size: int = 32
    model_path: str = (
        "/ml/models/resources/tf-deep-omr/resources/ml_models/symbol_recognition.h5"
    )
    vocab_path: str = (
        "/ml/models/resources/tf-deep-omr/resources/ml_models/vocabulary_semantic.txt"
    )
