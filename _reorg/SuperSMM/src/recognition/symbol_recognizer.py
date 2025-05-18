"""
Symbol Recognizer Facade Module

Provides a simplified interface to the symbol recognition system.
"""

import numpy as np
from typing import Dict, Any, List, Optional
import logging
from pathlib import Path

from .config.recognizer_config import SymbolRecognizerConfig
from .models.symbol_recognizer_core import SymbolRecognizerCore


class SymbolRecognizer:
    """Facade for the symbol recognition system."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the symbol recognizer.

        Args:
            config: Optional configuration dictionary
        """
        self.logger = logging.getLogger(__name__)
        self.core = SymbolRecognizerCore(config)

    def recognize_symbols(self, symbols: List[np.ndarray]) -> List[Dict[str, Any]]:
        """Recognize musical symbols from input images.

        Args:
            symbols: List of symbol images to recognize

        Returns:
            List of recognition results with predictions and confidences
        """
        if not symbols:
            self.logger.warning("No symbols provided for recognition")
            return []

        results = []
        for i, symbol in enumerate(symbols):
            try:
                # Get predictions from core
                predictions = self.core.predict(symbol)

                # Format result
                if predictions:
                    top_prediction = predictions[0]
                    result = {
                        "label": top_prediction["label"],
                        "confidence": top_prediction["confidence"],
                        "class_index": top_prediction["class_index"],
                        "alternatives": predictions[1:],
                        "raw_image": symbol,
                    }
                else:
                    result = {
                        "label": "unknown",
                        "confidence": 0.0,
                        "raw_image": symbol,
                    }

                results.append(result)

            except Exception as e:
                self.logger.error(f"Error recognizing symbol {i}: {e}")
                results.append({"error": str(e), "raw_image": symbol})

        return results

    def recognize_symbols_batch(
        self, symbols: List[np.ndarray]
    ) -> List[Dict[str, Any]]:
        """Recognize musical symbols in batch mode.

        Args:
            symbols: List of symbol images to recognize

        Returns:
            List of recognition results with predictions and confidences
        """
        if not symbols:
            self.logger.warning("No symbols provided for batch recognition")
            return []

        # Get batch predictions from core
        batch_predictions = self.core.predict_batch(symbols)

        # Format results
        results = []
        for i, predictions in enumerate(batch_predictions):
            if predictions:
                top_prediction = predictions[0]
                result = {
                    "symbol_index": i,
                    "label": top_prediction["label"],
                    "confidence": top_prediction["confidence"],
                    "class_index": top_prediction["class_index"],
                    "alternatives": predictions[1:],
                }
            else:
                result = {"symbol_index": i, "label": "unknown", "confidence": 0.0}

            results.append(result)

        return results
