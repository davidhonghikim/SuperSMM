"""
Recognition Pipeline Orchestrator Module

Coordinates the complete OMR recognition pipeline including symbol recognition and HMM decoding.
"""

import numpy as np
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

from ..symbol_recognizer import SymbolRecognizer
from ..hmm_decoder import SymbolHMMDecoder


class RecognitionPipeline:
    """Orchestrates the complete OMR recognition pipeline."""

    def __init__(
        self, model_path: Optional[str] = None, vocab_path: Optional[str] = None
    ):
        """Initialize the recognition pipeline.

        Args:
            model_path: Optional path to the symbol recognition model
            vocab_path: Optional path to the vocabulary file
        """
        self.logger = logging.getLogger(__name__)

        # Configuration
        config = {}
        if model_path:
            config["model_path"] = model_path
        if vocab_path:
            config["vocab_path"] = vocab_path

        # Initialize components
        self.recognizer = SymbolRecognizer(config)

        # HMM decoder will be initialized when needed with class labels
        self.hmm_decoder = None

    def process_page(self, page_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single page of sheet music.

        Args:
            page_data: Page data with symbols and metadata

        Returns:
            Processed page data with recognition results
        """
        # Extract symbols
        symbols = []
        for symbol in page_data.get("symbols", []):
            if "image" in symbol:
                symbols.append(symbol["image"])

        if not symbols:
            self.logger.warning("No symbols found in page data")
            return page_data

        # Recognize symbols
        recognition_results = self.recognizer.recognize_symbols(symbols)

        # Update symbol data with recognition results
        for i, result in enumerate(recognition_results):
            if i < len(page_data["symbols"]):
                page_data["symbols"][i].update(
                    {
                        "label": result.get("label"),
                        "confidence": result.get("confidence"),
                        "class_index": result.get("class_index"),
                        "alternatives": result.get("alternatives", []),
                    }
                )

        # Apply HMM decoding if enough symbols are recognized
        if len(symbols) > 1:
            self._apply_hmm_decoding(page_data)

        return page_data

    def _apply_hmm_decoding(self, page_data: Dict[str, Any]) -> None:
        """Apply HMM decoding to refine symbol recognition.

        Args:
            page_data: Page data with recognized symbols
        """
        # Extract probability matrix for HMM
        symbols = page_data.get("symbols", [])
        if not symbols or "class_index" not in symbols[0]:
            return

        # Initialize HMM decoder if needed
        if self.hmm_decoder is None:
            # Get class labels from first symbol's alternatives
            if "alternatives" in symbols[0] and symbols[0]["alternatives"]:
                class_labels = [symbols[0]["label"]] + [
                    alt["label"] for alt in symbols[0]["alternatives"]
                ]
                self.hmm_decoder = SymbolHMMDecoder(class_labels, logger=self.logger)
            else:
                self.logger.warning(
                    "Cannot initialize HMM decoder: no class labels available"
                )
                return

        # Create probability matrix for HMM
        n_classes = len(self.hmm_decoder.core.class_labels)
        prob_matrix = np.zeros((len(symbols), n_classes))

        # Fill probability matrix
        for i, symbol in enumerate(symbols):
            if "class_index" in symbol and symbol["class_index"] is not None:
                prob_matrix[i, symbol["class_index"]] = symbol["confidence"]

            # Add alternatives
            for alt in symbol.get("alternatives", []):
                if "class_index" in alt and alt["class_index"] is not None:
                    prob_matrix[i, alt["class_index"]] = alt["confidence"]

        # Apply HMM decoding
        try:
            decoded_symbols = self.hmm_decoder.decode(prob_matrix)

            # Update symbols with HMM decoding results
            for i, decoded in enumerate(decoded_symbols):
                if i < len(symbols):
                    symbols[i]["hmm_label"] = decoded["label"]
                    symbols[i]["hmm_confidence"] = decoded["confidence"]
                    symbols[i]["hmm_logprob"] = decoded["hmm_logprob"]
        except Exception as e:
            self.logger.error(f"HMM decoding failed: {e}")

    def process_sheet_music(
        self, sheet_data: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Process all pages in a sheet music document.

        Args:
            sheet_data: List of page data with symbols and metadata

        Returns:
            Processed sheet data with recognition results
        """
        processed_pages = []
        for page_data in sheet_data:
            processed_page = self.process_page(page_data)
            processed_pages.append(processed_page)

        return processed_pages
