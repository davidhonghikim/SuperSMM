"""
Main OMR processing orchestrator.

This module coordinates the OMR processing pipeline, including:
- PDF extraction
- Image preprocessing
- Staff line detection
"""

# Standard library imports
import logging
from typing import List, Dict, Any
from pathlib import Path

# Third-party imports
import numpy as np

# Local imports
from .pdf_processor import PDFProcessor
from .image_preprocessor import ImagePreprocessor
from .staff_remover import StaffRemover
from .symbol_segmenter import SymbolSegmenter
from ..models.symbol_classifier import SymbolClassifier
from ..pdf.pdf2image_converter import PDF2ImageConverter


class LocalOMRProcessor:
    """Local OMR processing pipeline.

    Coordinates the complete OMR workflow:
    1. PDF extraction
    2. Image preprocessing
    3. Staff line removal
    4. Symbol segmentation
    5. Symbol classification

    Attributes:
        logger (logging.Logger): Logger instance
        preprocessor (ImagePreprocessor): Image preprocessing component
        staff_remover (StaffRemover): Staff line removal component
        segmenter (SymbolSegmenter): Symbol segmentation component
        classifier (SymbolClassifier): Symbol classification component
    """

    """Main orchestrator for local OMR processing."""

    def __init__(self, model_path: str = None):
        """Initialize the OMR processor.

        Args:
            model_path (str, optional): Path to symbol classifier model
        """
        self.logger = logging.getLogger(__name__)

        # Initialize components
        self.pdf_processor = PDFProcessor(converter_class=PDF2ImageConverter)
        self.preprocessor = ImagePreprocessor()
        self.staff_remover = StaffRemover()
        self.segmenter = SymbolSegmenter()
        self.classifier = SymbolClassifier(model_path) if model_path else None

    def process_sheet_music(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Process a sheet music PDF.

        Args:
            pdf_path (str): Path to PDF file

        Returns:
            List[Dict[str, Any]]: Results for each page
        """
        try:
            # Extract PDF pages (get full dicts, not just images)
            pages = self.pdf_processor.converter.convert_pdf(pdf_path)
            if not pages:
                raise ValueError("No pages extracted from PDF")

            # Check for extraction errors
            if any("error" in page for page in pages):
                errors = [page["error"] for page in pages if "error" in page]
                raise ValueError(f"PDF extraction failed: {'; '.join(errors)}")

            # Process each page
            results = []
            for page in pages:
                self.logger.info("Processing page %d", page["page_number"])
                self.logger.debug(f"Page dict keys: {list(page.keys())}")
                self.logger.debug(f"Type of page['image']: {type(page['image'])}")
                image = page["image"]
                if image is None or not hasattr(image, "shape"):
                    self.logger.warning(
                        f"Skipping page {page['page_number']}: Invalid image (None or missing shape). Value: {image}"
                    )
                    results.append(
                        {
                            "error": "Invalid image for page",
                            "page_number": page["page_number"],
                            "page_size": page["size"],
                            "page_dpi": page["dpi"],
                        }
                    )
                    continue
                result = self._process_page(image)
                self.logger.debug(f"Result keys: {list(result.keys())}")
                result["page_number"] = page["page_number"]
                result["page_size"] = page["size"]
                result["page_dpi"] = page["dpi"]
                results.append(result)

            return results

        except Exception as e:
            self.logger.error("Sheet music processing failed: %s", e)
            return [{"error": str(e), "page_number": 0}]

    def _process_page(self, image: np.ndarray) -> Dict[str, Any]:
        """Process a single page image.

        Args:
            image (np.ndarray): Page image

        Returns:
            Dict[str, Any]: Processing results
        """
        try:
            self.logger.debug(
                f"_process_page: Received image type: {type(image)}; shape: {getattr(image, 'shape', None)}"
            )
            # Preprocess image
            prep_result = self.preprocessor.preprocess(image)
            self.logger.debug(
                f"preprocess() returned type: {type(prep_result)}; keys: {list(prep_result.keys()) if isinstance(prep_result, dict) else 'N/A'}; value: {prep_result}"
            )
            if "error" in prep_result:
                return {"error": prep_result["error"]}

            # Remove staff lines
            staff_result = self.staff_remover.remove_staff(prep_result["binary"])
            self.logger.debug(
                f"remove_staff() returned type: {type(staff_result)}; keys: {list(staff_result.keys()) if isinstance(staff_result, dict) else 'N/A'}; value: {staff_result}"
            )
            if "error" in staff_result:
                return {"error": staff_result["error"]}

            # Segment symbols
            symbols = self.segmenter.segment(staff_result["image"])
            self.logger.debug(
                f"segment() returned type: {type(symbols)}; length: {len(symbols) if hasattr(symbols, '__len__') else 'N/A'}"
            )
            if not symbols:
                return {"error": "No symbols detected"}

            # Classify symbols if classifier is available
            if self.classifier:
                symbol_images = [s["image"] for s in symbols]
                classifications = self.classifier.predict(symbol_images)

                # Merge classification results with symbol data
                for symbol, classification in zip(symbols, classifications):
                    symbol.update(classification)

            # Propagate all intermediate images for saving
            preprocessing_dict = {
                "original": image,
                "grayscale": prep_result.get("grayscale"),
                "denoised": prep_result.get("denoised"),
                "binary": prep_result.get("binary"),
                "staff_removed": staff_result.get("image"),
                "threshold_params": prep_result.get("threshold_params", {}),
            }
            return {
                "symbols": symbols,
                "staff_lines": staff_result["staff_lines"],
                "preprocessing": preprocessing_dict,
            }

        except Exception as e:
            self.logger.error("Page processing failed: %s", e)
            return {"error": str(e)}
