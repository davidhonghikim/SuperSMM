"""
PDF processing utilities for OMR.

Provides a high-level interface for PDF processing operations
including page extraction and image conversion.
"""

# Standard library imports
import logging
from typing import List, Dict, Any, Optional, Type

# Third-party imports
import numpy as np

# Local imports
from ..pdf.base_converter import BasePDFConverter
from ..pdf.pdf2image_converter import PDF2ImageConverter


class PDFProcessor:
    """High-level PDF processing interface.

    Handles PDF operations using configurable converter backends.
    Provides error handling and logging.

    Attributes:
        logger (logging.Logger): Logger instance
        converter (BasePDFConverter): PDF converter instance
    """

    def __init__(
        self,
        converter_class: Type[BasePDFConverter] = PDF2ImageConverter,
        **converter_kwargs,
    ):
        """Initialize the processor.

        Args:
            converter_class (Type[BasePDFConverter]): Converter class to use
            **converter_kwargs: Arguments for converter initialization
        """
        self.logger = logging.getLogger(__name__)
        self.converter = converter_class(**converter_kwargs)

    def extract_pages(self, pdf_path: str) -> List[np.ndarray]:
        """Extract pages from PDF as images.

        Args:
            pdf_path (str): Path to PDF file

        Returns:
            List[np.ndarray]: List of page images

        Raises:
            FileNotFoundError: If PDF file doesn't exist
            ValueError: If PDF processing fails
        """
        # Convert PDF pages
        results = self.converter.convert_pdf(pdf_path)

        # Check for errors
        if results and "error" in results[0]:
            raise ValueError(results[0]["error"])

        # Extract images
        return [r["image"] for r in results if r["image"] is not None]

    def get_page_metadata(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Get metadata for PDF pages.

        Args:
            pdf_path (str): Path to PDF file

        Returns:
            List[Dict[str, Any]]: Metadata for each page
        """
        try:
            results = self.converter.convert_pdf(pdf_path)
            return [
                {
                    "page_number": r["page_number"],
                    "size": r["size"],
                    "dpi": r["dpi"],
                    "error": r.get("error"),
                }
                for r in results
            ]

        except Exception as e:
            self.logger.error("Failed to get page metadata: %s", e)
            return [
                {
                    "page_number": 0,
                    "size": {"width": 0, "height": 0},
                    "dpi": 0,
                    "error": str(e),
                }
            ]
