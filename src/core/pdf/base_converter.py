"""
Base PDF Converter Module

Provides the base class for PDF conversion operations.
"""

# Standard library imports
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod

# Third-party imports
import numpy as np


class BasePDFConverter(ABC):
    """Base class for PDF conversion operations.

    Provides common functionality and interface for converting
    PDFs to images using different backends.

    Attributes:
        logger (logging.Logger): Logger instance
        dpi (int): Resolution for image conversion
        fmt (str): Output image format
    """

    def __init__(self, dpi: int = 300, fmt: str = "PNG"):
        """Initialize the converter.

        Args:
            dpi (int): Resolution for conversion, default 300
            fmt (str): Output format, default 'PNG'
        """
        self.logger = logging.getLogger(__name__)
        self.dpi = dpi
        self.fmt = fmt.upper()

    def convert_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Convert PDF to list of images.

        Args:
            pdf_path (str): Path to PDF file

        Returns:
            List[Dict[str, Any]]: List of conversion results
                Each dict contains:
                - image: np.ndarray of the page
                - page_number: int
                - size: Dict with width and height
                - dpi: int resolution used
                - error: Optional error message
        """
        try:
            # Validate input
            path = Path(pdf_path)
            if not path.exists():
                raise FileNotFoundError(f"PDF not found: {pdf_path}")
            if not path.suffix.lower() == ".pdf":
                raise ValueError(f"Not a PDF file: {pdf_path}")

            # Convert pages
            return self._convert_pages(path)

        except Exception as e:
            self.logger.error("PDF conversion failed: %s", e)
            return [
                {
                    "image": None,
                    "page_number": 0,
                    "size": {"width": 0, "height": 0},
                    "dpi": self.dpi,
                    "error": str(e),
                }
            ]

    @abstractmethod
    def _convert_pages(self, pdf_path: Path) -> List[Dict[str, Any]]:
        """Convert PDF pages to images.

        Args:
            pdf_path (Path): Path to PDF file

        Returns:
            List[Dict[str, Any]]: Conversion results

        This method must be implemented by subclasses.
        """
        pass
