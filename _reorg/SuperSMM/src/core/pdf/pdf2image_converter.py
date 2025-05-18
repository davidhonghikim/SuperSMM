"""
PDF2Image Converter Module

Implements PDF conversion using pdf2image library.
"""

# Standard library imports
import logging
from pathlib import Path
from typing import List, Dict, Any

# Third-party imports
import numpy as np
import cv2
from pdf2image import convert_from_path
from pdf2image.exceptions import PDFPageCountError

# Local imports
from .base_converter import BasePDFConverter


class PDF2ImageConverter(BasePDFConverter):
    """PDF converter using pdf2image library.

    Uses poppler backend through pdf2image to convert PDFs to images.
    Handles multi-page PDFs and provides detailed metadata.

    Attributes:
        thread_count (int): Number of threads for conversion
        use_cropbox (bool): Use cropbox instead of mediabox
        strict (bool): Raise error on corrupted PDFs
    """

    def __init__(
        self,
        dpi: int = 300,
        fmt: str = "PNG",
        thread_count: int = 4,
        use_cropbox: bool = True,
        strict: bool = False,
    ):
        """Initialize the converter.

        Args:
            dpi (int): Resolution for conversion
            fmt (str): Output format
            thread_count (int): Number of threads
            use_cropbox (bool): Use cropbox instead of mediabox
            strict (bool): Raise error on corrupted PDFs
        """
        super().__init__(dpi, fmt)
        self.thread_count = thread_count
        self.use_cropbox = use_cropbox
        self.strict = strict

    def _convert_pages(self, pdf_path: Path) -> List[Dict[str, Any]]:
        """Convert PDF pages using pdf2image.

        Args:
            pdf_path (Path): Path to PDF file

        Returns:
            List[Dict[str, Any]]: Conversion results
        """
        try:
            # Convert pages
            pil_images = convert_from_path(
                pdf_path,
                dpi=self.dpi,
                fmt=self.fmt,
                thread_count=self.thread_count,
                use_cropbox=self.use_cropbox,
                strict=self.strict,
            )

            # Process results
            results = []
            for i, pil_image in enumerate(pil_images, 1):
                # Convert to numpy array
                image = np.array(pil_image)
                print(
                    f"[PDF2ImageConverter] Page {i}: type={type(image)}, dtype={image.dtype}, shape={image.shape}"
                )
                if image.dtype != np.uint8:
                    print(
                        f"[PDF2ImageConverter] Casting page {i} from {image.dtype} to uint8"
                    )
                    image = image.astype(np.uint8)
                # Convert RGB to BGR for OpenCV compatibility
                if len(image.shape) == 3 and image.shape[2] == 3:
                    print(f"[PDF2ImageConverter] Converting page {i} from RGB to BGR")
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                results.append(
                    {
                        "image": image,
                        "page_number": i,
                        "size": {"width": pil_image.width, "height": pil_image.height},
                        "dpi": self.dpi,
                    }
                )

            return results

        except PDFPageCountError as e:
            self.logger.error("Failed to get page count: %s", e)
            return [
                {
                    "image": None,
                    "page_number": 0,
                    "size": {"width": 0, "height": 0},
                    "dpi": 0,
                    "error": str(e),
                }
            ]
        except Exception as e:
            self.logger.error("Page conversion failed: %s", e)
            return [
                {
                    "image": None,
                    "page_number": 0,
                    "size": {"width": 0, "height": 0},
                    "dpi": 0,
                    "error": str(e),
                }
            ]
