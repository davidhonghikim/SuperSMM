"""
Symbol Segmenter Module

Handles the segmentation of musical symbols from staff-removed images.
"""

# Standard library imports
import logging
from typing import List, Dict, Any, Tuple

# Third-party imports
import cv2
import numpy as np


class SymbolSegmenter:
    """Segments musical symbols from preprocessed images.

    Uses connected components analysis and contour detection to
    identify and extract individual musical symbols.

    Attributes:
        logger (logging.Logger): Logger instance
        min_symbol_size (int): Minimum size for valid symbols
        max_symbol_size (int): Maximum size for valid symbols
    """

    def __init__(self):
        """Initialize the segmenter."""
        self.logger = logging.getLogger(__name__)
        self.min_symbol_size = 20  # Minimum symbol dimension
        self.max_symbol_size = 200  # Maximum symbol dimension

    def _get_bounding_boxes(self, binary: np.ndarray) -> List[Dict[str, Any]]:
        """Get bounding boxes for connected components.

        Args:
            binary (np.ndarray): Binary image

        Returns:
            List[Dict[str, Any]]: List of bounding box information
        """
        try:
            # Find connected components
            _, _, stats, _ = cv2.connectedComponentsWithStats(binary)

            # Extract bounding boxes (skip background at index 0)
            boxes = []
            for i in range(1, len(stats)):
                x, y, w, h, area = stats[i]

                # Filter by size
                if (
                    self.min_symbol_size <= w <= self.max_symbol_size
                    and self.min_symbol_size <= h <= self.max_symbol_size
                ):
                    boxes.append(
                        {"x": x, "y": y, "width": w, "height": h, "area": area}
                    )

            return boxes

        except Exception as e:
            self.logger.error("Bounding box detection failed: %s", e)
            return []

    def _extract_symbol(self, image: np.ndarray, box: Dict[str, Any]) -> np.ndarray:
        """Extract a symbol using its bounding box.

        Args:
            image (np.ndarray): Source image
            box (Dict[str, Any]): Bounding box information

        Returns:
            np.ndarray: Extracted symbol image
        """
        try:
            x, y = box["x"], box["y"]
            w, h = box["width"], box["height"]
            return image[y : y + h, x : x + w]
        except Exception as e:
            self.logger.error("Symbol extraction failed: %s", e)
            return np.array([])

    def segment(self, binary: np.ndarray) -> List[Dict[str, Any]]:
        """Segment symbols from a binary image.

        Args:
            binary (np.ndarray): Binary image with staff lines removed

        Returns:
            List[Dict[str, Any]]: List of extracted symbols with metadata
        """
        try:
            # Get bounding boxes
            boxes = self._get_bounding_boxes(binary)

            # Extract symbols
            symbols = []
            for box in boxes:
                symbol_img = self._extract_symbol(binary, box)
                if symbol_img.size > 0:
                    symbols.append(
                        {
                            "image": symbol_img,
                            "position": {"x": box["x"], "y": box["y"]},
                            "size": {"width": box["width"], "height": box["height"]},
                            "area": box["area"],
                        }
                    )

            return symbols

        except Exception as e:
            self.logger.error("Symbol segmentation failed: %s", e)
            return []
