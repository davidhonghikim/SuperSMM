"""
Staff Remover Module

Handles the detection and removal of staff lines from sheet music images.
"""

# Standard library imports
import logging
from typing import Dict, Any, List, Tuple

# Third-party imports
import cv2
import numpy as np


class StaffRemover:
    """Removes staff lines from sheet music images.

    Uses image processing techniques to detect and remove staff lines
    while preserving musical symbols.

    Attributes:
        logger (logging.Logger): Logger instance
        min_line_length (int): Minimum length for staff lines
        line_thickness (int): Expected staff line thickness
    """

    def __init__(self):
        """Initialize the staff remover."""
        self.logger = logging.getLogger(__name__)
        self.min_line_length = 100
        self.line_thickness = 1

    def _detect_lines(self, binary: np.ndarray) -> List[Dict[str, Any]]:
        """Detect horizontal lines using morphological operations.

        Args:
            binary (np.ndarray): Binary image

        Returns:
            List[Dict[str, Any]]: Detected line information
        """
        try:
            # Create horizontal kernel
            kernel = cv2.getStructuringElement(
                cv2.MORPH_RECT, (self.min_line_length, self.line_thickness)
            )

            # Detect horizontal lines
            detected = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

            # Find contours of lines
            contours, _ = cv2.findContours(
                detected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            # Extract line information
            lines = []
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                if w >= self.min_line_length:
                    lines.append({"x": x, "y": y, "width": w, "height": h})

            return lines

        except Exception as e:
            self.logger.error("Line detection failed: %s", e)
            return []

    def _remove_lines(
        self, image: np.ndarray, lines: List[Dict[str, Any]]
    ) -> np.ndarray:
        """Remove detected staff lines from image.

        Args:
            image (np.ndarray): Input image
            lines (List[Dict[str, Any]]): Detected line information

        Returns:
            np.ndarray: Image with staff lines removed
        """
        try:
            print(
                f"[_remove_lines] image type: {type(image)}, shape: {getattr(image, 'shape', None)}, dtype: {getattr(image, 'dtype', None)}"
            )
            print(f"[_remove_lines] lines: {lines}")
            # Create mask of staff lines
            mask = np.zeros_like(image)
            for line in lines:
                y, h = line["y"], line["height"]
                print(f"[_remove_lines] line: y={y}, h={h}")
                mask[y : y + h, :] = 255

            # Remove lines using mask
            result = cv2.subtract(image, mask)
            print(
                f"[_remove_lines] result type: {type(result)}, shape: {getattr(result, 'shape', None)}, dtype: {getattr(result, 'dtype', None)}"
            )
            return result

        except Exception as e:
            print(f"[_remove_lines] Exception: {e}")
            self.logger.error("Line removal failed: %s", e)
            return image

    def remove_staff(self, binary: np.ndarray) -> Dict[str, Any]:
        """Remove staff lines from binary image.

        Args:
            binary (np.ndarray): Binary input image

        Returns:
            Dict[str, Any]: Result with processed image and metadata
        """
        try:
            print(
                f"[remove_staff] binary type: {type(binary)}, shape: {getattr(binary, 'shape', None)}, dtype: {getattr(binary, 'dtype', None)}"
            )
            # Detect staff lines
            lines = self._detect_lines(binary)
            print(f"[remove_staff] detected lines: {lines}")

            if not lines:
                return {
                    "image": binary,
                    "staff_lines": [],
                    "error": "No staff lines detected",
                }

            # Remove detected lines
            result = self._remove_lines(binary, lines)
            print(
                f"[remove_staff] result type: {type(result)}, shape: {getattr(result, 'shape', None)}, dtype: {getattr(result, 'dtype', None)}"
            )

            return {"image": result, "staff_lines": lines, "line_count": len(lines)}

        except Exception as e:
            print(f"[remove_staff] Exception: {e}")
            self.logger.error("Staff removal failed: %s", e)
            return {"image": binary, "staff_lines": [], "error": str(e)}
