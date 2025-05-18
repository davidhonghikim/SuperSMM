"""Image normalization functionality."""

import time
import cv2
import numpy as np
from typing import Optional, Dict, Any, Union
from ..config import PreprocessorConfig
from ..utils.logging import get_logger


class ImageNormalizer:
    """Handles image normalization operations."""

    def __init__(self, config: PreprocessorConfig):
        """Initialize the normalizer.

        Args:
            config: Configuration parameters
        """
        self.config = config
        self.logger = get_logger(__name__)
        self.perf_logger = get_logger("performance")

    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        """Normalize image size and color.

        Args:
            image: Input image

        Returns:
            Normalized image

        Raises:
            ValueError: If image is invalid
        """
        start_time = time.time()

        try:
            # Get image dimensions
            h, w = image.shape[:2]

            if h == 0 or w == 0:
                raise ValueError("Invalid image dimensions")

            # Calculate target size maintaining aspect ratio
            aspect_ratio = w / h
            self.logger.debug(
                f"Original dimensions: ({h}, {w}), aspect ratio: {aspect_ratio:.2f}"
            )

            # Calculate scales needed to meet min and max constraints
            scale_to_min_h = self.config.normalize_min_size / h
            scale_to_min_w = self.config.normalize_min_size / w
            scale_to_max_h = self.config.normalize_max_size / h
            scale_to_max_w = self.config.normalize_max_size / w

            self.logger.debug(
                f"Scale factors - min_h: {scale_to_min_h:.2f}, min_w: {scale_to_min_w:.2f}, "
                f"max_h: {scale_to_max_h:.2f}, max_w: {scale_to_max_w:.2f}"
            )

            # Use the larger of the min scales to ensure both dimensions are >= min_size
            scale = max(scale_to_min_h, scale_to_min_w)
            self.logger.debug(f"Initial scale (max of min scales): {scale:.2f}")

            # But if this would exceed max_size, use the smaller of the max scales
            if (
                h * scale > self.config.normalize_max_size
                or w * scale > self.config.normalize_max_size
            ):
                scale = min(scale_to_max_h, scale_to_max_w)
                self.logger.debug(f"Adjusted scale (min of max scales): {scale:.2f}")

            # Calculate new dimensions
            new_h = int(h * scale)
            new_w = int(w * scale)
            self.logger.debug(f"Final dimensions: ({new_h}, {new_w})")

            # Resize using computed dimensions
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

            # Convert to grayscale if needed
            if len(image.shape) > 2:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            return image

        except Exception as e:
            self.logger.error(f"Error during image normalization: {str(e)}")
            raise
        finally:
            # Log performance metrics
            duration = time.time() - start_time
            input_shape = getattr(image, "shape", None)
            output_shape = getattr(image, "shape", None)
            self.perf_logger.debug(
                "normalize_image completed in %.3fs. Input shape: %s, Output shape: %s",
                duration,
                input_shape[:2] if input_shape else "None",
                output_shape[:2] if output_shape else "None",
            )
