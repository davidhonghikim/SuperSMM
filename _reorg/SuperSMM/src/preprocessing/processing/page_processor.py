"""Page processing functionality."""

import time
import numpy as np
from typing import Dict
from ..config import PreprocessorConfig
from ..normalization.image_normalizer import ImageNormalizer
from ..utils.logging import get_logger


class PageProcessor:
    """Handles page-level image processing operations."""

    def __init__(self, config: PreprocessorConfig):
        """Initialize the processor.

        Args:
            config: Configuration parameters
        """
        self.config = config
        self.normalizer = ImageNormalizer(config)
        self.logger = get_logger(__name__)
        self.perf_logger = get_logger("performance")

    def process_page(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """Process a single page image through the complete preprocessing pipeline.

        Args:
            image: Input page image

        Returns:
            Dict containing processed images at various stages:
            - 'normalized': Size and color normalized image
            - 'binary': Binarized version of the image
            - 'no_staff': Image with staff lines removed

        Raises:
            ValueError: If image processing fails at any stage
        """
        start_time = time.time()
        result = {}

        try:
            # Normalize image
            normalized = self.normalizer.normalize_image(image)
            result["normalized"] = normalized

            # TODO: Add binarization and staff line removal
            # These will be implemented in separate modules

            return result

        except Exception as e:
            self.logger.error(f"Error during page processing: {str(e)}")
            raise
        finally:
            duration = time.time() - start_time
            self.perf_logger.debug("process_page completed in %.3fs", duration)
