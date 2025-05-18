"""Image normalization package.

This package handles all image normalization operations including:
- Image size normalization (maintaining aspect ratio)
- Color space conversion (RGB to grayscale)
- (Future) Image enhancement and filtering
- (Future) Contrast normalization

Typical usage:
    normalizer = ImageNormalizer(config)
    normalized_image = normalizer.normalize_image(input_image)
"""

from .image_normalizer import ImageNormalizer

__all__ = ["ImageNormalizer"]

# Configure logger for the normalization package
from ..utils import get_logger

logger = get_logger("preprocessing.normalization")
