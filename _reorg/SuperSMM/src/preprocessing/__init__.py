"""Preprocessing package for image processing and enhancement.

This package provides a complete pipeline for preprocessing sheet music images,
including size normalization, color space conversion, and staff line detection.

Key components:
- Configuration management (PreprocessorConfig)
- Image normalization (ImageNormalizer)
- Page processing pipeline (PageProcessor)
- Input validation utilities

Typical usage:
    from preprocessing import PreprocessorConfig, PageProcessor

    config = PreprocessorConfig(
        normalize_min_size=500,
        normalize_max_size=800
    )
    processor = PageProcessor(config)
    result = processor.process_page(image)
"""

from .config import PreprocessorConfig
from .normalization import ImageNormalizer
from .processing import PageProcessor
from .validators import validate_image, validate_size_range, ensure_odd

__all__ = [
    "PreprocessorConfig",
    "ImageNormalizer",
    "PageProcessor",
    "validate_image",
    "validate_size_range",
    "ensure_odd",
]

# Configure root logger
from .utils import get_logger

logger = get_logger("preprocessing")
