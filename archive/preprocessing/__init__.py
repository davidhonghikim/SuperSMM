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

import logging
from ..utils.logger import setup_logger

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

# Configure logger using centralized configuration
logger = setup_logger(
    name="preprocessing",
    log_type="app",
    log_level=logging.INFO,
    log_to_console=True,
    log_to_file=True
)
