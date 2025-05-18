"""Page processing package.

This package handles high-level image processing operations including:
- Complete page processing pipeline coordination
- Multi-stage image processing (normalization, binarization, etc.)
- Result aggregation and validation
- (Future) Staff line detection and removal
- (Future) Symbol segmentation

Typical usage:
    processor = PageProcessor(config)
    result = processor.process_page(image)
    normalized = result['normalized']
    binary = result['binary']  # Future
    no_staff = result['no_staff']  # Future
"""

from .page_processor import PageProcessor

__all__ = ["PageProcessor"]

# Configure logger for the processing package
from ..utils import get_logger

logger = get_logger("preprocessing.processing")
