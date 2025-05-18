"""Core functionality for the SuperSMM OMR system.

This package contains the core components of the Optical Music Recognition system,
including the main pipeline, configuration management, and common utilities.
"""

from .pipeline.omr_pipeline import OMRPipeline
from .exceptions import (
    OMRException,
    ConfigurationError,
    PreprocessingError,
    SegmentationError,
    RecognitionError,
    ExportError,
    create_error_handler,
)

__all__ = [
    "OMRPipeline",
    "OMRException",
    "ConfigurationError",
    "PreprocessingError",
    "SegmentationError",
    "RecognitionError",
    "ExportError",
    "create_error_handler",
]
