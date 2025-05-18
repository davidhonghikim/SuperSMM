"""
SuperSMM - Advanced Sheet Music Recognition

A comprehensive Optical Music Recognition (OMR) system for converting sheet music
into machine-readable formats.

This package provides the core functionality for processing, analyzing, and recognizing
sheet music, including preprocessing, segmentation, and symbol recognition.
"""

# Package version
__version__ = "0.1.0"

# Import key components to make them available at the package level
from .core.symbols import SymbolType
from .recognition import (
    SymbolRecognitionResult,
    SymbolRecognizer,
    recognize_symbols_in_image,
)
from .segmentation import (
    BoundingBox,
    Staff,
    StaffLine,
    Symbol,
    SymbolSegmenter,
    StaffDetector,
    detect_staff_lines,
    segment_symbols,
    filter_symbols_by_size,
    group_symbols_into_notes,
)
from .preprocessing import (
    normalize_image,
    binarize_image,
    deskew_image,
    remove_staff_lines,
    detect_staff_lines,
    random_rotation,
    random_zoom,
    random_brightness,
    add_noise,
)
from .utils.logging_config import setup_logging

# Define what gets imported with 'from supersmm import *'
__all__ = [
    # Recognition
    'SymbolType',
    'SymbolRecognitionResult',
    'SymbolRecognizer',
    'recognize_symbols_in_image',
    
    # Segmentation
    'BoundingBox',
    'Staff',
    'StaffLine',
    'Symbol',
    'SymbolSegmenter',
    'StaffDetector',
    'detect_staff_lines',
    'segment_symbols',
    'filter_symbols_by_size',
    'group_symbols_into_notes',
    
    # Preprocessing
    'normalize_image',
    'binarize_image',
    'deskew_image',
    'remove_staff_lines',
    'detect_staff_lines',
    'random_rotation',
    'random_zoom',
    'random_brightness',
    'add_noise',
    
    # Utils
    'setup_logging',
]

# Set up default logging configuration when the package is imported
import logging
logging.getLogger(__name__).addHandler(logging.NullHandler())