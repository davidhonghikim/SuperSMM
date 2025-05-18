"""Tests for the symbol segmentation module."""

import numpy as np
import pytest
import cv2
from unittest.mock import patch

from src.core.processors.symbol_segmenter import SymbolSegmenter


@pytest.fixture
def segmenter():
    """Create a symbol segmenter instance."""
    return SymbolSegmenter()


@pytest.fixture
def test_binary():
    """Create a test binary image with symbol-like components."""
    image = np.zeros((200, 300), dtype=np.uint8)
    
    # Add some "symbols"
    cv2.circle(image, (50, 50), 15, 255, -1)  # Filled circle
    cv2.rectangle(image, (100, 100), (130, 140), 255, -1)  # Filled rectangle
    
    return image


class TestSymbolSegmenter:
    """Test suite for SymbolSegmenter class."""
    
    def test_get_bounding_boxes_success(self, segmenter, test_binary):
        """Test successful bounding box detection."""
        boxes = segmenter._get_bounding_boxes(test_binary)
        
        assert len(boxes) > 0
        for box in boxes:
            assert 'x' in box
            assert 'y' in box
            assert 'width' in box
            assert 'height' in box
            assert 'area' in box
            
            # Check size constraints
            assert segmenter.min_symbol_size <= box['width'] <= segmenter.max_symbol_size
            assert segmenter.min_symbol_size <= box['height'] <= segmenter.max_symbol_size
            
    def test_get_bounding_boxes_no_symbols(self, segmenter):
        """Test bounding box detection with no symbols."""
        empty_image = np.zeros((100, 100), dtype=np.uint8)
        boxes = segmenter._get_bounding_boxes(empty_image)
        
        assert len(boxes) == 0
        
    def test_get_bounding_boxes_error(self, segmenter):
        """Test bounding box detection error handling."""
        invalid_image = np.array([])
        boxes = segmenter._get_bounding_boxes(invalid_image)
        
        assert len(boxes) == 0
        
    def test_extract_symbol_success(self, segmenter, test_binary):
        """Test successful symbol extraction."""
        box = {'x': 40, 'y': 40, 'width': 30, 'height': 30}
        symbol = segmenter._extract_symbol(test_binary, box)
        
        assert symbol.shape == (30, 30)
        assert symbol.dtype == test_binary.dtype
        assert np.sum(symbol) > 0  # Should contain some symbol pixels
        
    def test_extract_symbol_error(self, segmenter, test_binary):
        """Test symbol extraction error handling."""
        invalid_box = {'x': -1, 'y': -1, 'width': 1000, 'height': 1000}
        symbol = segmenter._extract_symbol(test_binary, invalid_box)
        
        assert symbol.size == 0
        
    def test_segment_success(self, segmenter, test_binary):
        """Test full segmentation pipeline."""
        symbols = segmenter.segment(test_binary)
        
        assert len(symbols) > 0
        for symbol in symbols:
            assert 'image' in symbol
            assert 'position' in symbol
            assert 'size' in symbol
            assert 'area' in symbol
            
            assert isinstance(symbol['image'], np.ndarray)
            assert symbol['image'].size > 0
            assert 'x' in symbol['position']
            assert 'y' in symbol['position']
            assert 'width' in symbol['size']
            assert 'height' in symbol['size']
            
    def test_segment_no_symbols(self, segmenter):
        """Test segmentation with no symbols present."""
        empty_image = np.zeros((100, 100), dtype=np.uint8)
        symbols = segmenter.segment(empty_image)
        
        assert len(symbols) == 0
        
    def test_segment_error(self, segmenter, test_binary):
        """Test segmentation error handling."""
        with patch.object(segmenter, '_get_bounding_boxes', side_effect=Exception("Test error")):
            symbols = segmenter.segment(test_binary)
            
        assert len(symbols) == 0
