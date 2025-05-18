"""Tests for the Image Processor module."""

import pytest
import numpy as np
import cv2
from src.core.processors.image_processor import preprocess_image

@pytest.fixture
def test_image():
    """Create a test image."""
    return np.ones((100, 100, 3), dtype=np.uint8) * 255

def test_preprocess_image_success(test_image):
    """Test successful image preprocessing."""
    result = preprocess_image(test_image)
    
    assert 'grayscale' in result
    assert 'denoised' in result
    assert 'binary' in result
    assert 'threshold_params' in result
    
    assert result['grayscale'].shape == (100, 100)
    assert result['denoised'].shape == (100, 100)
    assert result['binary'].shape == (100, 100)
    
    # Check threshold parameters
    assert 'block_size' in result['threshold_params']
    assert 'c_value' in result['threshold_params']

def test_preprocess_image_invalid_input():
    """Test handling of invalid input."""
    invalid_image = np.ones((100, 100), dtype=np.uint8)  # Missing color channels
    
    result = preprocess_image(invalid_image)
    
    assert 'error' in result
    assert result['binary'] is None
    assert result['denoised'] is None
    assert 'grayscale' in result  # Should still return grayscale version

def test_preprocess_image_empty():
    """Test handling of empty image."""
    empty_image = np.array([])
    
    result = preprocess_image(empty_image.reshape(0, 0, 3))
    
    assert 'error' in result
    assert result['binary'] is None
