"""Tests for the image preprocessor module."""

import numpy as np
import pytest
import cv2
from unittest.mock import patch

from src.core.processors.image_preprocessor import ImagePreprocessor


@pytest.fixture
def preprocessor():
    """Create an image preprocessor instance."""
    return ImagePreprocessor()


@pytest.fixture
def test_image():
    """Create a test image."""
    return np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)


class TestImagePreprocessor:
    """Test suite for ImagePreprocessor class."""
    
    def test_to_grayscale_color_image(self, preprocessor, test_image):
        """Test grayscale conversion of color image."""
        result = preprocessor.to_grayscale(test_image)
        
        assert result.ndim == 2
        assert result.shape == (100, 100)
        assert result.dtype == np.uint8
        
    def test_to_grayscale_already_gray(self, preprocessor):
        """Test grayscale conversion of already gray image."""
        gray_image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        result = preprocessor.to_grayscale(gray_image)
        
        assert result is gray_image  # Should return same object
        
    def test_to_grayscale_error(self, preprocessor):
        """Test grayscale conversion error handling."""
        invalid_image = np.array([])  # Invalid image
        result = preprocessor.to_grayscale(invalid_image)
        
        assert result is invalid_image  # Should return input on error
        
    def test_denoise(self, preprocessor, test_image):
        """Test image denoising."""
        gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
        result = preprocessor.denoise(gray)
        
        assert result.shape == gray.shape
        assert result.dtype == gray.dtype
        
    def test_denoise_error(self, preprocessor):
        """Test denoising error handling."""
        invalid_image = np.array([])
        result = preprocessor.denoise(invalid_image)
        
        assert result is invalid_image
        
    def test_threshold(self, preprocessor, test_image):
        """Test image thresholding."""
        gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
        result = preprocessor.threshold(gray)
        
        assert result.shape == gray.shape
        assert result.dtype == np.uint8
        assert set(np.unique(result)).issubset({0, 255})  # Binary image
        
    def test_threshold_error(self, preprocessor):
        """Test thresholding error handling."""
        invalid_image = np.array([])
        result = preprocessor.threshold(invalid_image)
        
        assert result is invalid_image
        
    def test_preprocess_pipeline_success(self, preprocessor, test_image):
        """Test full preprocessing pipeline."""
        result = preprocessor.preprocess(test_image)
        
        assert 'grayscale' in result
        assert 'denoised' in result
        assert 'binary' in result
        assert 'threshold_params' in result
        assert 'error' not in result
        
        assert result['grayscale'].ndim == 2
        assert result['denoised'].ndim == 2
        assert result['binary'].ndim == 2
        assert isinstance(result['threshold_params'], dict)
        
    def test_preprocess_pipeline_invalid_input(self, preprocessor):
        """Test preprocessing pipeline with invalid input."""
        result = preprocessor.preprocess(None)
        
        assert 'error' in result
        assert result['grayscale'] is None
        assert result['denoised'] is None
        assert result['binary'] is None
        
    def test_preprocess_pipeline_error(self, preprocessor, test_image):
        """Test preprocessing pipeline error handling."""
        with patch.object(preprocessor, 'to_grayscale', side_effect=Exception("Test error")):
            result = preprocessor.preprocess(test_image)
            
        assert 'error' in result
        assert str(result['error']) == "Test error"
