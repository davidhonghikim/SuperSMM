"""Tests for the staff line removal module."""

import numpy as np
import pytest
import cv2
from unittest.mock import patch

from src.core.processors.staff_remover import StaffRemover


@pytest.fixture
def remover():
    """Create a staff remover instance."""
    return StaffRemover()


@pytest.fixture
def test_binary():
    """Create a test binary image with horizontal lines."""
    image = np.zeros((200, 300), dtype=np.uint8)
    
    # Add horizontal staff lines
    for y in range(20, 180, 20):
        image[y:y+2, 10:290] = 255
        
    return image


class TestStaffRemover:
    """Test suite for StaffRemover class."""
    
    def test_detect_lines_success(self, remover, test_binary):
        """Test successful staff line detection."""
        lines = remover._detect_lines(test_binary)
        
        assert len(lines) > 0
        for line in lines:
            assert 'x' in line
            assert 'y' in line
            assert 'width' in line
            assert 'height' in line
            assert line['width'] >= remover.min_line_length
            
    def test_detect_lines_no_lines(self, remover):
        """Test line detection with no lines present."""
        empty_image = np.zeros((100, 100), dtype=np.uint8)
        lines = remover._detect_lines(empty_image)
        
        assert len(lines) == 0
        
    def test_detect_lines_error(self, remover):
        """Test line detection error handling."""
        invalid_image = np.array([])
        lines = remover._detect_lines(invalid_image)
        
        assert len(lines) == 0
        
    def test_remove_lines_success(self, remover, test_binary):
        """Test successful staff line removal."""
        # First detect lines
        lines = remover._detect_lines(test_binary)
        assert len(lines) > 0
        
        # Remove lines
        result = remover._remove_lines(test_binary, lines)
        
        assert result.shape == test_binary.shape
        assert result.dtype == test_binary.dtype
        assert np.sum(result) < np.sum(test_binary)  # Should have removed some pixels
        
    def test_remove_lines_no_lines(self, remover, test_binary):
        """Test line removal with no lines detected."""
        result = remover._remove_lines(test_binary, [])
        
        assert np.array_equal(result, test_binary)  # Should return original image
        
    def test_remove_lines_error(self, remover, test_binary):
        """Test line removal error handling."""
        lines = [{'y': -1, 'height': 1000}]  # Invalid line info
        result = remover._remove_lines(test_binary, lines)
        
        assert np.array_equal(result, test_binary)  # Should return original image
        
    def test_remove_staff_success(self, remover, test_binary):
        """Test full staff removal pipeline."""
        result = remover.remove_staff(test_binary)
        
        assert 'image' in result
        assert 'staff_lines' in result
        assert 'line_count' in result
        assert 'error' not in result
        
        assert result['image'].shape == test_binary.shape
        assert len(result['staff_lines']) > 0
        assert result['line_count'] > 0
        
    def test_remove_staff_no_lines(self, remover):
        """Test staff removal with no lines present."""
        empty_image = np.zeros((100, 100), dtype=np.uint8)
        result = remover.remove_staff(empty_image)
        
        assert 'error' in result
        assert result['staff_lines'] == []
        assert np.array_equal(result['image'], empty_image)
        
    def test_remove_staff_error(self, remover, test_binary):
        """Test staff removal error handling."""
        with patch.object(remover, '_detect_lines', side_effect=Exception("Test error")):
            result = remover.remove_staff(test_binary)
            
        assert 'error' in result
        assert str(result['error']) == "Test error"
        assert result['staff_lines'] == []
