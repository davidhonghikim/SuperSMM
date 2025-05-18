"""Unit tests for preprocessing validators."""

import pytest
import numpy as np
from src.preprocessing.validators import validate_image, validate_size_range, ensure_odd

def test_validate_image_none():
    """Test validation of None input."""
    is_valid, error = validate_image(None)
    assert not is_valid
    assert "None" in error

def test_validate_image_wrong_type():
    """Test validation of non-numpy array input."""
    is_valid, error = validate_image([1, 2, 3])
    assert not is_valid
    assert "numpy.ndarray" in error

def test_validate_image_wrong_dims():
    """Test validation of array with wrong dimensions."""
    # 1D array
    arr_1d = np.array([1, 2, 3])
    is_valid, error = validate_image(arr_1d)
    assert not is_valid
    assert "dimensions" in error
    
    # 4D array
    arr_4d = np.zeros((10, 10, 10, 10))
    is_valid, error = validate_image(arr_4d)
    assert not is_valid
    assert "dimensions" in error

def test_validate_image_valid():
    """Test validation of valid image arrays."""
    # 2D grayscale
    gray = np.zeros((100, 100))
    is_valid, error = validate_image(gray)
    assert is_valid
    assert error == ""
    
    # 3D RGB
    rgb = np.zeros((100, 100, 3))
    is_valid, error = validate_image(rgb)
    assert is_valid
    assert error == ""
    
    # 3D RGBA
    rgba = np.zeros((100, 100, 4))
    is_valid, error = validate_image(rgba)
    assert is_valid
    assert error == ""

def test_validate_size_range():
    """Test size range validation."""
    # Valid size
    is_valid, error = validate_size_range(50, 0, 100, "test_size")
    assert is_valid
    assert error == ""
    
    # Below minimum
    is_valid, error = validate_size_range(-1, 0, 100, "test_size")
    assert not is_valid
    assert "below minimum" in error
    
    # Above maximum
    is_valid, error = validate_size_range(200, 0, 100, "test_size")
    assert not is_valid
    assert "exceeds maximum" in error

def test_ensure_odd():
    """Test odd number enforcement."""
    # Already odd
    assert ensure_odd(7, "test_param") == 7
    
    # Convert even to odd
    assert ensure_odd(8, "test_param") == 9
    
    # Negative numbers
    assert ensure_odd(-7, "test_param") == -7
    assert ensure_odd(-8, "test_param") == -7
