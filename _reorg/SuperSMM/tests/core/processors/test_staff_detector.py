"""Tests for the Staff Detector module."""

import pytest
import numpy as np
from src.core.processors.staff_detector import (
    detect_staff_lines,
    _filter_horizontal_lines,
    _estimate_staff_line_spacing,
    _group_staff_lines
)

@pytest.fixture
def test_image():
    """Create a test binary image with horizontal lines."""
    image = np.zeros((200, 200), dtype=np.uint8)
    # Add 5 horizontal lines (staff system)
    y_positions = [50, 60, 70, 80, 90]
    for y in y_positions:
        image[y, 20:180] = 255
    return image

@pytest.fixture
def test_lines():
    """Create test line data."""
    return np.array([
        [[20, 50, 180, 50]],  # Line 1
        [[20, 60, 180, 60]],  # Line 2
        [[20, 70, 180, 70]],  # Line 3
        [[20, 80, 180, 80]],  # Line 4
        [[20, 90, 180, 90]],  # Line 5
        [[20, 150, 180, 150]]  # Separate line
    ])

def test_detect_staff_lines(test_image):
    """Test staff line detection."""
    result = detect_staff_lines(test_image)
    
    assert 'total_lines' in result
    assert 'horizontal_lines' in result
    assert 'staff_line_spacing' in result
    assert 'staff_systems' in result
    assert 'line_positions' in result

def test_filter_horizontal_lines(test_lines):
    """Test horizontal line filtering."""
    filtered = _filter_horizontal_lines(test_lines)
    
    assert len(filtered) == 6  # All lines are horizontal
    assert all(line[1] == line[3] for line in filtered)  # y1 == y2

def test_estimate_staff_line_spacing():
    """Test staff line spacing estimation."""
    lines = [
        [0, 50, 100, 50],
        [0, 60, 100, 60],
        [0, 70, 100, 70]
    ]
    
    spacing = _estimate_staff_line_spacing(lines)
    assert spacing == 10.0  # Lines are 10 pixels apart

def test_group_staff_lines():
    """Test grouping lines into staff systems."""
    lines = [
        [0, 50, 100, 50],
        [0, 60, 100, 60],
        [0, 70, 100, 70],
        [0, 150, 100, 150]  # Separate line
    ]
    
    systems = _group_staff_lines(lines, max_gap=20)
    
    assert len(systems) == 2  # Two groups
    assert len(systems[0]) == 3  # First group has 3 lines
    assert len(systems[1]) == 1  # Second group has 1 line

def test_empty_input():
    """Test handling of empty input."""
    result = detect_staff_lines(np.zeros((100, 100), dtype=np.uint8))
    
    assert result['total_lines'] == 0
    assert result['horizontal_lines'] == 0
    assert result['staff_line_spacing'] == 0.0
    assert len(result['staff_systems']) == 0
    assert len(result['line_positions']) == 0
