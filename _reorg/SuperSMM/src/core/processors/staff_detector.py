"""
Staff line detection for OMR.

This module handles the detection and analysis of staff lines in sheet music
using computer vision techniques.
"""

import cv2
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger("staff_detector")


def detect_staff_lines(binary_image: np.ndarray) -> Dict[str, Any]:
    """
    Detect staff lines in sheet music.

    Args:
        binary_image (np.ndarray): Preprocessed binary image

    Returns:
        Dict[str, Any]: Staff line detection results
            - total_lines: Total number of detected lines
            - horizontal_lines: Number of horizontal lines
            - staff_line_spacing: Average spacing between staff lines
            - staff_systems: List of detected staff systems
            - line_positions: List of line y-positions
    """
    try:
        # Detect lines using Hough Transform
        lines = cv2.HoughLinesP(
            binary_image,
            1,
            np.pi / 180,
            threshold=100,
            minLineLength=100,
            maxLineGap=10,
        )

        if lines is None:
            logger.warning("No lines detected in image")
            return _create_empty_result()

        # Filter horizontal lines
        horizontal_lines = _filter_horizontal_lines(lines)

        # Group lines into staff systems
        staff_systems = _group_staff_lines(horizontal_lines)

        # Calculate staff line spacing
        spacing = _estimate_staff_line_spacing(horizontal_lines)

        result = {
            "total_lines": len(lines),
            "horizontal_lines": len(horizontal_lines),
            "staff_line_spacing": spacing,
            "staff_systems": staff_systems,
            "line_positions": [line[1] for line in horizontal_lines],
        }

        logger.info(
            f"Staff line detection completed: {len(staff_systems)} systems found"
        )
        return result

    except Exception as e:
        logger.error(f"Staff line detection error: {e}")
        return _create_empty_result()


def _filter_horizontal_lines(lines: np.ndarray) -> List[List[int]]:
    """
    Filter and extract horizontal lines from HoughLinesP output.

    Args:
        lines (np.ndarray): Lines detected by HoughLinesP

    Returns:
        List[List[int]]: Filtered horizontal lines
    """
    horizontal_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if abs(y2 - y1) < 5:  # Nearly horizontal
            horizontal_lines.append([x1, y1, x2, y2])
    return horizontal_lines


def _estimate_staff_line_spacing(lines: List[List[int]]) -> float:
    """
    Estimate average spacing between staff lines.

    Args:
        lines (List[List[int]]): Detected horizontal lines

    Returns:
        float: Estimated staff line spacing
    """
    if len(lines) < 2:
        return 0.0

    # Sort lines by y-coordinate
    sorted_lines = sorted(lines, key=lambda l: l[1])

    # Calculate spacing between consecutive lines
    spacings = [
        sorted_lines[i + 1][1] - sorted_lines[i][1]
        for i in range(len(sorted_lines) - 1)
    ]

    return np.mean(spacings) if spacings else 0.0


def _group_staff_lines(
    lines: List[List[int]], max_gap: int = 50
) -> List[List[List[int]]]:
    """
    Group staff lines into staff systems.

    Args:
        lines (List[List[int]]): Detected horizontal lines
        max_gap (int): Maximum vertical gap between lines in same staff

    Returns:
        List[List[List[int]]]: Grouped staff lines
    """
    if not lines:
        return []

    # Sort lines by y-coordinate
    sorted_lines = sorted(lines, key=lambda l: l[1])

    # Group lines into staff systems
    systems = [[sorted_lines[0]]]
    for line in sorted_lines[1:]:
        if line[1] - systems[-1][-1][1] > max_gap:
            # Start new staff system
            systems.append([line])
        else:
            # Add to current staff system
            systems[-1].append(line)

    return systems


def _create_empty_result() -> Dict[str, Any]:
    """Create an empty result dictionary when no lines are detected."""
    return {
        "total_lines": 0,
        "horizontal_lines": 0,
        "staff_line_spacing": 0.0,
        "staff_systems": [],
        "line_positions": [],
    }
