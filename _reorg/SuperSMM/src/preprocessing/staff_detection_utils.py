import cv2
import numpy as np
import logging
from typing import List, Tuple, Optional

# Initialize a module-level logger.
# This can be configured from outside, or we can pass a logger instance to functions.
# For now, let's use a default logger.
logger = logging.getLogger(__name__)


def _group_staff_lines_by_y(
    lines: List[Tuple[int, int, int, int]],
    y_tolerance=5,
    logger_instance: Optional[logging.Logger] = None,
) -> List[Tuple[int, int, int, int]]:
    """
    Groups detected line segments by their y-coordinate to form consolidated staff lines.

    Args:
        lines (List[Tuple[int, int, int, int]]): A list of line segments, where each segment
                                                 is (x1, y1, x2, y2). It's assumed for
                                                 horizontal-ish lines y1 and y2 are very close,
                                                 and the primary y-coordinate for grouping is y1 (or an average).
                                                 This implementation specifically uses line[1] (y1) for sorting and initial grouping.
        y_tolerance (int): The maximum difference in y-coordinates for lines to be
                           considered part of the same staff line.
        logger_instance (Optional[logging.Logger]): An optional logger instance.

    Returns:
        List[Tuple[int, int, int, int]]: A list of merged line segments,
                                         each represented as (min_x, avg_y, max_x, avg_y).
    """
    log = logger_instance or logger
    if not lines:
        log.debug("No lines provided to _group_staff_lines_by_y.")
        return []

    # Sort by y-coordinate (line[1]), then by x-coordinate (line[0]) for consistent processing
    lines.sort(key=lambda line: (line[1], line[0]))
    log.debug(f"Sorted lines for grouping: {lines}")

    merged_lines = []
    if not lines:  # Should be caught by the first check, but as a safeguard
        return []

    # Initialize the first group with the first line
    current_group_y_coords = [lines[0][1]]  # Store all y-coords in the current group
    current_group_segments = [
        lines[0]
    ]  # Store all segments (x1,y1,x2,y2) in the current group

    for i in range(1, len(lines)):
        current_line_segment = lines[i]
        line_y = current_line_segment[1]  # Use y1 for comparison

        # Calculate the average y-coordinate of the current group
        avg_group_y = sum(current_group_y_coords) / len(current_group_y_coords)

        if abs(line_y - avg_group_y) <= y_tolerance:
            # Add current line to the existing group
            current_group_y_coords.append(line_y)
            current_group_segments.append(current_line_segment)
        else:
            # Current line starts a new group. Finalize the previous group.
            if current_group_segments:
                # Calculate the representative y for the finalized group
                final_y = int(
                    round(sum(current_group_y_coords) / len(current_group_y_coords))
                )
                # Merge segments: take min of all x1s and max of all x2s
                min_x1 = min(s[0] for s in current_group_segments)
                max_x2 = max(s[2] for s in current_group_segments)
                merged_lines.append((min_x1, final_y, max_x2, final_y))
                log.debug(
                    f"Finalized group: {(min_x1, final_y, max_x2, final_y)} from {len(current_group_segments)} segments."
                )

            # Start a new group with the current line
            current_group_y_coords = [line_y]
            current_group_segments = [current_line_segment]

    # Finalize the last group after the loop
    if current_group_segments:
        final_y = int(round(sum(current_group_y_coords) / len(current_group_y_coords)))
        min_x1 = min(s[0] for s in current_group_segments)
        max_x2 = max(s[2] for s in current_group_segments)
        merged_lines.append((min_x1, final_y, max_x2, final_y))
        log.debug(
            f"Finalized last group: {(min_x1, final_y, max_x2, final_y)} from {len(current_group_segments)} segments."
        )

    log.info(
        f"Grouped {len(lines)} initial line segments into {len(merged_lines)} merged lines."
    )
    return merged_lines


def detect_staff_lines(
    binary_image: np.ndarray, logger_instance: Optional[logging.Logger] = None
) -> np.ndarray:
    """
    Detects staff lines in a binarized image using Hough Transform and grouping.

    Args:
        binary_image (np.ndarray): The input binary image (black background, white foreground).
        logger_instance (Optional[logging.Logger]): An optional logger instance.

    Returns:
        np.ndarray: An array of detected staff lines, each represented as (x1, y, x2, y).
                    Returns an empty array if no lines are detected or if input is invalid.
    """
    log = logger_instance or logger
    if binary_image is None:
        log.error("Input binary_image to detect_staff_lines is None.")
        return np.array([], dtype=np.int32).reshape(0, 4)

    if len(binary_image.shape) != 2 or binary_image.dtype != np.uint8:
        log.error(
            f"detect_staff_lines expects a 2D uint8 binary image. Got shape {binary_image.shape}, dtype {binary_image.dtype}"
        )
        return np.array([], dtype=np.int32).reshape(0, 4)

    h, w = binary_image.shape
    if w == 0 or h == 0:
        log.warning("Input binary_image for staff detection is empty.")
        return np.array([], dtype=np.int32).reshape(0, 4)

    # Adjust HoughLinesP parameters based on image size
    # Threshold: number of points to detect a line. Proportional to image width.
    # MinLineLength: minimum length of a line. Proportional to image width.
    # MaxLineGap: maximum allowed gap between points on the same line. Proportional to image width.
    hough_threshold = max(
        50, w // 10
    )  # Lower threshold for potentially fragmented lines
    min_line_length = max(
        50, w // 5
    )  # Lines should be reasonably long compared to width
    max_line_gap = max(10, w // 30)  # Allow larger gaps for fragmented real-world lines

    log.debug(
        f"HoughLinesP params: threshold={hough_threshold}, minLineLength={min_line_length}, maxLineGap={max_line_gap}"
    )

    lines = cv2.HoughLinesP(
        binary_image,
        rho=1,
        theta=np.pi / 180,
        threshold=hough_threshold,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap,
    )

    if lines is None:
        log.info("No lines detected by HoughLinesP.")
        return np.array([], dtype=np.int32).reshape(0, 4)

    log.debug(f"HoughLinesP detected {len(lines)} raw line segments.")

    # Filter for horizontal lines and average their y-coordinates
    horizontal_lines = []
    y_diff_tolerance = max(
        3, h // 100
    )  # Tolerance for y1 vs y2 to be considered horizontal

    for line_segment in lines:
        x1, y1, x2, y2 = line_segment[0]
        if abs(y1 - y2) < y_diff_tolerance:  # Line is mostly horizontal
            # Ensure x1 <= x2 for consistency
            if x1 > x2:
                x1, x2 = x2, x1
                y1, y2 = (
                    y2,
                    y1,
                )  # Swap y as well to keep segment integrity if they were different

            avg_y = (y1 + y2) // 2
            horizontal_lines.append(
                (x1, avg_y, x2, avg_y)
            )  # Store as (x1, y_avg, x2, y_avg)
        # else:
        # log.debug(f"Filtered out non-horizontal line: {(x1,y1,x2,y2)} with y_diff {abs(y1-y2)}")

    if not horizontal_lines:
        log.info(
            "No sufficiently horizontal lines found after filtering HoughP output."
        )
        return np.array([], dtype=np.int32).reshape(0, 4)

    log.info(f"Found {len(horizontal_lines)} horizontal-ish segments before grouping.")

    # Group these horizontal line segments by their y-coordinate
    # The y_tolerance for grouping should be small, related to staff line thickness or small variations
    grouping_y_tolerance = max(
        3, h // 150
    )  # e.g., for height 1500, tol=10; for 300, tol=3
    grouped_lines = _group_staff_lines_by_y(
        horizontal_lines, y_tolerance=grouping_y_tolerance, logger_instance=log
    )

    final_lines_np = np.array(grouped_lines, dtype=np.int32)
    # Ensure correct shape for empty result (0,4) not (0,)
    if final_lines_np.ndim == 1 and final_lines_np.size == 0:
        final_lines_np = final_lines_np.reshape(0, 4)

    log.info(f"Detected {len(final_lines_np)} consolidated staff-like line segments.")
    return final_lines_np
