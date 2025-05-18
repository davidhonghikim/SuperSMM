# symbol_segmenter.py
import numpy as np
import cv2
from typing import List, Dict, Any, Tuple
import logging
import time
import os


class SymbolSegmenter:
    def __init__(self, config: Dict[str, Any] = None):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("[SYMBOL_SEGMENTER_INIT]")
        default_config = {
            "min_symbol_size": 5,
            "max_symbol_size": 300,
            "min_aspect_ratio": 0.05,
            "max_aspect_ratio": 10.0,
            "text_min_area": 50,
            "text_min_aspect_ratio": 3.0,
            "debug_image_path": "debug_output/segmentation",
            "contour_retrieval_mode": cv2.RETR_LIST,
            "contour_approximation_method": cv2.CHAIN_APPROX_SIMPLE,
            "segmentation_strategy": "connected_components",
            "gaussian_blur_ksize": (5, 5),
            "adaptive_thresh_block_size": 11,
            "adaptive_thresh_C": 2,
            "morph_open_ksize": (3, 3),
            "morph_close_ksize": (3, 3),
            "min_contour_area": 5,
            "max_contour_area": 20000,
            "symbol_aspect_ratio_min": 0.05,
            "symbol_aspect_ratio_max": 10.0,
            "text_aspect_ratio_min_heuristic": 2.0,
        }
        if config:
            default_config.update(config)
        self.config = default_config

    def _save_debug_image(self, image_name: str, image_data: np.ndarray):
        if not self.config.get("debug_image_path"):
            return
        try:
            debug_dir = self.config["debug_image_path"]
            os.makedirs(debug_dir, exist_ok=True)
            save_path = os.path.join(debug_dir, image_name)
            img_to_save = image_data.copy()
            if img_to_save.dtype != np.uint8:
                img_to_save = img_to_save.astype(np.uint8)
            if img_to_save.max() == 1 and img_to_save.min() == 0:
                img_to_save *= 255
            cv2.imwrite(save_path, img_to_save)
            self.logger.debug(f"Saved debug image: {save_path}")
        except Exception as e:
            self.logger.error(f"Error saving debug image {image_name}: {e}")

    def _segment_symbols_heuristic_morphology(
        self, preprocessed_image: np.ndarray
    ) -> Tuple[List[np.ndarray], List[Tuple[np.ndarray, Tuple[int, int, int, int]]]]:
        self.logger.info("Using heuristic and morphology strategy")
        gray_image = preprocessed_image
        if len(preprocessed_image.shape) == 3 and preprocessed_image.shape[2] == 3:
            gray_image = cv2.cvtColor(preprocessed_image, cv2.COLOR_BGR2GRAY)

        blurred = cv2.GaussianBlur(gray_image, self.config["gaussian_blur_ksize"], 0)
        self._save_debug_image("heuristic_1_blurred.png", blurred)

        binary_image = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            self.config["adaptive_thresh_block_size"],
            self.config["adaptive_thresh_C"],
        )
        self._save_debug_image("heuristic_2_adaptive_binary.png", binary_image)

        kernel_open = np.ones(self.config["morph_open_ksize"], np.uint8)
        opened = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel_open)
        self._save_debug_image("heuristic_3_opened.png", opened)

        kernel_close = np.ones(self.config["morph_close_ksize"], np.uint8)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_close)
        self._save_debug_image("heuristic_4_closed.png", closed)

        contours, _ = cv2.findContours(
            closed,
            self.config["contour_retrieval_mode"],
            self.config["contour_approximation_method"],
        )
        self.logger.debug(f"Found {len(contours)} raw contours (heuristic)")

        symbol_candidates = []
        text_candidates = []

        for i, cnt in enumerate(contours):
            x, y, w, h = cv2.boundingRect(cnt)
            area = cv2.contourArea(cnt)
            if not (
                self.config["min_contour_area"] < area < self.config["max_contour_area"]
            ):
                continue

            aspect_ratio = w / float(h) if h > 0 else 0
            # Crop from original preprocessed image for better quality if it was grayscale, else from closed binary
            if len(preprocessed_image.shape) == 2 or (
                len(preprocessed_image.shape) == 3 and preprocessed_image.shape[2] == 1
            ):
                cropped_image = gray_image[y : y + h, x : x + w]
            else:  # If original was color, heuristic processing was on grayscale, so crop from processed 'closed' image
                cropped_image = closed[y : y + h, x : x + w]

            is_text = (
                aspect_ratio > self.config["text_aspect_ratio_min_heuristic"]
                and area > self.config["text_min_area"]
            )
            is_symbol = (
                self.config["symbol_aspect_ratio_min"]
                < aspect_ratio
                < self.config["symbol_aspect_ratio_max"]
                and w > self.config["min_symbol_size"]
                and h > self.config["min_symbol_size"]
                and w < self.config["max_symbol_size"]
                and h < self.config["max_symbol_size"]
            )

            if is_text:
                text_candidates.append((cropped_image, (x, y, w, h)))
            elif is_symbol:
                symbol_candidates.append(cropped_image)

        self.logger.info(
            f"Heuristic strategy: {len(symbol_candidates)} symbols, {len(text_candidates)} text candidates"
        )
        return symbol_candidates, text_candidates

    def _segment_symbols_connected_components(
        self, preprocessed_image: np.ndarray
    ) -> Tuple[List[np.ndarray], List[Tuple[np.ndarray, Tuple[int, int, int, int]]]]:
        self.logger.info("Using connected components strategy")
        gray_image = preprocessed_image
        if len(preprocessed_image.shape) == 3 and preprocessed_image.shape[2] == 3:
            gray_image = cv2.cvtColor(preprocessed_image, cv2.COLOR_BGR2GRAY)

        _, binary_image = cv2.threshold(
            gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        self._save_debug_image("cc_1_ots_binary.png", binary_image)

        contours, _ = cv2.findContours(
            binary_image,
            self.config["contour_retrieval_mode"],
            self.config["contour_approximation_method"],
        )
        self.logger.debug(f"Found {len(contours)} raw contours (CC)")

        symbol_candidates = []
        text_candidates = []

        for i, cnt in enumerate(contours):
            x, y, w, h = cv2.boundingRect(cnt)
            area = w * h
            if not (
                self.config["min_symbol_size"] <= w < self.config["max_symbol_size"]
                and self.config["min_symbol_size"] <= h < self.config["max_symbol_size"]
            ):
                continue

            aspect_ratio = w / float(h) if h > 0 else 0
            cropped_image = binary_image[y : y + h, x : x + w]

            is_text = (
                aspect_ratio > self.config["text_min_aspect_ratio"]
                and area > self.config["text_min_area"]
            )
            is_symbol = (
                self.config["min_aspect_ratio"]
                < aspect_ratio
                < self.config["max_aspect_ratio"]
            )

            if is_text:
                text_candidates.append((cropped_image, (x, y, w, h)))
            elif is_symbol:
                symbol_candidates.append(cropped_image)

        self.logger.info(
            f"CC strategy: {len(symbol_candidates)} symbols, {len(text_candidates)} text candidates"
        )
        return symbol_candidates, text_candidates

    def segment_symbols(self, preprocessed_image: np.ndarray) -> Dict[str, Any]:
        self.logger.info(
            f"Segmenting symbols using strategy: {self.config['segmentation_strategy']}"
        )
        start_time = time.time()

        symbol_crops = []
        text_crops_with_bboxes = []

        if self.config["segmentation_strategy"] == "heuristics_and_morphology":
            symbol_crops, text_crops_with_bboxes = (
                self._segment_symbols_heuristic_morphology(preprocessed_image)
            )
        elif self.config["segmentation_strategy"] == "connected_components":
            symbol_crops, text_crops_with_bboxes = (
                self._segment_symbols_connected_components(preprocessed_image)
            )
        else:
            self.logger.error(
                f"Unknown segmentation strategy: {self.config['segmentation_strategy']}. Falling back to CC."
            )
            symbol_crops, text_crops_with_bboxes = (
                self._segment_symbols_connected_components(preprocessed_image)
            )

        processing_time = time.time() - start_time
        self.logger.info(
            f"Segmentation completed in {processing_time:.2f}s: {len(symbol_crops)} symbols, {len(text_crops_with_bboxes)} text regions."
        )
        return {
            "symbol_crops": symbol_crops,
            "text_crops_with_bboxes": text_crops_with_bboxes,
            "metadata": {
                "processing_time_seconds": processing_time,
                "strategy_used": self.config["segmentation_strategy"],
            },
        }


# SYMBOL_SEGMENTER_DEBUG_VERSION: 2025-05-10-TRACE-V1
import numpy as np
import cv2
from typing import List, Dict, Any
import logging

print("[SYMBOL_SEGMENTER_DEBUG_VERSION: 2025-05-10-TRACE-V1]")


class SymbolSegmenter:
    """
    Advanced symbol segmentation for sheet music processing.

    Handles the extraction and isolation of individual musical symbols
    from preprocessed sheet music images.
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the symbol segmenter with optional configuration.

        Args:
            config (Dict[str, Any], optional): Configuration dictionary
            for segmentation parameters. Defaults to None.
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(
            f"[SYMBOL_SEGMENTER_INIT] Version: 2025-05-10-TRACE-V1 (using merged config)"
        )
        default_config = {
            "min_symbol_size": 5,
            "max_symbol_size": 300,
            "min_aspect_ratio": 0.05,
            "max_aspect_ratio": 10.0,
            "text_min_area": 50,  # Default min area for text candidates
            "text_min_aspect_ratio": 3.0,  # Default min aspect ratio for text
            "debug_image_path": "debug_output/segmentation",
            "contour_retrieval_mode": cv2.RETR_LIST,  # Heuristic method might work better with RETR_LIST
            "contour_approximation_method": cv2.CHAIN_APPROX_SIMPLE,
            "segmentation_strategy": "connected_components",  # Default strategy
            # Configs specific to heuristic_morphology strategy, matching extract_and_save_symbols defaults
            "gaussian_blur_ksize": (5, 5),
            "adaptive_thresh_block_size": 11,
            "adaptive_thresh_C": 2,
            "morph_open_ksize": (3, 3),
            "morph_close_ksize": (3, 3),
            "min_contour_area": 5,  # from extract_and_save_symbols
            "max_contour_area": 20000,  # from extract_and_save_symbols
            "symbol_aspect_ratio_min": 0.05,  # from extract_and_save_symbols
            "symbol_aspect_ratio_max": 10.0,  # from extract_and_save_symbols
            "text_aspect_ratio_min": 2.0,  # from extract_and_save_symbols (more lenient than text_min_aspect_ratio)
        }
        if config:
            default_config.update(config)  # Merge provided config into defaults
        self.config = default_config

    def find_connected_components(self, binary_image: np.ndarray) -> List[np.ndarray]:
        """
        Find and extract individual symbol candidates using connected component analysis.

        Args:
            binary_image (np.ndarray): Preprocessed binary image of sheet music.

        Returns:
            List[np.ndarray]: List of extracted symbol images
        """
        # Validate input
        if binary_image is None or binary_image.size == 0:
            return []

        # Ensure binary image
        if len(binary_image.shape) > 2:
            binary_image = cv2.cvtColor(binary_image, cv2.COLOR_BGR2GRAY)

        # Threshold if not already binary

        import cv2
        import numpy as np

        # Try both Otsu and adaptive thresholding for diagnostics
        # Otsu (already applied)
        contours_otsu, _ = cv2.findContours(
            binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        print(f"[SEG_DEBUG] Otsu contours found: {len(contours_otsu)}")
        self.logger.info(f"[SEG_DEBUG] Otsu contours found: {len(contours_otsu)}")
        # Adaptive thresholding
        adaptive = cv2.adaptiveThreshold(
            binary_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        contours_adapt, _ = cv2.findContours(
            adaptive, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        print(f"[SEG_DEBUG] Adaptive contours found: {len(contours_adapt)}")
        self.logger.info(f"[SEG_DEBUG] Adaptive contours found: {len(contours_adapt)}")
        # Use Otsu by default
        contours = contours_otsu
        if len(contours_otsu) == 0 and len(contours_adapt) > 0:
            print("[SEG_DEBUG] Switching to adaptive threshold contours!")
            self.logger.info("[SEG_DEBUG] Switching to adaptive threshold contours!")
            contours = contours_adapt

        # --- DEBUG: Save the image used for contour finding ---
        if self.config.get("debug_image_path"):
            try:
                os.makedirs(self.config["debug_image_path"], exist_ok=True)

                # Prepare binary_image for saving (ensure 0-255, uint8)
                binary_image_to_save = binary_image.copy()
                if binary_image_to_save.dtype != np.uint8:
                    binary_image_to_save = binary_image_to_save.astype(np.uint8)
                if (
                    binary_image_to_save.max() == 1 and binary_image_to_save.min() == 0
                ):  # if it's 0/1, scale to 0/255
                    binary_image_to_save = binary_image_to_save * 255

                save_path_binary = os.path.join(
                    self.config["debug_image_path"], "segmentation_debug_binary.png"
                )
                cv2.imwrite(save_path_binary, binary_image_to_save)
                self.logger.debug(
                    f"[SEG_DEBUG] Saved binary image to {save_path_binary}"
                )

                # Prepare contour_image_input for saving (it's the same as binary_image here)
                contour_input_to_save = binary_image.copy()
                if contour_input_to_save.dtype != np.uint8:
                    contour_input_to_save = contour_input_to_save.astype(np.uint8)
                if (
                    contour_input_to_save.max() == 1
                    and contour_input_to_save.min() == 0
                ):
                    contour_input_to_save = contour_input_to_save * 255

                save_path_contour_input = os.path.join(
                    self.config["debug_image_path"], "debug_contour_input.png"
                )
                cv2.imwrite(save_path_contour_input, contour_input_to_save)
                self.logger.debug(
                    f"[SEG_DEBUG] Saved contour input image to {save_path_contour_input}"
                )

            except Exception as e:
                self.logger.error(
                    f"[SEG_DEBUG] Error saving debug pre-contour images: {e}"
                )
        # --- END DEBUG ---

        symbol_candidates = []
        text_candidates = []
        for i, cnt in enumerate(contours):
            x, y, w, h = cv2.boundingRect(cnt)
            print(
                f"[SEG_DEBUG] Contour {i} RAW_SIZE: w={w}, h={h}, min_cfg={self.config['min_symbol_size']}, max_cfg={self.config['max_symbol_size']}"
            )
            self.logger.info(
                f"[SEG_DEBUG] Contour {i} RAW_SIZE: w={w}, h={h}, min_cfg={self.config['min_symbol_size']}, max_cfg={self.config['max_symbol_size']}"
            )
            if i < 5:
                print(f"[SEG_DEBUG] Contour {i}: x={x}, y={y}, w={w}, h={h}")
                self.logger.info(f"[SEG_DEBUG] Contour {i}: x={x}, y={y}, w={w}, h={h}")
            area = w * h
            aspect_ratio = max(w / h, h / w)
            # Heuristic: likely text if small area or extreme aspect ratio
            if (
                w >= self.config["min_symbol_size"]
                and h >= self.config["min_symbol_size"]
                and w < self.config["max_symbol_size"]
                and h < self.config["max_symbol_size"]
            ):
                symbol = binary_image[y : y + h, x : x + w]
                # Improved text filter: area >= 120, width >= 8, height >= 8, aspect_ratio > 8.0
                if aspect_ratio > 8.0 and area >= 120 and w >= 8 and h >= 8:
                    text_candidates.append((symbol, (x, y, w, h)))
                elif area >= 100 and area <= 20000 and aspect_ratio <= 8.0:
                    symbol_candidates.append(symbol)
                else:
                    # too small or ambiguous, skip as noise
                    pass
            else:
                print(
                    f"[SEG_DEBUG] Contour {i} filtered out: area={area}, aspect_ratio={aspect_ratio:.2f}"
                )
                self.logger.info(
                    f"[SEG_DEBUG] Contour {i} filtered out: area={area}, aspect_ratio={aspect_ratio:.2f}"
                )
        print(
            f"[SEG_DEBUG] {len(symbol_candidates)} symbol candidates after bounding box filtering."
        )
        self.logger.info(
            f"[SEG_DEBUG] {len(symbol_candidates)} symbol candidates after bounding box filtering."
        )
        if len(contours) == 0:
            print(
                "[SEG_DEBUG] No contours found in binary image! Check thresholding and preprocessing."
            )
            self.logger.warning(
                "[SEG_DEBUG] No contours found in binary image! Check thresholding and preprocessing."
            )
        if len(symbol_candidates) == 0:
            print(
                "[SEG_DEBUG] All contours filtered out by size. Check min/max symbol size."
            )
            self.logger.warning(
                "[SEG_DEBUG] All contours filtered out by size. Check min/max symbol size."
            )
        return symbol_candidates, text_candidates

    def segment_symbols(self, preprocessed_image: np.ndarray) -> Dict[str, Any]:
        """
        Comprehensive symbol segmentation pipeline

        Args:
            preprocessed_image (np.ndarray): Preprocessed sheet music image

        Returns:
            Dict[str, Any]: Segmentation results
        """
        import time
        import numpy as np
        import cv2

        start_time = time.time()
        # Input image diagnostics
        self.logger.info(
            f"[SEG_DEBUG] segment_symbols input: type={type(preprocessed_image)}, "
            f"shape={getattr(preprocessed_image, 'shape', None)}, "
            f"dtype={getattr(preprocessed_image, 'dtype', None)}"
        )
        unique_vals = (
            np.unique(preprocessed_image)
            if hasattr(preprocessed_image, "dtype")
            and np.issubdtype(preprocessed_image.dtype, np.integer)
            else None
        )
        if unique_vals is not None:
            print(
                f"[SEG_DEBUG] Input image unique values: min={unique_vals.min()}, max={unique_vals.max()}, count={len(unique_vals)}"
            )
            self.logger.info(
                f"[SEG_DEBUG] Input image unique values: min={unique_vals.min()}, max={unique_vals.max()}, count={len(unique_vals)}"
            )
        # Convert to grayscale if needed
        if len(preprocessed_image.shape) == 3 and preprocessed_image.shape[2] == 3:
            gray_image = cv2.cvtColor(preprocessed_image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = preprocessed_image
        # Binarize image
        _, binary_image = cv2.threshold(
            gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        # Save binary image for inspection
        if self.config.get("debug_image_path"):
            try:
                os.makedirs(self.config["debug_image_path"], exist_ok=True)

                # Prepare binary_image for saving (ensure 0-255, uint8)
                binary_image_to_save = binary_image.copy()
                if binary_image_to_save.dtype != np.uint8:
                    binary_image_to_save = binary_image_to_save.astype(np.uint8)
                if (
                    binary_image_to_save.max() == 1 and binary_image_to_save.min() == 0
                ):  # if it's 0/1, scale to 0/255
                    binary_image_to_save = binary_image_to_save * 255

                save_path_binary = os.path.join(
                    self.config["debug_image_path"], "segmentation_debug_binary.png"
                )
                cv2.imwrite(save_path_binary, binary_image_to_save)
                self.logger.debug(
                    f"[SEG_DEBUG] Saved binary image to {save_path_binary}"
                )

            except Exception as e:
                self.logger.error(f"[SEG_DEBUG] Error saving debug binary image: {e}")
        # Find symbol candidates
        symbol_candidates, text_candidates = self.find_connected_components(
            binary_image
        )
        print(f"[SEG_DEBUG] Found {len(symbol_candidates)} raw symbol candidates.")
        self.logger.info(
            f"[SEG_DEBUG] Found {len(symbol_candidates)} raw symbol candidates."
        )
        # Filtering diagnostics (if implemented)
        filtered_candidates = []
        for idx, c in enumerate(symbol_candidates):
            if hasattr(c, "shape") and len(c.shape) >= 2:
                h, w = c.shape[:2]
                if h < 5 or w < 5:
                    print(
                        f"[SEG_DEBUG] Candidate {idx} filtered: too small (h={h}, w={w})"
                    )
                    self.logger.info(
                        f"[SEG_DEBUG] Candidate {idx} filtered: too small (h={h}, w={w})"
                    )
                    continue
            filtered_candidates.append(c)
        if len(filtered_candidates) == 0:
            print(
                "[SEG_DEBUG] All candidates filtered out. Check size/aspect ratio thresholds!"
            )
            self.logger.warning(
                "[SEG_DEBUG] All candidates filtered out. Check size/aspect ratio thresholds!"
            )
        else:
            print(
                f"[SEG_DEBUG] {len(filtered_candidates)} candidates remain after filtering."
            )
            self.logger.info(
                f"[SEG_DEBUG] {len(filtered_candidates)} candidates remain after filtering."
            )
        processing_time = time.time() - start_time
        return {
            "symbol_candidates": filtered_candidates,
            "metadata": {
                "total_symbols": len(filtered_candidates),
                "processing_time": processing_time,
            },
        }
