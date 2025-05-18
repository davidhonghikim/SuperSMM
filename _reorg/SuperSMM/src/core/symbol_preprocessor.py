"""
Symbol Preprocessor Module

This module provides image preprocessing capabilities for musical symbol recognition.
It handles various preprocessing steps including grayscale conversion, noise reduction,
thresholding, and symbol segmentation.
"""

# Standard library imports
import logging

# Third-party imports
import cv2
import numpy as np

# Local application imports
from ..segmentation.symbol_segmenter import SymbolSegmenter


class SymbolPreprocessor:
    """Handles preprocessing of musical symbols for recognition.

    This class implements various image processing techniques to prepare
    musical symbols for recognition. It includes methods for noise reduction,
    thresholding, and symbol segmentation.

    Attributes:
        logger (logging.Logger): Logger instance for this class
    """

    def __init__(self):
        self.logger = logging.getLogger("symbol_preprocessor")

    def preprocess_sheet_music(self, image: np.ndarray) -> dict:
        """
        Comprehensive sheet music preprocessing pipeline

        Args:
            image (np.ndarray): Input sheet music image

        Returns:
            Dict of preprocessed images for different stages
        """
        results = {
            "original": image,
            "grayscale": self.convert_to_grayscale(image),
            "denoised": self.denoise_image(image),
            "binarized": self.binarize_image(image),
            "staff_removed": self.remove_staff_lines(image),
        }
        return results

    def convert_to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """Convert image to grayscale"""
        try:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        except Exception as e:
            self.logger.error("Grayscale conversion error: %s", e)
            return image

    def process_image_file(self, image_path: str, output_path: str = None) -> bool:
        """Process a single image file and optionally save the result.

        Args:
            image_path (str): Path to the input image file
            output_path (str, optional): Path to save the processed image

        Returns:
            bool: True if processing was successful, False otherwise
        """
        # TODO: implement this method
        pass

    def denoise_image(self, image: np.ndarray) -> np.ndarray:
        """
        Advanced image denoising

        Techniques:
        1. Fast Non-Local Means Denoising
        2. Bilateral Filtering
        """
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image

            # Non-local means denoising
            denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)

            # Bilateral filtering for edge preservation
            filtered = cv2.bilateralFilter(denoised, 9, 75, 75)

            return filtered
        except Exception as e:
            self.logger.error("Denoising error: %s", e)
            return image

    def binarize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Advanced binarization with multiple techniques

        Techniques:
        1. Adaptive Thresholding
        2. Otsu's Binarization
        3. Sauvola Binarization
        """
        try:
            # Ensure grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image

            # Adaptive thresholding
            adaptive_thresh = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
            )

            # Otsu's method
            _, otsu_thresh = cv2.threshold(
                gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
            )

            # Combine methods
            combined = cv2.bitwise_and(adaptive_thresh, otsu_thresh)

            return combined
        except Exception as e:
            self.logger.error("Thresholding error: %s", e)
            return image

    def remove_staff_lines(self, image: np.ndarray) -> np.ndarray:
        """
        Remove staff lines while preserving musical symbols

        Techniques:
        1. Morphological operations
        2. Line detection and removal
        """
        try:
            # Ensure binary image
            if len(image.shape) == 3:
                binary = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                _, binary = cv2.threshold(
                    binary, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
                )
            else:
                binary = image

            # Horizontal line detection
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
            detected_lines = cv2.morphologyEx(
                binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=2
            )

            # Remove detected lines
            staff_removed = cv2.subtract(binary, detected_lines)

            return staff_removed
        except Exception as e:
            self.logger.error("Staff line removal error: %s", e)
            return image

    def extract_connected_components(self, binary_image: np.ndarray) -> list:
        """
        Extract potential musical symbols as connected components

        Returns:
            List of symbol candidate images
        """
        try:
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                binary_image, connectivity=8
            )
            symbols = []
            for label_idx in range(1, num_labels):
                x, y, w, h, area = stats[label_idx]
                if (10 < area < 500) and (0.5 < w / h < 2):
                    symbol = binary_image[y : y + h, x : x + w]
                    symbols.append(symbol)
            return symbols
        except Exception as e:
            self.logger.error("Symbol extraction error: %s", e)
            return []

    def extract_symbol_candidates(self, preprocessed_image: np.ndarray) -> list:
        """
        Extract potential musical symbol candidates from preprocessed image
        using the refined SymbolSegmenter.

        Args:
            preprocessed_image (np.ndarray): Preprocessed binary image,
                                          expected to be suitable for SymbolSegmenter
                                          (e.g., inverted binary).

        Returns:
            List of potential symbol images (np.ndarray crops)
        """
        try:
            # Instantiate SymbolSegmenter with the robust 'heuristics_and_morphology' strategy
            segmenter_config = {
                "segmentation_strategy": "heuristics_and_morphology",
                # Other config values will use SymbolSegmenter's defaults if not specified here
            }
            segmenter = SymbolSegmenter(config=segmenter_config)

            # segment_symbols returns a dict, not a tuple
            result = segmenter.segment_symbols(preprocessed_image)
            symbol_crops = result.get("symbol_crops", [])

            self.logger.info(
                f"Extracted {len(symbol_crops)} symbol candidates using SymbolSegmenter"
            )
            return symbol_crops
        except Exception as e:
            self.logger.error(
                f"Symbol candidate extraction error using SymbolSegmenter: {e}"
            )
            # For more detailed debugging, uncomment the following:
            # import traceback
            # self.logger.error(f"Traceback: {traceback.format_exc()}")
            return []


# Example usage


def main():
    # Load a sample sheet music image
    image = cv2.imread(
        "/Users/danger/CascadeProjects/LOO/SheetMasterMusic/sample_files/pdfs/Somewhere Over the Rainbow.pdf[0]"
    )

    preprocessor = SymbolPreprocessor()
    processed = preprocessor.preprocess_sheet_music(image)

    # Extract symbols
    symbols = preprocessor.extract_connected_components(processed["staff_removed"])

    # Visualize results
    for i, symbol in enumerate(symbols):
        cv2.imwrite(f"/tmp/symbol_{i}.png", symbol)


if __name__ == "__main__":
    main()
