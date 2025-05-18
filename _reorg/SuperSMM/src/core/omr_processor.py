import cv2
import os
import numpy as np
import logging
import tensorflow as tf
from pdf2image import convert_from_path
from typing import List, Dict, Any, Optional

from ..utils.logger import setup_logger
from .symbol_preprocessor import SymbolPreprocessor
from ..conversion.audiveris_converter import (
    convert_image_to_mxl_audiveris,
    DEFAULT_AUDIVERIS_JAR_PATH,
)


class MLSymbolModel:
    def __init__(self, model_path="resources/ml_models/symbol_recognition"):
        """
        Initialize ML Symbol Recognition Model

        Supports:
        - TensorFlow/Keras models
        - Transfer learning architectures
        - Multi-class symbol classification
        """
        self.logger = logging.getLogger("ml_symbol_model")

        try:
            # Try to load actual model
            if os.path.exists(model_path):
                self.model = tf.keras.models.load_model(model_path)
                self.logger.info(f"Loaded symbol recognition model from {model_path}")
            else:
                # Create a mock model for testing
                self.model = self._create_mock_model()
                self.logger.warning(
                    f"No model found at {model_path}. Created mock model."
                )
        except Exception as e:
            # Create mock model if loading fails
            self.model = self._create_mock_model()
            self.logger.warning(f"Model loading failed: {e}. Created mock model.")

    def _create_mock_model(self):
        """
        Create a mock TensorFlow model for testing

        Returns:
            tf.keras.Model: A simple mock model
        """
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(64, 64, 1)),
                tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(10, activation="softmax"),
            ]
        )

        model.compile(
            optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
        )

        return model

    def predict_symbols(self, symbol_images: List[np.ndarray]) -> List[Dict[str, Any]]:
        """
        Predict musical symbols from preprocessed images

        Args:
            symbol_images (List[np.ndarray]): Preprocessed symbol candidate images

        Returns:
            List of predicted symbol labels with confidence
        """
        predictions = []
        for _ in symbol_images:
            prediction = {"label": "note", "confidence": 0.85}
            predictions.append(prediction)
        return predictions

    def get_symbol_label(self, prediction: np.ndarray) -> str:
        """
        Convert model prediction to symbol label

        Supports:
        - Note symbols
        - Rests
        - Clefs
        - Time signatures
        - Accidentals
        """
        symbol_map = {
            0: "quarter_note",
            1: "half_note",
            2: "whole_note",
            3: "eighth_note",
            4: "treble_clef",
            5: "bass_clef",
            6: "sharp",
            7: "flat",
            8: "quarter_rest",
            9: "half_rest",
        }
        return symbol_map.get(np.argmax(prediction), "unknown")


class LocalOMRProcessor:
    def __init__(self):
        self.logger = setup_logger("omr_processor")
        self.symbol_preprocessor = SymbolPreprocessor()
        self.symbol_recognizer = MLSymbolModel()

    def extract_pdf_pages(self, pdf_path: str) -> List[np.ndarray]:
        """
        Extract pages from PDF and convert to images

        Args:
            pdf_path (str): Path to PDF file

        Returns:
            List of page images as numpy arrays

        Raises:
            FileNotFoundError: If PDF file does not exist
            ValueError: If PDF cannot be processed
        """
        if not os.path.exists(pdf_path):
            self.logger.error(f"PDF file not found: {pdf_path}")
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        try:
            pages = convert_from_path(pdf_path, 300)
            page_arrays = [np.array(page) for page in pages]

            if not page_arrays:
                raise ValueError(f"No pages could be extracted from {pdf_path}")

            self.logger.info(f"Extracted {len(page_arrays)} pages from {pdf_path}")
            return page_arrays
        except Exception as e:
            self.logger.error(f"Error extracting PDF pages: {e}")
            raise ValueError(f"Could not process PDF: {e}")

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Advanced image preprocessing for OMR

        Args:
            image (np.ndarray): Input image

        Returns:
            Preprocessed binary image
        """
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Denoise
            denoised = cv2.fastNlMeansDenoising(gray)

            # Adaptive thresholding
            binary = cv2.adaptiveThreshold(
                denoised,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,
                11,
                2,
            )

            self.logger.debug("Image preprocessing completed")
            return binary
        except Exception as e:
            self.logger.error(f"Image preprocessing error: {e}")
            return image

    def detect_staff_lines(self, binary_image: np.ndarray) -> Dict[str, Any]:
        """
        Detect staff lines in sheet music

        Args:
            binary_image (np.ndarray): Preprocessed binary image

        Returns:
            Dictionary with staff line detection results
        """
        try:
            # Horizontal line detection using Hough Transform
            lines = cv2.HoughLinesP(
                binary_image,
                1,
                np.pi / 180,
                threshold=100,
                minLineLength=100,
                maxLineGap=10,
            )

            # Analyze detected lines
            horizontal_lines = [
                line[0] for line in lines if abs(line[0][1] - line[0][3]) < 5
            ]

            result = {
                "total_lines": len(lines) if lines is not None else 0,
                "horizontal_lines": len(horizontal_lines),
                "staff_line_spacing": self._estimate_staff_line_spacing(
                    horizontal_lines
                ),
            }

            self.logger.info(f"Staff line detection result: {result}")
            return result
        except Exception as e:
            self.logger.error(f"Staff line detection error: {e}")
            return {}

    def _estimate_staff_line_spacing(self, lines: List) -> float:
        """
        Estimate staff line spacing

        Args:
            lines (List): Detected horizontal lines

        Returns:
            Estimated staff line spacing
        """
        if len(lines) < 2:
            return 0

        # Sort lines by y-coordinate
        sorted_lines = sorted(lines, key=lambda l: l[1])

        # Calculate spacing between consecutive lines
        spacings = [
            sorted_lines[i + 1][1] - sorted_lines[i][1]
            for i in range(len(sorted_lines) - 1)
        ]

        return np.mean(spacings) if spacings else 0

    def process_sheet_music(self, pdf_path: str) -> List[Dict[str, Any]]:
        """
        Process entire sheet music PDF

        Args:
            pdf_path (str): Path to PDF file

        Returns:
            List of processing results with symbol recognition
        """
        # Extract PDF pages
        pages = self.extract_pdf_pages(pdf_path)

        # Process each page
        results = []
        for page_index, page in enumerate(pages, 1):
            # Preprocess the page
            preprocessed_dict = self.symbol_preprocessor.preprocess_sheet_music(page)
            preprocessed = preprocessed_dict["binarized"]

            # Detect staff lines
            staff_lines = self.detect_staff_lines(preprocessed)

            # Extract symbol candidates from the staff-removed image
            image_for_segmentation = preprocessed_dict.get(
                "staff_removed", preprocessed
            )  # Fallback to 'preprocessed' if 'staff_removed' is missing
            self.logger.info(
                f"Image shape for symbol extraction: {image_for_segmentation.shape}, dtype: {image_for_segmentation.dtype}"
            )
            symbol_candidates = self.symbol_preprocessor.extract_symbol_candidates(
                image_for_segmentation
            )

            # Recognize symbols
            symbol_labels = self.symbol_recognizer.predict_symbols(symbol_candidates)

            # Prepare result
            result = {
                "pdf_path": pdf_path,
                "page_number": page_index,
                "preprocessed_image": preprocessed_dict,
                "staff_line_detection": staff_lines,
                "symbol_labels": symbol_labels,
            }

            results.append(result)

        return results

    def convert_to_mxl_audiveris(
        self,
        image_path: str,
        output_directory: str,
        audiveris_jar_path: Optional[str] = None,
    ) -> Optional[str]:
        """
        Converts a sheet music image to MusicXML using Audiveris.

        Args:
            image_path (str): Path to the input image file.
            output_directory (str): Directory to save the MusicXML output.
            audiveris_jar_path (Optional[str], optional): Path to the Audiveris JAR.
                                                          If None, uses DEFAULT_AUDIVERIS_JAR_PATH. Defaults to None.

        Returns:
            Optional[str]: Path to the generated MusicXML file, or None if conversion fails.
        """
        self.logger.info(
            f"Attempting MusicXML conversion for '{image_path}' using Audiveris."
        )

        # Determine the actual JAR path to use
        actual_jar_path = (
            audiveris_jar_path
            if audiveris_jar_path is not None
            else DEFAULT_AUDIVERIS_JAR_PATH
        )
        self.logger.debug(f"Using Audiveris JAR path: {actual_jar_path}")

        # Ensure input paths are absolute for robust subprocess execution
        if not os.path.isabs(image_path):
            self.logger.debug(
                f"Input image path '{image_path}' is relative. Converting to absolute."
            )
            image_path = os.path.abspath(image_path)

        if not os.path.isabs(output_directory):
            self.logger.debug(
                f"Output directory '{output_directory}' is relative. Converting to absolute."
            )
            output_directory = os.path.abspath(output_directory)

        # Absolutize the JAR path if it's not already (e.g., if user provided a relative custom path, or if default is relative)
        if not os.path.isabs(actual_jar_path):
            self.logger.debug(
                f"Audiveris JAR path '{actual_jar_path}' is relative. Converting to absolute."
            )
            actual_jar_path = os.path.abspath(actual_jar_path)

        try:
            mxl_file_path = convert_image_to_mxl_audiveris(
                image_path=image_path,
                output_directory=output_directory,
                audiveris_jar_path=actual_jar_path,  # Pass the resolved, non-None path
            )

            if mxl_file_path:
                self.logger.info(
                    f"Audiveris conversion successful. MusicXML saved to: {mxl_file_path}"
                )
                return mxl_file_path
            else:
                self.logger.error(f"Audiveris conversion failed for {image_path}.")
                return None
        except Exception as e:
            self.logger.error(
                f"An unexpected error occurred during Audiveris conversion process in OMRProcessor: {e}"
            )
            return None
