"""Optical Music Recognition (OMR) pipeline.

This module provides a complete pipeline for processing sheet music images
and extracting musical notation data.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any, Union
from pathlib import Path
import json
import numpy as np
import cv2

from ..preprocessing import (
    normalize_image,
    binarize_image,
    deskew_image,
    detect_staff_lines,
    remove_staff_lines,
)

from ..segmentation import (
    BoundingBox,
    detect_symbols,
    group_symbols_by_staff,
    group_symbols_into_measures,
    group_symbols_into_notes,
    group_notes_into_chords,
)

from ..recognition import (
    SymbolRecognizer,
    SymbolRecognitionResult,
    recognize_symbols_in_image,
)


@dataclass
class OMRResult:
    """Result of the OMR pipeline."""

    # Input information
    input_path: str
    image_shape: Tuple[int, int, int]  # (height, width, channels)

    # Processing metadata
    processing_time: float  # in seconds
    timestamp: float  # Unix timestamp

    # Detection results
    staffs: List[Any]  # List of detected staffs
    symbols: List[Dict[str, Any]]  # List of detected symbols
    recognition_results: List[Dict[str, Any]]  # List of recognition results

    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "input_path": self.input_path,
            "image_shape": self.image_shape,
            "processing_time": self.processing_time,
            "timestamp": self.timestamp,
            "staffs": self.staffs,
            "symbols": self.symbols,
            "recognition_results": [
                r if isinstance(r, dict) else r.to_dict()
                for r in self.recognition_results
            ],
            "metadata": self.metadata,
        }

    def save(self, output_path: Union[str, Path]) -> bool:
        """Save results to a JSON file.

        Args:
            output_path: Path to save the results

        Returns:
            True if save was successful, False otherwise
        """
        try:
            with open(output_path, "w") as f:
                json.dump(self.to_dict(), f, indent=2)
            return True
        except Exception as e:
            logging.getLogger(__name__).error(f"Error saving OMR results: {e}")
            return False

    @classmethod
    def load(cls, file_path: Union[str, Path]) -> Optional["OMRResult"]:
        """Load results from a JSON file.

        Args:
            file_path: Path to the JSON file

        Returns:
            OMRResult object or None if loading failed
        """
        try:
            with open(file_path, "r") as f:
                data = json.load(f)

            return cls(
                input_path=data["input_path"],
                image_shape=tuple(data["image_shape"]),
                processing_time=data["processing_time"],
                timestamp=data["timestamp"],
                staffs=data["staffs"],
                symbols=data["symbols"],
                recognition_results=[
                    SymbolRecognitionResult.from_dict(r) if isinstance(r, dict) else r
                    for r in data["recognition_results"]
                ],
                metadata=data.get("metadata", {}),
            )
        except Exception as e:
            logging.getLogger(__name__).error(f"Error loading OMR results: {e}")
            return None


class OMRPipeline:
    """Optical Music Recognition pipeline.

    This class provides a complete pipeline for processing sheet music images
    and extracting musical notation data.
    """

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        use_gpu: bool = False,
        debug: bool = False,
        output_dir: Optional[Union[str, Path]] = None,
    ):
        """Initialize the OMR pipeline.

        Args:
            model_path: Path to pre-trained symbol recognition model
            use_gpu: Whether to use GPU for inference
            debug: Whether to enable debug mode (saves intermediate results)
            output_dir: Directory to save output files (if None, no files are saved)
        """
        self.logger = logging.getLogger(__name__)
        self.debug = debug
        self.output_dir = Path(output_dir) if output_dir else None

        # Create output directories if they don't exist
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize symbol recognizer
        self.recognizer = SymbolRecognizer(
            model_path=model_path,
            use_gpu=use_gpu,
        )

        # Initialize processing statistics
        self.stats = {
            "total_images_processed": 0,
            "total_processing_time": 0.0,
            "avg_processing_time": 0.0,
        }

    def process_image(
        self,
        image_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
    ) -> OMRResult:
        """Process a single sheet music image.

        Args:
            image_path: Path to the input image
            output_path: Path to save the results (if None, uses output_dir)

        Returns:
            OMRResult object containing the processing results
        """
        start_time = time.time()

        try:
            # Load the image
            self.logger.info(f"Processing image: {image_path}")
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Failed to load image: {image_path}")

            # Preprocess the image
            self.logger.info("Preprocessing image...")
            preprocessed = self._preprocess_image(image)

            # Detect staff lines
            self.logger.info("Detecting staff lines...")
            staffs = detect_staff_lines(preprocessed)

            # Remove staff lines for symbol detection
            self.logger.info("Removing staff lines...")
            staff_removed = remove_staff_lines(preprocessed, staffs)

            # Detect symbols
            self.logger.info("Detecting symbols...")
            symbols = detect_symbols(staff_removed)

            # Group symbols by staff
            self.logger.info("Grouping symbols by staff...")
            symbols_by_staff = group_symbols_by_staff(symbols, staffs)

            # Recognize symbols
            self.logger.info("Recognizing symbols...")
            recognition_results = []

            for staff_idx, staff_symbols in symbols_by_staff.items():
                if staff_idx == -1:  # Unassigned symbols
                    continue

                # Group symbols into measures
                measures = group_symbols_into_measures(staff_symbols, staffs[staff_idx])

                # Process each measure
                for measure_idx, measure_symbols in enumerate(measures):
                    # Group symbols into notes
                    notes = group_symbols_into_notes(measure_symbols, staffs[staff_idx])

                    # Group notes into chords
                    chords = group_notes_into_chords(notes)

                    # Recognize symbols
                    for symbol in measure_symbols:
                        result = self.recognizer.recognize(
                            image=preprocessed,
                            symbols=[symbol],
                            staff=staffs[staff_idx],
                        )
                        if result:
                            recognition_results.extend(result)

            # Calculate processing time
            processing_time = time.time() - start_time

            # Update statistics
            self.stats["total_images_processed"] += 1
            self.stats["total_processing_time"] += processing_time
            self.stats["avg_processing_time"] = (
                self.stats["total_processing_time"]
                / self.stats["total_images_processed"]
            )

            # Create result object
            result = OMRResult(
                input_path=str(image_path),
                image_shape=image.shape,
                processing_time=processing_time,
                timestamp=time.time(),
                staffs=[s.to_dict() for s in staffs],
                symbols=[s.to_dict() for s in symbols],
                recognition_results=recognition_results,
                metadata={
                    "num_staffs": len(staffs),
                    "num_symbols": len(symbols),
                    "num_recognized": len(recognition_results),
                },
            )

            # Save results if output path is specified
            if output_path or self.output_dir:
                output_path = output_path or (
                    self.output_dir / f"{Path(image_path).stem}_result.json"
                )
                result.save(output_path)
                self.logger.info(f"Results saved to: {output_path}")

            self.logger.info(
                f"Processing complete. Time: {processing_time:.2f}s, "
                f"Staffs: {len(staffs)}, Symbols: {len(symbols)}"
            )

            return result

        except Exception as e:
            self.logger.error(
                f"Error processing image {image_path}: {e}", exc_info=True
            )
            raise

    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess an image for OMR.

        Args:
            image: Input image (BGR format)

        Returns:
            Preprocessed image (grayscale)
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Normalize image
        normalized = normalize_image(gray)

        # Binarize image
        binary = binarize_image(normalized)

        # Deskew image
        deskewed, _ = deskew_image(binary)

        return deskewed

    def process_batch(
        self,
        input_paths: List[Union[str, Path]],
        output_dir: Optional[Union[str, Path]] = None,
        num_workers: int = 1,
    ) -> List[OMRResult]:
        """Process a batch of sheet music images.

        Args:
            input_paths: List of input image paths
            output_dir: Directory to save results (if None, uses self.output_dir)
            num_workers: Number of worker processes to use (not yet implemented)

        Returns:
            List of OMRResult objects
        """
        results = []
        output_dir = Path(output_dir) if output_dir else self.output_dir

        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)

        # Process each image
        for i, image_path in enumerate(input_paths):
            try:
                self.logger.info(
                    f"Processing image {i+1}/{len(input_paths)}: {image_path}"
                )

                # Determine output path
                if output_dir:
                    output_path = output_dir / f"{Path(image_path).stem}_result.json"
                else:
                    output_path = None

                # Process the image
                result = self.process_image(image_path, output_path)
                results.append(result)

            except Exception as e:
                self.logger.error(f"Error processing {image_path}: {e}", exc_info=True)

        return results

    def get_statistics(self) -> Dict[str, Any]:
        """Get processing statistics.

        Returns:
            Dictionary containing processing statistics
        """
        return self.stats.copy()

    def reset_statistics(self) -> None:
        """Reset processing statistics."""
        self.stats = {
            "total_images_processed": 0,
            "total_processing_time": 0.0,
            "avg_processing_time": 0.0,
        }


def create_omr_pipeline(
    model_path: Optional[Union[str, Path]] = None,
    use_gpu: bool = False,
    debug: bool = False,
    output_dir: Optional[Union[str, Path]] = None,
) -> OMRPipeline:
    """Create and configure an OMR pipeline.

    This is a convenience function that creates and configures an OMR pipeline
    with default settings.

    Args:
        model_path: Path to pre-trained symbol recognition model
        use_gpu: Whether to use GPU for inference
        debug: Whether to enable debug mode
        output_dir: Directory to save output files

    Returns:
        Configured OMRPipeline instance
    """
    return OMRPipeline(
        model_path=model_path,
        use_gpu=use_gpu,
        debug=debug,
        output_dir=output_dir,
    )
