"""Optical Music Recognition (OMR) Pipeline.

This module implements the main OMR pipeline that orchestrates the entire process
of converting sheet music images into machine-readable music notation.
"""

import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from dataclasses import asdict

import cv2
import numpy as np
import psutil
import tensorflow as tf

from ..exceptions import (
    PreprocessingError,
    SegmentationError,
    ConfigurationError,
    RecognitionError,
    create_error_handler,
)
from ...config.omr_config import OMRConfig
from ...config.config_manager import ConfigManager
from ...utils.decorators import log_performance
from ...utils.logging_config import setup_logging
from ...preprocessing.advanced_preprocessor import AdvancedPreprocessor
from ...segmentation.symbol import SymbolSegmenter
from ...recognition.symbol_recognizer import SymbolRecognizer


def load_model(model_path: Union[str, Path], **kwargs) -> Any:
    """Load a machine learning model from the given path.
    
    This is a convenience function that loads a pre-trained model from disk.
    It's primarily used for testing and demonstration purposes.
    
    Args:
        model_path: Path to the model file
        **kwargs: Additional arguments to pass to the model loading function
        
    Returns:
        Loaded model
        
    Raises:
        FileNotFoundError: If the model file doesn't exist
        ValueError: If the model format is not supported
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
        
    # This is a placeholder implementation
    # In a real implementation, you would load the actual model here
    # For example:
    # return tf.keras.models.load_model(model_path, **kwargs)
    
    # For testing purposes, return a mock model
    class MockModel:
        def predict(self, x):
            return np.zeros((len(x), 100))  # Return dummy predictions
            
    return MockModel()


class OMRPipeline:
    """Main OMR pipeline class that orchestrates the entire OMR process.

    This class manages the preprocessing, segmentation, and recognition stages
    of the OMR pipeline, along with configuration, logging, and error handling.
    """

    def __init__(
        self,
        config_path: Optional[Union[str, Path]] = None,
        preprocessor: Optional[AdvancedPreprocessor] = None,
        segmenter: Optional[SymbolSegmenter] = None,
        recognizer: Optional[SymbolRecognizer] = None,
        custom_config: Optional[Union[Dict[str, Any], OMRConfig]] = None,
    ) -> None:
        """Initialize the OMR pipeline.

        Args:
            config_path: Path to configuration file
            preprocessor: Optional custom preprocessor instance
            segmenter: Optional custom segmenter instance
            recognizer: Optional custom recognizer instance
            custom_config: Optional custom configuration
        """
        # Initialize logger first with default settings
        setup_logging(
            logger_name="omr_pipeline", log_level=logging.INFO, log_dir="logs"
        )
        self.logger = logging.getLogger("omr_pipeline")

        try:
            # Initialize configuration
            config_manager = ConfigManager(
                config_class=OMRConfig, config_path=config_path, env_prefix="SMM"
            )

            # Update with custom config if provided
            if custom_config:
                if isinstance(custom_config, OMRConfig):
                    self._config = custom_config
                else:
                    config_manager.update(custom_config)
                    self._config = config_manager.get_config()
            else:
                self._config = config_manager.get_config()

            # Initialize logger with config settings
            setup_logging(
                logger_name="omr_pipeline",
                log_level=(
                    logging.INFO
                    if not hasattr(self._config.logging, "level")
                    else self._config.logging.level
                ),
                log_dir=(
                    str(Path(self._config.logging.file).parent)
                    if hasattr(self._config.logging, "file")
                    else "logs"
                ),
            )
            self.logger = logging.getLogger("omr_pipeline")

            # Initialize performance tracking
            self.performance_metrics = {
                "total_processing_time": 0,
                "pages_processed": 0,
                "errors_encountered": 0,
                "max_memory_usage": 0,
            }

            # Initialize modules with configuration
            self.preprocessor = preprocessor or AdvancedPreprocessor(
                config=self._config.preprocessing
            )
            self.segmenter = segmenter or SymbolSegmenter(
                min_symbol_size=self._config.segmentation.min_symbol_size,
                max_symbol_size=self._config.segmentation.max_symbol_size,
                min_aspect_ratio=getattr(self._config.segmentation, 'min_aspect_ratio', 0.1),
                max_aspect_ratio=getattr(self._config.segmentation, 'max_aspect_ratio', 10.0),
                min_solidity=getattr(self._config.segmentation, 'min_solidity', 0.3),
                min_extent=getattr(self._config.segmentation, 'min_extent', 0.2),
                merge_overlapping=getattr(self._config.segmentation, 'merge_overlapping', True),
                iou_threshold=getattr(self._config.segmentation, 'iou_threshold', 0.3)
            )
            self.recognizer = recognizer or SymbolRecognizer(
                model_path=getattr(self._config.recognition, 'model_path', None),
                use_gpu=getattr(self._config.recognition, 'use_gpu', False),
                confidence_threshold=getattr(self._config.recognition, 'confidence_threshold', 0.5)
            )

            # Create output and cache directories
            self.output_dir = Path(self._config.output_dir)
            self.cache_dir = Path(self._config.cache_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.cache_dir.mkdir(parents=True, exist_ok=True)

            # Validate modules
            self._validate_modules()

        except ConfigurationError:
            raise
        except Exception as e:
            self.logger.error("Pipeline module initialization failed: %s", str(e))
            raise ConfigurationError(
                "Failed to initialize OMR pipeline modules",
                context={"error": str(e), "config_path": config_path},
            ) from e

    @property
    def config(self) -> Dict[str, Any]:
        """Get the current configuration.

        Returns:
            Complete configuration dictionary
        """
        return asdict(self._config)

    def _validate_modules(self) -> None:
        """
        Validate that all required modules are properly initialized

        Raises:
            ConfigurationError: If any module is improperly configured
        """
        if not hasattr(self, "preprocessor") or not self.preprocessor:
            raise ConfigurationError("Preprocessor module not properly initialized")
        if not hasattr(self, "segmenter") or not self.segmenter:
            raise ConfigurationError("Segmenter module not properly initialized")
        if not hasattr(self, "recognizer") or not self.recognizer:
            raise ConfigurationError("Recognizer module not properly initialized")

    @log_performance
    def process_image(self, *args, **kwargs) -> Dict[str, Any]:
        """Alias for process_sheet_music for backward compatibility.
        
        This method is maintained for backward compatibility with existing code.
        New code should use process_sheet_music instead.
        
        Returns:
            Dict[str, Any]: Processed sheet music results
        """
        return self.process_sheet_music(*args, **kwargs)
        
    def process_batch(self, input_paths: List[Union[str, Path]], output_dir: Optional[Union[str, Path]] = None) -> List[Dict[str, Any]]:
        """Process multiple sheet music images in batch.
        
        Args:
            input_paths: List of paths to sheet music images or numpy arrays
            output_dir: Base directory for saving outputs (optional)
            
        Returns:
            List of processing results for each input
            
        Raises:
            ValueError: If no input paths are provided
        """
        if not input_paths:
            raise ValueError("No input paths provided")
            
        results = []
        for i, input_path in enumerate(input_paths):
            try:
                result = self.process_sheet_music(
                    input_path_or_image=input_path,
                    output_dir_base=output_dir,
                    page_identifier=str(i)
                )
                results.append(result)
            except Exception as e:
                self.logger.error(f"Error processing {input_path}: {str(e)}")
                results.append({"error": str(e), "input_path": str(input_path)})
                
        return results
        
    def _get_input_type(self, input_path_or_image: Union[str, Path, np.ndarray]) -> str:
        """Determine the type of input.
        
        Args:
            input_path_or_image: Input to check (file path or numpy array)
            
        Returns:
            str: One of 'pdf', 'image', or 'array'
            
        Raises:
            ValueError: If input type is not supported
            FileNotFoundError: If input file doesn't exist
        """
        if isinstance(input_path_or_image, np.ndarray):
            return 'array'
            
        if not isinstance(input_path_or_image, (str, Path)):
            raise ValueError(f"Unsupported input type: {type(input_path_or_image)}")
            
        path = Path(input_path_or_image)
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {path}")
            
        if path.suffix.lower() == '.pdf':
            return 'pdf'
        elif path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.tiff', '.bmp']:
            return 'image'
        else:
            raise ValueError(f"Unsupported file type: {path.suffix}")

    def process_sheet_music(
        self,
        input_path_or_image: Union[str, Path, np.ndarray],
        output_dir_base: Optional[Union[str, Path]] = None,
        page_identifier: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Process a sheet music image through the entire pipeline

        Args:
            input_path_or_image: Path to the sheet music image or numpy array
            output_dir_base: Base directory for saving intermediate outputs
            page_identifier: Identifier for the current page when processing multiple pages

        Returns:
            Dict[str, Any]: Processed sheet music results including preprocessing,
                           segmentation, and recognition results

        Raises:
            PreprocessingError: If preprocessing fails
            SegmentationError: If symbol segmentation fails
            RecognitionError: If symbol recognition fails
            ValueError: If PDF has no pages or input is invalid
        """
        start_time = time.time()
        process = psutil.Process()
        initial_memory = process.memory_info().rss / (1024 * 1024)  # MB
        results = {}

        try:
            # Set up output directory with input filename as prefix
            input_path = Path(input_path_or_image) if isinstance(input_path_or_image, (str, Path)) else None
            filename = input_path.stem if input_path and input_path.suffix else 'output'
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Create main output directory with filename and timestamp
            output_dir = Path(output_dir_base or self.output_dir) / f"{filename}_{timestamp}"
            
            # Create page-specific subdirectory if processing multiple pages
            if page_identifier and not str(page_identifier).startswith('page_'):
                output_dir = output_dir / f"page_{page_identifier}"
            elif page_identifier:
                output_dir = output_dir / page_identifier
                
            output_dir.mkdir(parents=True, exist_ok=True)

            # Log start of processing
            self.logger.info("Starting OMR processing")
            if isinstance(input_path_or_image, (str, Path)):
                self.logger.info("Input path: %s", str(input_path_or_image))
            else:
                self.logger.info("Processing numpy array input")

            # Determine input type and preprocess accordingly
            input_type = self._get_input_type(input_path_or_image)
            
            if input_type == 'pdf':
                # Process PDF file
                results = {
                    'pdf_path': str(input_path_or_image),
                    'pages': self.preprocessor.process_pdf(input_path_or_image)
                }
                
                # Process each page's results through segmentation and recognition
                for i, page in enumerate(results['pages']):
                    try:
                        if 'normalized_image' in page and page['normalized_image'] is not None:
                            # Segment symbols
                            page['symbols'] = self.segmenter.segment_symbols(page['normalized_image'])
                            
                            # Recognize symbols if any were found
                            if page['symbols']:
                                page['recognized'] = self.recognizer.recognize(page['symbols'])
                    except Exception as e:
                        self.logger.error(f"Error processing page {i + 1}: {str(e)}", exc_info=True)
                        page['error'] = str(e)
                
            else:  # image or array
                # Process single image or array
                result = self.preprocessor.process_page(
                    input_path_or_image,
                    page_identifier=str(input_path_or_image) if input_type == 'image' else 'input_array',
                    output_dir=str(output_dir)
                )
                
                # Process through segmentation and recognition if we have a valid image
                if 'normalized_image' in result and result['normalized_image'] is not None:
                    try:
                        result['symbols'] = self.segmenter.segment_symbols(result['normalized_image'])
                        
                        if result['symbols']:
                            result['recognized'] = self.recognizer.recognize(result['symbols'])
                    except Exception as e:
                        self.logger.error(f"Processing failed: {str(e)}", exc_info=True)
                        result['error'] = str(e)
                
                results = result

            # Calculate performance metrics
            end_time = time.time()
            final_memory = process.memory_info().rss / (1024 * 1024)  # MB
            processing_time = end_time - start_time

            # Add performance metrics to results
            if isinstance(results, dict):
                results['performance'] = {
                    'processing_time_seconds': processing_time,
                    'memory_usage_mb': final_memory - initial_memory,
                    'timestamp': datetime.now().isoformat(),
                }

            # Update performance metrics
            self.performance_metrics["total_processing_time"] += processing_time
            self.performance_metrics["pages_processed"] += 1
            self.performance_metrics["max_memory_usage"] = max(
                self.performance_metrics["max_memory_usage"], final_memory
            )

            # Export results
            self.export_results(results, output_dir)

            return results

        except Exception as e:
            self.performance_metrics["errors_encountered"] += 1
            self.logger.error("OMR processing failed: %s", str(e), exc_info=True)
            raise PreprocessingError(
                f"Processing failed: {str(e)}",
                context={"input": str(input_path_or_image)},
            ) from e

    def export_results(
        self, results: Dict[str, Any], output_dir: Union[str, Path]
    ) -> None:
        """
        Export processing results to specified directory

        Args:
            results: Processing results dictionary
            output_dir: Directory to export results to
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create a debug directory for intermediate outputs
        debug_dir = output_dir / "debug"
        debug_dir.mkdir(exist_ok=True)

        # Get base filename from the output directory
        base_name = output_dir.parent.name if 'page_' in output_dir.name else output_dir.name
        
        # Remove timestamp from base_name if present
        if '_' in base_name and base_name.count('_') >= 2:  # At least one underscore in the name part and one in the timestamp
            base_name = '_'.join(base_name.split('_')[:-2])  # Remove the last two parts (timestamp components)


        # Save results as JSON
        results_file = output_dir / f"{base_name}_results.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2, default=str)
        self.logger.info(f"Saved results to: {results_file}")

        # Save pages data if available (for PDFs)
        if "pages" in results and isinstance(results["pages"], list):
            for i, page in enumerate(results["pages"]):
                if not isinstance(page, dict):
                    continue
                
                # Get page identifier from results or use index
                page_id = page.get('page_identifier', f"page_{i:03d}")
                if page_id.startswith('page_'):
                    page_suffix = page_id[5:]  # Remove 'page_' prefix
                else:
                    page_suffix = page_id
                
                # Save normalized image if available
                if "normalized_image" in page and page["normalized_image"] is not None:
                    img_path = debug_dir / f"{base_name}_{page_suffix}_normalized.png"
                    cv2.imwrite(str(img_path), page["normalized_image"])
                    self.logger.debug(f"Saved normalized image: {img_path}")
                
                # Save binary image if available
                if "binary_image" in page and page["binary_image"] is not None:
                    img_path = debug_dir / f"{base_name}_{page_suffix}_binary.png"
                    cv2.imwrite(str(img_path), page["binary_image"])
                    self.logger.debug(f"Saved binary image: {img_path}")
                
                # Save staff-removed image if available
                if "no_staff_image" in page and page["no_staff_image"] is not None:
                    img_path = debug_dir / f"{base_name}_{page_suffix}_no_staff.png"
                    cv2.imwrite(str(img_path), page["no_staff_image"])
                    self.logger.debug(f"Saved staff-removed image: {img_path}")
                
                # Save symbols if available
                if "symbols" in page and page["symbols"]:
                    symbols = page["symbols"]
                    if hasattr(symbols[0], 'to_dict'):  # If symbols are objects with to_dict method
                        symbols_dict = [s.to_dict() for s in symbols]
                    else:
                        symbols_dict = symbols
                    
                    symbols_file = debug_dir / f"{base_name}_{page_suffix}_symbols.json"
                    with open(symbols_file, 'w') as f:
                        json.dump(symbols_dict, f, indent=2, default=str)
                    self.logger.debug(f"Saved symbols to: {symbols_file}")

        # For single page results (non-PDF)
        elif isinstance(results, dict):
            # Get page identifier from results or use 'page_000'
            page_id = results.get('page_identifier', 'page_000')
            if page_id.startswith('page_'):
                page_suffix = page_id[5:]  # Remove 'page_' prefix
            else:
                page_suffix = '000'
            
            # Save normalized image if available
            if "normalized_image" in results and results["normalized_image"] is not None:
                img_path = debug_dir / f"{base_name}_{page_suffix}_normalized.png"
                cv2.imwrite(str(img_path), results["normalized_image"])
                self.logger.debug(f"Saved normalized image: {img_path}")
            
            # Save binary image if available
            if "binary_image" in results and results["binary_image"] is not None:
                img_path = debug_dir / f"{base_name}_{page_suffix}_binary.png"
                cv2.imwrite(str(img_path), results["binary_image"])
                self.logger.debug(f"Saved binary image: {img_path}")
            
            # Save staff-removed image if available
            if "no_staff_image" in results and results["no_staff_image"] is not None:
                img_path = debug_dir / f"{base_name}_{page_suffix}_no_staff.png"
                cv2.imwrite(str(img_path), results["no_staff_image"])
                self.logger.debug(f"Saved staff-removed image: {img_path}")

        self.logger.info(f"All results exported to: {output_dir.absolute()}")

    def get_performance_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive performance report

        Returns:
            Performance metrics dictionary
        """
        return {
            **self.performance_metrics,
            "average_processing_time_per_page": (
                self.performance_metrics["total_processing_time"]
                / max(1, self.performance_metrics["pages_processed"])
            ),
            "error_rate": (
                self.performance_metrics["errors_encountered"]
                / max(1, self.performance_metrics["pages_processed"])
            ),
            "timestamp": datetime.now().isoformat(),
        }


def create_omr_pipeline(
    config_path: Optional[Union[str, Path]] = None,
    preprocessor: Optional[AdvancedPreprocessor] = None,
    segmenter: Optional[SymbolSegmenter] = None,
    recognizer: Optional[SymbolRecognizer] = None,
    custom_config: Optional[Union[Dict[str, Any], OMRConfig]] = None,
) -> OMRPipeline:
    """Create and configure an OMR pipeline with the given components.

    This is a convenience function to create a pre-configured OMRPipeline
    instance with default components if none are provided.

    Args:
        config_path: Path to a configuration file
        preprocessor: Optional preprocessor instance
        segmenter: Optional symbol segmenter instance
        recognizer: Optional symbol recognizer instance
        custom_config: Optional configuration dictionary or OMRConfig instance

    Returns:
        Configured OMRPipeline instance
    """
    return OMRPipeline(
        config_path=config_path,
        preprocessor=preprocessor,
        segmenter=segmenter,
        recognizer=recognizer,
        custom_config=custom_config,
    )
