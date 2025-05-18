from dataclasses import dataclass, field
from pathlib import Path
import cv2
import numpy as np
import logging
import time
from typing import Dict, Any, List, Tuple, Optional, Union, Callable, ClassVar
import fitz  # PyMuPDF

from .staff_detection_utils import detect_staff_lines
from .pdf_processor import extract_and_process_pdf_pages

from dataclasses import dataclass, field
from typing import ClassVar, Dict


@dataclass
class PreprocessorConfig:
    """Configuration for the AdvancedPreprocessor.

    Attributes:
        gaussian_blur_ksize: Kernel size for Gaussian blur, must be odd numbers
        adaptive_threshold_block_size: Block size for adaptive thresholding, must be odd
        adaptive_threshold_c: Constant subtracted from mean in adaptive threshold
        normalize_min_size: Minimum size for normalization
        normalize_max_size: Maximum size for normalization
        clahe_clip_limit: Clip limit for CLAHE histogram equalization
        clahe_grid_size: Grid size for CLAHE histogram equalization
        denoise_h: h parameter for non-local means denoising
        denoise_template_window_size: Template window size for denoising, must be odd
        save_intermediate_stages: Whether to save intermediate processing stages
        output_dir: Directory for intermediate stage outputs if enabled
    """

    # Processing parameters
    gaussian_blur_ksize: Tuple[int, int] = field(
        default=(5, 5),
        metadata={"validate": lambda x: all(k % 2 == 1 and k > 0 for k in x)},
    )
    adaptive_threshold_block_size: int = field(
        default=11, metadata={"validate": lambda x: x % 2 == 1 and x > 0}
    )
    adaptive_threshold_c: int = field(
        default=2, metadata={"validate": lambda x: x >= 0}
    )
    normalize_min_size: int = field(default=64, metadata={"validate": lambda x: x > 0})
    normalize_max_size: int = field(
        default=4096, metadata={"validate": lambda x: x > 64}
    )
    clahe_clip_limit: float = field(default=2.0, metadata={"validate": lambda x: x > 0})
    clahe_grid_size: Tuple[int, int] = field(
        default=(8, 8), metadata={"validate": lambda x: all(k > 0 for k in x)}
    )
    denoise_h: float = field(default=10.0, metadata={"validate": lambda x: x > 0})
    denoise_template_window_size: int = field(
        default=7, metadata={"validate": lambda x: x % 2 == 1 and x > 0}
    )

    # Output configuration
    save_intermediate_stages: bool = False
    output_dir: Optional[Path] = field(
        default=None, metadata={"validate": lambda x: x is None or x.is_dir()}
    )

    # Class-level constants
    REQUIRED_ODD_PARAMS: ClassVar[List[str]] = [
        "adaptive_threshold_block_size",
        "denoise_template_window_size",
    ]

    def __post_init__(self):
        """Validate configuration parameters after initialization."""
        self._validate_params()

    def _validate_params(self) -> None:
        """Validate all configuration parameters.

        Raises:
            ValueError: If any parameter fails validation
        """
        for field_name, field_def in self.__dataclass_fields__.items():
            if validate_fn := field_def.metadata.get("validate"):
                value = getattr(self, field_name)
                if value is not None and not validate_fn(value):
                    raise ValueError(
                        f"Invalid value for {field_name}: {value}. "
                        "Must satisfy validation requirements."
                    )

        # Validate normalize_min_size < normalize_max_size
        if self.normalize_min_size >= self.normalize_max_size:
            raise ValueError(
                f"normalize_min_size ({self.normalize_min_size}) must be less than "
                f"normalize_max_size ({self.normalize_max_size})"
            )

        # Validate output_dir if saving intermediate stages
        if self.save_intermediate_stages and self.output_dir is None:
            raise ValueError(
                "output_dir must be specified when save_intermediate_stages is True"
            )


class AdvancedPreprocessor:
    """Advanced image preprocessing for OMR.

    This class handles all image preprocessing steps including:
    - Gaussian blur for noise reduction
    - Adaptive thresholding for binarization
    - CLAHE for contrast enhancement
    - Non-local means denoising
    - Image normalization

    Attributes:
        config (PreprocessorConfig): Configuration for preprocessing parameters
        logger (logging.Logger): Logger instance for this class
    """

    def __init__(
        self, config: Optional[Union[Dict[str, Any], PreprocessorConfig]] = None
    ):
        """Initialize the AdvancedPreprocessor with configuration.

        Args:
            config: Either a PreprocessorConfig instance or a dictionary of config overrides.
                   If a dictionary is provided, it will be used to override default values.

        Raises:
            ValueError: If config validation fails
        """
        self.logger = logging.getLogger(__name__)

        # Handle config initialization
        if isinstance(config, PreprocessorConfig):
            self.config = config
        elif isinstance(config, dict):
            try:
                default_config = PreprocessorConfig()
                # Create a new config with overrides
                config_dict = {k: v for k, v in default_config.__dict__.items()}
                for key, value in config.items():
                    if hasattr(default_config, key):
                        config_dict[key] = value
                self.config = PreprocessorConfig(**config_dict)
            except Exception as e:
                self.logger.error(f"Failed to create config from dict: {e}")
                raise
        else:
            self.config = PreprocessorConfig()

        # Log initialization
        self.logger.debug(
            "AdvancedPreprocessor initialized with config: %s",
            {k: v for k, v in self.config.__dict__.items() if not k.startswith("_")},
        )

        # Initialize CLAHE for contrast enhancement
        self.clahe = cv2.createCLAHE(
            clipLimit=self.config.clahe_clip_limit,
            tileGridSize=self.config.clahe_grid_size,
        )

        # Initialize performance monitoring
        self.perf_logger = logging.getLogger(f"{__name__}.performance")

        # Create output directory if needed
        if self.config.save_intermediate_stages and self.config.output_dir:
            self.config.output_dir.mkdir(parents=True, exist_ok=True)

    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        """Normalize an image by resizing, converting to grayscale, and enhancing contrast.

        Args:
            image: Input image as numpy array (2D grayscale or 3D color)

        Returns:
            Normalized grayscale image as numpy array

        Raises:
            ValueError: If image is None or has invalid dimensions
            TypeError: If image is not a numpy array
        """
        # Input validation
        if image is None:
            self.logger.error("Input image to normalize is None")
            raise ValueError("Input image cannot be None")
        if not isinstance(image, np.ndarray):
            self.logger.error(f"Invalid input type for normalization: {type(image)}")
            raise TypeError(f"Expected numpy.ndarray, got {type(image)}")
        if len(image.shape) < 2 or len(image.shape) > 3:
            self.logger.error(
                f"Invalid image dimensions: {image.shape}. Expected 2D or 3D."
            )
            raise ValueError(f"Invalid image dimensions: {image.shape}")

        # Start performance tracking
        start_time = time.time()

        try:
            # Resize if necessary
            h, w = image.shape[:2]
            if not (
                self.config.normalize_min_size <= h <= self.config.normalize_max_size
                and self.config.normalize_min_size
                <= w
                <= self.config.normalize_max_size
            ):
                self.logger.info(
                    f"Image size {(h, w)} outside range ({self.config.normalize_min_size}-"
                    f"{self.config.normalize_max_size}). Resizing."
                )
                if (
                    h > self.config.normalize_max_size
                    or w > self.config.normalize_max_size
                ):
                    scale_factor = min(
                        self.config.normalize_max_size / h,
                        self.config.normalize_max_size / w,
                    )
                    image = cv2.resize(
                        image,
                        None,
                        fx=scale_factor,
                        fy=scale_factor,
                        interpolation=cv2.INTER_AREA,
                    )

            # Convert to grayscale if needed
            if len(image.shape) == 3 and image.shape[2] > 1:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Apply contrast enhancement
            image = self.clahe.apply(image)

            # Apply denoising if configured
            image = cv2.fastNlMeansDenoising(
                image,
                h=self.config.denoise_h,
                templateWindowSize=self.config.denoise_template_window_size,
            )

            # Save intermediate result if configured
            if self.config.save_intermediate_stages and self.config.output_dir:
                cv2.imwrite(str(self.config.output_dir / "normalized_image.png"), image)

            return image

        except Exception as e:
            self.logger.error(f"Error during image normalization: {str(e)}")
            raise
        finally:
            # Log performance metrics
            duration = time.time() - start_time
            self.perf_logger.debug(
                "normalize_image completed in %.3fs. Input shape: %s, Output shape: %s",
                duration,
                image.shape[:2],
                image.shape[:2],
            )

        if len(image.shape) == 2:
            gray_image = image.copy()  # Work on a copy
        else:  # E.g. 3-channel but last channel is 1, or other unexpected formats
            self.logger.warning(
                f"Image has unexpected shape {image.shape}. Attempting grayscale conversion."
            )
            try:
                gray_image = cv2.cvtColor(
                    image,
                    cv2.COLOR_BGRA2GRAY if image.shape[2] == 4 else cv2.COLOR_BGR2GRAY,
                )
            except cv2.error as e:
                self.logger.error(
                    f"Could not convert image of shape {image.shape} to grayscale: {e}"
                )
                raise ValueError(f"Could not convert image to grayscale: {e}")

        if gray_image is None or gray_image.size == 0:
            raise ValueError("Grayscale conversion failed or resulted in empty image")

        clahe = cv2.createCLAHE(
            clipLimit=self.clahe_clip_limit, tileGridSize=self.clahe_grid_size
        )
        equalized = clahe.apply(gray_image)

        denoised = cv2.fastNlMeansDenoising(
            equalized,
            None,
            h=self.denoise_h,
            templateWindowSize=self.denoise_template_window_size,
            searchWindowSize=21,
        )

        if denoised is None or denoised.size == 0:
            raise ValueError("Image normalization failed at denoising stage")

        self.logger.info(
            f"Image normalized. Original input shape: {image.shape}, Output shape: {denoised.shape}"
        )
        return denoised

    def binarize_image(self, image: np.ndarray) -> np.ndarray:
        if image is None:
            self.logger.error("Input image to binarize is None.")
            raise ValueError("Input image for binarization cannot be None.")
        if len(image.shape) != 2:
            self.logger.error(
                f"Binarization expects a 2D grayscale image, got shape: {image.shape}"
            )
            if (
                len(image.shape) == 3 and image.shape[2] == 1
            ):  # Grayscale but with a redundant channel
                image = image.reshape(image.shape[0], image.shape[1])
            elif len(image.shape) == 3 and image.shape[2] > 1:
                self.logger.warning(
                    f"Binarize input is not grayscale ({image.shape}), attempting conversion from BGR."
                )
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                raise ValueError(
                    "Binarization requires a 2D grayscale image or one convertible to it."
                )

        binary_adaptive = cv2.adaptiveThreshold(
            image,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            self.adaptive_threshold_block_size,
            self.adaptive_threshold_c,
        )
        return binary_adaptive

    def remove_staff_lines(
        self, binary_image: np.ndarray, staff_lines_coords: Optional[np.ndarray] = None
    ) -> np.ndarray:
        # This method now primarily relies on HMM. `staff_lines_coords` is for fallback or alternative methods.
        if binary_image is None:
            self.logger.error("Input binary_image to remove_staff_lines is None.")
            return None  # Or raise error

        img_no_staffs = None
        try:
            from src.preprocessing.hmm_staff_removal import hmm_remove_staff_lines

            self.logger.info("Attempting HMM staff line removal.")
            # Ensure binary_image is a copy if hmm_remove_staff_lines modifies inplace or if we need original later
            img_no_staffs = hmm_remove_staff_lines(
                binary_image.copy(), logger=self.logger
            )
            if img_no_staffs is not None:
                self.logger.info("HMM staff line removal successful.")
                return img_no_staffs
            else:
                self.logger.warning(
                    "HMM staff removal returned None. Will try fallback if lines provided."
                )
        except ImportError:
            self.logger.warning(
                "hmm_remove_staff_lines module not found. Staff lines will not be removed by HMM."
            )
        except Exception as e:
            self.logger.error(f"Error during HMM staff removal: {e}", exc_info=True)
            self.logger.warning("Falling back due to HMM error.")

        # Fallback: Simple geometric removal if HMM failed/unavailable AND staff_lines_coords are provided
        if staff_lines_coords is not None and staff_lines_coords.size > 0:
            self.logger.info(
                f"Falling back to simple geometric staff line removal using {len(staff_lines_coords)} detected lines."
            )
            img_copy = binary_image.copy()
            # staff_lines_coords is expected to be (N,4) with (x1, y, x2, y)
            for (
                x1,
                y_coord,
                x2,
                _,
            ) in staff_lines_coords:  # y1 and y2 are same (y_coord)
                # Erase a few pixels around the detected y. Typical staff line thickness is 1-3 pixels.
                # Erasing with thickness 3 should cover it. (line itself, 1px above, 1px below)
                cv2.line(
                    img_copy, (x1, y_coord), (x2, y_coord), 0, thickness=3
                )  # 0 for black (erase in binary_inv)
            self.logger.info("Simple geometric staff line removal applied as fallback.")
            return img_copy

        self.logger.warning(
            "No staff line removal method successfully applied or no lines for fallback. Returning original binary image."
        )
        return binary_image.copy()  # Return a copy of original binary if all fails

    def process_pdf(
        self,
        pdf_path: Union[str, Path],
        output_dir_base: Optional[Union[str, Path]] = None,
        page_range: Union[int, Tuple[int, int], None] = None,
    ) -> List[Dict[str, Any]]:
        """
        Processes a PDF document, extracting images from specified pages and processing them.

        This method delegates the core PDF handling and page iteration to
        `extract_and_process_pdf_pages` from the `pdf_processor` module.

        Args:
            pdf_path (str): The path to the PDF file.
            output_dir_base (Optional[str]): Base directory for saving processed images.
                                             A subdirectory named after the PDF will be created here.
                                             If None, images are not saved to disk from this method's context.
            page_range (Union[int, Tuple[int, int], None]):
                - If int: Process only that specific page (0-indexed).
                - If tuple (start, end): Process pages in that range (inclusive, 0-indexed).
                - If None: Process all pages.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, where each dictionary contains
                                  the processing results for a page, including any errors.
        """
        # Convert paths to Path objects
        pdf_path = Path(pdf_path)
        if output_dir_base:
            output_dir_base = Path(output_dir_base)

        self.logger.info(
            f"Processing PDF: {pdf_path.name} with output_dir: {output_dir_base} and page_range: {page_range}"
        )

        return extract_and_process_pdf_pages(
            pdf_path=pdf_path,
            process_page_func=self.process_page,
            logger_instance=self.logger,
            output_dir_base=output_dir_base,
            page_range=page_range,
        )

    def _save_intermediate_image(
        self,
        image: np.ndarray,
        output_dir: Optional[Union[str, Path]],
        stage_filename_suffix: str,
        page_identifier: str,
        results_key_path: str,
        results: Dict[str, Any],
    ) -> None:
        """Save intermediate processing stage image if enabled in config.

        Args:
            image: Image array to save
            output_dir: Directory to save the image in
            stage_filename_suffix: Suffix for the output filename
            page_identifier: Identifier for the current page
            results_key_path: Key path in results dict to store the saved path
            results: Results dictionary to update with saved path
        """
        if output_dir and self.config["save_intermediate_stages"]:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            # Create filename with stage info
            stage_name = stage_filename_suffix.split(".")[0].replace("_", " ")
            save_path = output_dir / f"{page_identifier}_{stage_name}.png"

            try:
                cv2.imwrite(str(save_path), image)
                results[results_key_path] = str(save_path)
                self.logger.info(
                    f"[{page_identifier}] Saved {stage_name} to {save_path.name}"
                )
            except Exception as e:
                self.logger.error(
                    f"[{page_identifier}] Failed to save {stage_name}: {e}",
                    exc_info=True,
                )

    def _load_and_validate_image_stage(
        self,
        image_input: Union[str, np.ndarray],
        results: Dict[str, Any],
        page_identifier: str,
        output_dir: Optional[str],
    ) -> Optional[np.ndarray]:
        self.logger.debug(f"[{page_identifier}] Stage 1: Loading image input.")
        current_image_stage: Optional[np.ndarray] = None
        try:
            if isinstance(image_input, str):
                if not os.path.exists(image_input):
                    raise FileNotFoundError(f"Image path does not exist: {image_input}")
                current_image_stage = cv2.imread(image_input)
                if current_image_stage is None:
                    raise ValueError(f"cv2.imread failed for {image_input}")
            elif isinstance(image_input, np.ndarray):
                current_image_stage = image_input.copy()
            else:
                raise TypeError(
                    f"Invalid image_input type: {type(image_input)}. Expected str or np.ndarray."
                )

            results["original_image"] = current_image_stage.copy()
            self._save_intermediate_image(
                current_image_stage,
                output_dir,
                "_00_original.png",
                page_identifier,
                "original_image_path",
                results,
            )
            return current_image_stage
        except Exception as e:
            self.logger.error(
                f"[{page_identifier}] Error during image loading: {e}", exc_info=True
            )
            results.update({"error": str(e), "error_stage": "loading"})
            return None

    def _normalize_image_stage(
        self,
        current_image_stage: Optional[np.ndarray],
        results: Dict[str, Any],
        page_identifier: str,
        output_dir: Optional[str],
    ) -> Optional[np.ndarray]:
        if current_image_stage is None:
            return None
        self.logger.debug(f"[{page_identifier}] Stage 2: Normalizing image.")
        try:
            normalized_img = self.normalize_image(current_image_stage)
            results["normalized_image"] = normalized_img.copy()
            self._save_intermediate_image(
                normalized_img,
                output_dir,
                "_01_normalized.png",
                page_identifier,
                "normalized_image_path",
                results,
            )
            return normalized_img
        except Exception as e:
            self.logger.error(
                f"[{page_identifier}] Error during normalization: {e}", exc_info=True
            )
            results.update({"error": str(e), "error_stage": "normalization"})
            return current_image_stage  # Return original if normalization fails but we want to proceed

    def _binarize_image_stage(
        self,
        current_image_stage: Optional[np.ndarray],
        results: Dict[str, Any],
        page_identifier: str,
        output_dir: Optional[str],
    ) -> Optional[np.ndarray]:
        if current_image_stage is None:
            return None
        self.logger.debug(f"[{page_identifier}] Stage 3: Binarizing image.")
        try:
            binary_img = self.binarize_image(current_image_stage)
            results["binary_image"] = binary_img.copy()
            self._save_intermediate_image(
                binary_img,
                output_dir,
                "_02_binary.png",
                page_identifier,
                "binary_image_path",
                results,
            )
            return binary_img
        except Exception as e:
            self.logger.error(
                f"[{page_identifier}] Error during binarization: {e}", exc_info=True
            )
            results.update({"error": str(e), "error_stage": "binarization"})
            return current_image_stage  # Return previous stage if binarization fails

    def _process_staffs_stage(
        self,
        binary_image_for_staffs: Optional[np.ndarray],
        results: Dict[str, Any],
        page_identifier: str,
        output_dir: Optional[str],
    ):
        if binary_image_for_staffs is None:
            self.logger.warning(
                f"[{page_identifier}] Skipping staff line detection and removal as binary image is not available."
            )
            results["image_without_staffs"] = (
                None  # Or a copy of original if that's preferred fallback
            )
            return

        self.logger.debug(f"[{page_identifier}] Stage 4: Processing staff lines.")
        try:
            self.logger.debug(f"[{page_identifier}] Stage 4a: Detecting staff lines.")
            detected_lines_coords = detect_staff_lines(
                binary_image_for_staffs, logger_instance=self.logger
            )
            results["staff_lines_detected"] = detected_lines_coords

            if detected_lines_coords is not None and detected_lines_coords.size > 0:
                self.logger.info(
                    f"[{page_identifier}] Detected {len(detected_lines_coords)} staff line segments."
                )
                if (
                    output_dir and self.save_intermediate_stages
                ):  # Direct check as _save_intermediate_image is for single images
                    img_with_lines_drawn = cv2.cvtColor(
                        binary_image_for_staffs, cv2.COLOR_GRAY2BGR
                    )
                    for x1, y_val, x2, _ in detected_lines_coords:
                        cv2.line(
                            img_with_lines_drawn,
                            (x1, y_val),
                            (x2, y_val),
                            (0, 255, 0),
                            1,
                        )
                    self._save_intermediate_image(
                        img_with_lines_drawn,
                        output_dir,
                        f"_02a_binary_with_lines.png",
                        page_identifier,
                        "binary_with_lines_path",
                        results,
                    )
            else:
                self.logger.info(
                    f"[{page_identifier}] No staff lines detected or detection returned empty."
                )

            self.logger.debug(f"[{page_identifier}] Stage 4b: Removing staff lines.")
            img_no_staffs = self.remove_staff_lines(
                binary_image_for_staffs, detected_lines_coords
            )

            if img_no_staffs is not None:
                results["image_without_staffs"] = img_no_staffs.copy()
                self._save_intermediate_image(
                    img_no_staffs,
                    output_dir,
                    "_03_no_staffs.png",
                    page_identifier,
                    "image_without_staffs_path",
                    results,
                )
            else:
                self.logger.warning(
                    f"[{page_identifier}] Staff line removal returned None. Using binary image as fallback."
                )
                results["image_without_staffs"] = (
                    binary_image_for_staffs.copy()
                )  # Fallback
                self._save_intermediate_image(
                    binary_image_for_staffs,
                    output_dir,
                    "_03_no_staffs_fallback.png",
                    page_identifier,
                    "image_without_staffs_path",
                    results,
                )

        except Exception as e:
            self.logger.error(
                f"[{page_identifier}] Error during staff line detection or removal: {e}",
                exc_info=True,
            )
            results.update({"error": str(e), "error_stage": "staff_processing"})
            if (
                results.get("image_without_staffs") is None
            ):  # Ensure fallback if error occurred mid-stage
                results["image_without_staffs"] = binary_image_for_staffs.copy()
                self._save_intermediate_image(
                    binary_image_for_staffs,
                    output_dir,
                    "_03_no_staffs_fallback_error.png",
                    page_identifier,
                    "image_without_staffs_path",
                    results,
                )

    def process_page(
        self,
        image_input: Union[str, np.ndarray],
        output_dir: Optional[str] = None,
        page_identifier: str = "page",
    ) -> Dict[str, Any]:
        self.logger.info(f"Starting processing for page: {page_identifier}")
        results: Dict[str, Any] = {
            "original_image_path": None,
            "original_image": None,
            "normalized_image_path": None,
            "normalized_image": None,
            "binary_image_path": None,
            "binary_image": None,
            "staff_lines_detected": None,
            "binary_with_lines_path": None,  # For the image with staff lines drawn
            "image_without_staffs_path": None,
            "image_without_staffs": None,
            "error": None,
            "error_stage": None,
            "page_identifier": page_identifier,
        }

        # Stage 1: Load and Validate Image
        current_image_stage = self._load_and_validate_image_stage(
            image_input, results, page_identifier, output_dir
        )
        if current_image_stage is None:
            self.logger.error(
                f"[{page_identifier}] Critical error during image loading. Aborting further processing for this page."
            )
            return results

        # Stage 2: Normalize Image
        current_image_stage = self._normalize_image_stage(
            current_image_stage, results, page_identifier, output_dir
        )
        # Normalization helper returns original image if it fails, so current_image_stage should still be valid if not None initially.

        # Stage 3: Binarize Image
        current_image_stage = self._binarize_image_stage(
            current_image_stage, results, page_identifier, output_dir
        )
        # Binarization helper returns previous stage image if it fails.

        # Stage 4: Staff Line Detection & Removal
        # This stage specifically needs the 'binary_image' from results.
        binary_image_for_staffs = results.get("binary_image")  # Use .get() for safety
        self._process_staffs_stage(
            binary_image_for_staffs, results, page_identifier, output_dir
        )

        self.logger.info(
            f"Finished processing for page: {page_identifier}. Final error status: {results.get('error')}"
        )
        return results
