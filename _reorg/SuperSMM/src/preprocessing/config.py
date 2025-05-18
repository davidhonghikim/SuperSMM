"""Configuration management for the preprocessing module.

This module contains the configuration dataclass and validation logic for
preprocessing parameters.
"""

from dataclasses import dataclass, field
from pathlib import Path
import logging
from typing import ClassVar, Dict, List, Tuple, Optional, Union

logger = logging.getLogger(__name__)


@dataclass
class PreprocessorConfig:
    """Configuration for image preprocessing parameters.

    This class manages all configuration parameters for the AdvancedPreprocessor,
    including validation and type checking.

    Attributes:
        normalize_min_size (int): Minimum size for normalized images
        normalize_max_size (int): Maximum size for normalized images
        clahe_clip_limit (float): Clip limit for CLAHE
        clahe_grid_size (Tuple[int, int]): Grid size for CLAHE
        denoise_h (float): h parameter for denoising
        denoise_template_window_size (int): Window size for denoising
        save_intermediate_stages (bool): Whether to save intermediate processing results
        output_dir (Optional[Path]): Directory for intermediate results if enabled
    """

    # Required parameters with defaults
    normalize_min_size: int = field(default=800)
    normalize_max_size: int = field(default=1200)
    clahe_clip_limit: float = field(default=2.0)
    clahe_grid_size: Tuple[int, int] = field(default=(8, 8))
    denoise_h: float = field(default=10.0)
    denoise_template_window_size: int = field(default=7)

    # Optional parameters
    save_intermediate_stages: bool = field(default=False)
    output_dir: Optional[Path] = field(default=None)

    # Class-level constants
    REQUIRED_ODD_PARAMS: ClassVar[List[str]] = ["denoise_template_window_size"]

    def __post_init__(self):
        """Validate configuration parameters after initialization."""
        self._validate_params()

    def _validate_params(self) -> None:
        """Validate all configuration parameters.

        Raises:
            ValueError: If any parameter fails validation
        """
        # Validate numeric ranges
        if self.normalize_min_size >= self.normalize_max_size:
            raise ValueError(
                f"normalize_min_size ({self.normalize_min_size}) must be less than "
                f"normalize_max_size ({self.normalize_max_size})"
            )

        if self.clahe_clip_limit <= 0:
            raise ValueError(
                f"clahe_clip_limit must be positive, got {self.clahe_clip_limit}"
            )

        if any(x <= 0 for x in self.clahe_grid_size):
            raise ValueError(
                f"clahe_grid_size values must be positive, got {self.clahe_grid_size}"
            )

        if self.denoise_h <= 0:
            raise ValueError(f"denoise_h must be positive, got {self.denoise_h}")

        # Validate odd-numbered parameters
        for param_name in self.REQUIRED_ODD_PARAMS:
            value = getattr(self, param_name)
            if value % 2 == 0:
                logger.warning(
                    f"{param_name} must be odd, got {value}. "
                    f"Automatically adjusting to {value + 1}"
                )
                setattr(self, param_name, value + 1)

        # Validate output directory if intermediate stages are enabled
        if self.save_intermediate_stages and self.output_dir is None:
            self.output_dir = Path(
                "/ml/models/resources/tf-deep-omr/intermediate_results"
            )
            logger.info(f"Setting default output directory to: {self.output_dir}")
