from dataclasses import dataclass, field
from typing import Tuple, Dict, Any
from pathlib import Path


@dataclass
class PreprocessingConfig:
    gaussian_blur_ksize: Tuple[int, int] = (5, 5)
    adaptive_threshold_block_size: int = 11
    adaptive_threshold_c: int = 2
    normalize_min_size: int = 64
    normalize_max_size: int = 4096
    clahe_clip_limit: float = 2.0
    clahe_grid_size: Tuple[int, int] = (8, 8)
    denoise_h: float = 10.0
    denoise_template_window_size: int = 7
    save_intermediate_stages: bool = False
    output_dir: str = "preprocessing_output"


@dataclass
class SegmentationConfig:
    min_symbol_size: int = 10
    max_symbol_size: int = 500
    staff_line_spacing: int = 10
    staff_line_thickness: int = 2


@dataclass
class RecognitionConfig:
    confidence_threshold: float = 0.5
    model_path: str = (
        "/ml/models/resources/tf-deep-omr/resources/ml_models/symbol_recognition.h5"
    )
    batch_size: int = 32


@dataclass
class LoggingConfig:
    level: str = "INFO"
    file: str = "logs/omr_pipeline.log"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


@dataclass
class OMRConfig:
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    recognition: RecognitionConfig = field(default_factory=RecognitionConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    output_dir: str = "output"
    cache_dir: str = ".cache"
    debug_mode: bool = False
