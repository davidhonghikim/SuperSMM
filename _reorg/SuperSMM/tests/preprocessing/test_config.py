"""Unit tests for preprocessing configuration."""

import pytest
from pathlib import Path
from src.preprocessing import PreprocessorConfig

def test_default_config():
    """Test default configuration initialization."""
    config = PreprocessorConfig()
    assert config.normalize_min_size == 800
    assert config.normalize_max_size == 1200
    assert config.clahe_clip_limit == 2.0
    assert config.clahe_grid_size == (8, 8)
    assert config.denoise_h == 10.0
    assert config.denoise_template_window_size == 7
    assert not config.save_intermediate_stages
    assert config.output_dir is None

def test_custom_config():
    """Test custom configuration initialization."""
    config = PreprocessorConfig(
        normalize_min_size=1000,
        normalize_max_size=2000,
        clahe_clip_limit=3.0,
        clahe_grid_size=(16, 16),
        denoise_h=15.0,
        denoise_template_window_size=9,
        save_intermediate_stages=True,
        output_dir=Path("/tmp/test")
    )
    assert config.normalize_min_size == 1000
    assert config.normalize_max_size == 2000
    assert config.clahe_clip_limit == 3.0
    assert config.clahe_grid_size == (16, 16)
    assert config.denoise_h == 15.0
    assert config.denoise_template_window_size == 9
    assert config.save_intermediate_stages
    assert config.output_dir == Path("/tmp/test")

def test_invalid_size_range():
    """Test validation of size range."""
    with pytest.raises(ValueError):
        PreprocessorConfig(
            normalize_min_size=1000,
            normalize_max_size=500  # Invalid: min > max
        )

def test_invalid_clahe_params():
    """Test validation of CLAHE parameters."""
    with pytest.raises(ValueError):
        PreprocessorConfig(clahe_clip_limit=-1.0)
    
    with pytest.raises(ValueError):
        PreprocessorConfig(clahe_grid_size=(0, 8))

def test_invalid_denoise_params():
    """Test validation of denoising parameters."""
    with pytest.raises(ValueError):
        PreprocessorConfig(denoise_h=-5.0)

def test_even_window_size_adjustment():
    """Test automatic adjustment of even window sizes to odd."""
    config = PreprocessorConfig(denoise_template_window_size=8)
    assert config.denoise_template_window_size == 9  # Should be adjusted to odd

def test_output_dir_validation():
    """Test validation of output directory configuration."""
    with pytest.raises(ValueError):
        PreprocessorConfig(
            save_intermediate_stages=True,
            output_dir=None  # Invalid: output_dir required when saving stages
        )
