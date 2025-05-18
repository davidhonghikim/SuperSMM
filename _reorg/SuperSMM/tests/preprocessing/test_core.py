"""Unit tests for core preprocessing functionality."""

import pytest
import numpy as np
from pathlib import Path
from src.preprocessing import AdvancedPreprocessor, PreprocessorConfig

@pytest.fixture
def sample_image():
    """Create a sample test image."""
    return np.random.randint(0, 255, (1000, 800), dtype=np.uint8)

@pytest.fixture
def color_image():
    """Create a sample color test image."""
    return np.random.randint(0, 255, (1000, 800, 3), dtype=np.uint8)

@pytest.fixture
def temp_output_dir(tmp_path):
    """Create a temporary output directory."""
    output_dir = tmp_path / "test_output"
    output_dir.mkdir()
    return output_dir

def test_normalizer_init(sample_image):
    """Test normalizer initialization."""
    normalizer = ImageNormalizer(PreprocessorConfig())
    assert normalizer.config is not None
    assert normalizer.logger is not None

def test_normalizer_custom_config(sample_image):
    """Test normalizer with custom config."""
    config = PreprocessorConfig(
        normalize_min_size=300,
        normalize_max_size=400
    )
    normalizer = ImageNormalizer(config)
    assert normalizer.config.normalize_min_size == 300
    assert normalizer.config.normalize_max_size == 400

def test_normalize_image_size(sample_image):
    """Test image size normalization."""
    normalizer = ImageNormalizer(PreprocessorConfig(
        normalize_min_size=500,
        normalize_max_size=600
    ))

    # Get original aspect ratio
    orig_h, orig_w = sample_image.shape[:2]
    orig_aspect = orig_w / orig_h

    # Process image
    result = normalizer.normalize_image(sample_image)
    h, w = result.shape[:2]

    # Check size constraints
    assert 500 <= h <= 600 and 500 <= w <= 600, \
        f"Image dimensions ({h}, {w}) outside allowed range (500-600)"

    # Check aspect ratio is preserved (within 1% tolerance)
    new_aspect = w / h
    assert abs(new_aspect - orig_aspect) / orig_aspect < 0.01, \
        f"Aspect ratio not preserved: {orig_aspect:.2f} -> {new_aspect:.2f}"

def test_normalize_color_to_gray(color_image):
    """Test color to grayscale conversion."""
    normalizer = ImageNormalizer(PreprocessorConfig())
    result = normalizer.normalize_image(color_image)
    assert len(result.shape) == 2, \
        f"Expected grayscale image (2 dims), got {len(result.shape)} dims"

def test_save_intermediate_results(sample_image, temp_output_dir):
    """Test saving of intermediate processing results."""
    config = PreprocessorConfig(
        save_intermediate_stages=True,
        output_dir=temp_output_dir
    )
    processor = PageProcessor(config)
    processor.process_page(sample_image)
    
    # Check that intermediate files were created
    assert (temp_output_dir / "normalized_image.png").exists()
    assert (temp_output_dir / "binary_image.png").exists()

def test_process_page_output_format(sample_image):
    """Test process_page output format."""
    processor = PageProcessor(PreprocessorConfig())
    result = processor.process_page(sample_image)
    
    assert isinstance(result, dict), "Result should be a dictionary"
    assert 'normalized' in result, "Missing 'normalized' key in result"
    
    # Check image formats
    assert isinstance(result["normalized"], np.ndarray)
    assert isinstance(result["binary"], np.ndarray)
    assert len(result["normalized"].shape) == 2  # Grayscale
    assert len(result["binary"].shape) == 2  # Binary

def test_invalid_inputs(sample_image):
    """Test handling of invalid inputs."""
    normalizer = ImageNormalizer(PreprocessorConfig())
    
    # Test with None
    with pytest.raises(ValueError):
        normalizer.normalize_image(None)
        
    # Test with empty array
    with pytest.raises(ValueError):
        normalizer.normalize_image(np.array([]))
        
    # Test with wrong type
    with pytest.raises(ValueError):
        normalizer.normalize_image([1, 2, 3])
    
    # Wrong dimensions
    with pytest.raises(ValueError, match="dimensions"):
        normalizer.normalize_image(np.zeros((10,)))  # 1D array
    with pytest.raises(ValueError, match="dimensions"):
        processor.normalize_image(np.zeros((10, 10, 10, 10)))  # 4D array
