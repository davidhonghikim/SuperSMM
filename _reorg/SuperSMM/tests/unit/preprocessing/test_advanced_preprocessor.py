import pytest
import numpy as np
import cv2
from src.preprocessing.advanced_preprocessor import AdvancedPreprocessor


@pytest.fixture
def sample_image():
    """Create a sample grayscale image for testing"""
    return np.random.randint(0, 256, (100, 100), dtype=np.uint8)


def test_advanced_preprocessor_initialization():
    """Test initialization of AdvancedPreprocessor"""
    preprocessor = AdvancedPreprocessor()
    assert preprocessor is not None
    assert preprocessor.logger is not None


def test_normalize_image(sample_image):
    """Test image normalization"""
    preprocessor = AdvancedPreprocessor()
    normalized = preprocessor.normalize_image(sample_image)

    assert normalized is not None
    assert normalized.shape == sample_image.shape
    assert normalized.dtype == sample_image.dtype


def test_binarize_image(sample_image):
    """Test image binarization"""
    preprocessor = AdvancedPreprocessor()
    binarized = preprocessor.binarize_image(sample_image)

    assert binarized is not None
    assert binarized.shape == sample_image.shape
    assert np.array_equal(
        binarized,
        binarized.astype(bool).astype(
            np.uint8) * 255)


def test_detect_staff_lines(sample_image):
    """Test staff line detection"""
    preprocessor = AdvancedPreprocessor()
    binarized = preprocessor.binarize_image(sample_image)
    staff_lines = preprocessor.detect_staff_lines(binarized)

    import numpy as np
    assert isinstance(staff_lines, (list, np.ndarray))
    # Staff lines might be empty depending on the random image
    for line in staff_lines:
        assert isinstance(line, tuple)
        assert len(line) == 2


def test_remove_staff_lines(sample_image):
    """Test staff line removal"""
    preprocessor = AdvancedPreprocessor()
    binarized = preprocessor.binarize_image(sample_image)
    staff_lines = preprocessor.detect_staff_lines(binarized)

    no_staff_lines = preprocessor.remove_staff_lines(binarized, staff_lines)

    assert no_staff_lines is not None
    assert no_staff_lines.shape == binarized.shape


def test_process_page(sample_image):
    """Test complete preprocessing pipeline"""
    preprocessor = AdvancedPreprocessor()
    result = preprocessor.process_page(sample_image)

    assert isinstance(result, dict)
    assert set(
        result.keys()) == {
        'original',
        'normalized',
        'binary',
        'no_staff_lines',
        'staff_lines'}

    for key, value in result.items():
        assert value is not None
        if key != 'staff_lines':
            assert value.shape == sample_image.shape


def test_color_image_normalization():
    """Test normalization of a color image"""
    color_image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    preprocessor = AdvancedPreprocessor()
    normalized = preprocessor.normalize_image(color_image)

    assert normalized is not None
    assert normalized.shape == color_image.shape[:2]
    assert normalized.dtype == np.uint8
