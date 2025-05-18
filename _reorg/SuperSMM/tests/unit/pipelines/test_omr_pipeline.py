import os
import pytest
import numpy as np
import tempfile
import cv2

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
print('DEBUG sys.path:', sys.path)

from core.omr_pipeline import OMRPipeline
from core.omr_exceptions import (PreprocessingError, SegmentationError, RecognitionError, ConfigurationError)

@pytest.fixture
def sample_image():
    """Create a sample image for testing"""
    image = np.zeros((500, 500, 3), dtype=np.uint8)
    cv2.rectangle(image, (100, 100), (400, 400), (255, 255, 255), -1)
    return image


@pytest.fixture
def temp_image_file(sample_image):
    """Create a temporary image file for testing"""
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
        cv2.imwrite(temp_file.name, sample_image)
        yield temp_file.name
    os.unlink(temp_file.name)


class TestOMRPipeline:
    def test_pipeline_initialization(self):
        """Test basic pipeline initialization"""
        pipeline = OMRPipeline()
        assert pipeline.preprocessor is not None
        assert pipeline.segmenter is not None
        assert pipeline.recognizer is not None

    def test_configuration_loading(self):
        """Test configuration loading from different sources"""
        # Test default configuration
        pipeline = OMRPipeline()
        assert 'preprocessing' in pipeline.config
        assert 'segmentation' in pipeline.config
        assert 'recognition' in pipeline.config

        # Test custom configuration
        custom_config = {
            'preprocessing': {'min_image_size': (200, 200)},
            'recognition': {'confidence_threshold': 0.7}
        }
        pipeline = OMRPipeline(custom_config=custom_config)
        assert pipeline.config['preprocessing']['min_image_size'] == (200, 200)
        assert pipeline.config['recognition']['confidence_threshold'] == 0.7

    def test_configuration_error_handling(self):
        """Test configuration error handling"""
        with pytest.raises(ConfigurationError):
            OMRPipeline(config_path='/nonexistent/path/config.yml')

    def test_process_sheet_music(self, temp_image_file):
        """Test complete sheet music processing"""
        pipeline = OMRPipeline()
        results = pipeline.process_sheet_music(temp_image_file)

        # Validate results structure
        assert 'preprocessing' in results
        assert 'segmentation' in results
        assert 'recognition' in results
        assert 'metadata' in results

    def test_debug_image_export(self, temp_image_file):
        """Test debug image export functionality"""
        pipeline = OMRPipeline()
        results = pipeline.process_sheet_music(temp_image_file)

        with tempfile.TemporaryDirectory() as temp_dir:
            pipeline.export_debug_images(results, output_dir=temp_dir)

            # Check if debug images are created
            debug_images = [
                'original.png', 'normalized.png',
                'binary.png', 'no_staff_lines.png'
            ]
            for img in debug_images:
                assert os.path.exists(os.path.join(temp_dir, img))

    def test_music_theory_analysis(self, temp_image_file):
        """Test music theory analysis generation"""
        pipeline = OMRPipeline()
        results = pipeline.process_sheet_music(temp_image_file)

        music_theory = pipeline.generate_music_theory_analysis(results)

        assert 'total_symbols' in music_theory
        assert 'symbol_distribution' in music_theory
        assert 'confidence_metrics' in music_theory
        assert 'warnings' in music_theory

    def test_error_handling(self):
        """Test various error scenarios"""
        pipeline = OMRPipeline()

        # Test invalid image path
        with pytest.raises(ValueError):
            pipeline.process_sheet_music('/nonexistent/path/image.png')

        # Test image size constraints
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
            # Create an oversized image
            large_image = np.zeros((10000, 10000, 3), dtype=np.uint8)
            cv2.imwrite(temp_file.name, large_image)

            with pytest.raises(ValueError):
                pipeline.process_sheet_music(temp_file.name)

            os.unlink(temp_file.name)


def test_pipeline_performance():
    """Basic performance test for the OMR pipeline"""
    import time

    pipeline = OMRPipeline()

    # Create a test image
    test_image = np.zeros((500, 500, 3), dtype=np.uint8)
    cv2.rectangle(test_image, (100, 100), (400, 400), (255, 255, 255), -1)

    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
        cv2.imwrite(temp_file.name, test_image)

        start_time = time.time()
        results = pipeline.process_sheet_music(temp_file.name)
        processing_time = time.time() - start_time

        os.unlink(temp_file.name)

    # Performance assertion (adjust threshold as needed)
    assert processing_time < 5.0, f"Processing took too long: {processing_time} seconds"
    assert results is not None, "Processing returned no results"
