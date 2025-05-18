import os
import pytest
import numpy as np
import cv2

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))
from core.omr_pipeline import OMRPipeline
from core.omr_exceptions import PreprocessingError, SegmentationError, RecognitionError


class TestOMRPipeline:
    def test_pipeline_initialization(self, omr_pipeline):
        """Test OMR Pipeline initialization"""
        assert omr_pipeline is not None
        assert hasattr(omr_pipeline, 'preprocessor')
        assert hasattr(omr_pipeline, 'segmenter')
        assert hasattr(omr_pipeline, 'recognizer')

    def test_process_sheet_music_with_sample_image(
            self, omr_pipeline, sample_image, tmp_path):
        """Test processing a sample sheet music image"""
        # Save sample image to a temporary file
        test_image_path = os.path.join(tmp_path, 'test_sheet_music.png')
        cv2.imwrite(test_image_path, sample_image)

        # Process the image
        results = omr_pipeline.process_sheet_music(test_image_path)

        # Validate results
        assert 'preprocessing' in results
        assert 'segmentation' in results
        assert 'recognition' in results
        assert 'metadata' in results

    def test_export_results(self, omr_pipeline, sample_image, tmp_path):
        """Test exporting processing results"""
        # Save sample image to a temporary file
        test_image_path = os.path.join(tmp_path, 'test_sheet_music.png')
        cv2.imwrite(test_image_path, sample_image)

        # Process the image
        results = omr_pipeline.process_sheet_music(test_image_path)

        # Create output directory
        output_dir = os.path.join(tmp_path, 'output')
        os.makedirs(output_dir, exist_ok=True)

        # Export results
        omr_pipeline.export_results(results, output_dir)

        # Check exported files
        assert os.path.exists(os.path.join(output_dir, 'preprocessed.png'))
        assert os.path.exists(os.path.join(output_dir, 'segmentation.png'))
        assert os.path.exists(
            os.path.join(
                output_dir,
                'recognition_results.json'))

    def test_error_handling(self, omr_pipeline):
        """Test error handling for non-existent image"""
        with pytest.raises(FileNotFoundError):
            omr_pipeline.process_sheet_music('/path/to/non_existent_image.png')
