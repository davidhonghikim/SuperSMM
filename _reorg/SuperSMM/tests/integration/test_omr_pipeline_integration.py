import os
import pytest
import numpy as np
import cv2

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))
from core.omr_pipeline import OMRPipeline
from preprocessing.advanced_preprocessor import AdvancedPreprocessor
from segmentation.symbol_segmenter import SymbolSegmenter
from recognition.symbol_recognizer import SymbolRecognizer


@pytest.mark.integration
class TestOMRPipelineIntegration:
    def test_full_pipeline_integration(self, sample_image, tmp_path):
        """Test full OMR pipeline integration"""
        # Save sample image
        test_image_path = os.path.join(tmp_path, 'test_sheet_music.png')
        cv2.imwrite(test_image_path, sample_image)

        # Initialize pipeline with custom configurations
        custom_segmenter_config = {
            'min_symbol_size': 2,
            'max_symbol_size': 750,
            'segmentation_strategy': 'connected_components'
        }
        pipeline = OMRPipeline(
            preprocessor=AdvancedPreprocessor(),
            segmenter=SymbolSegmenter(config=custom_segmenter_config),
            recognizer=SymbolRecognizer()
        )

        # Process sheet music
        results = pipeline.process_sheet_music(test_image_path)

        # Validate integration results
        assert 'preprocessing' in results
        assert 'segmentation' in results
        assert 'recognition' in results

        # Check preprocessing output
        preprocessed_dict = results['preprocessing']
        assert preprocessed_dict is not None
        assert 'binary_image' in preprocessed_dict
        assert preprocessed_dict['binary_image'] is not None # Ensure binary_image was created
        # Compare shapes, ignoring color channel for sample_image
        assert preprocessed_dict['binary_image'].shape == sample_image.shape[:2]

        # Check segmentation output
        symbol_candidates = results['segmentation']
        assert len(symbol_candidates) > 0

        # Check recognition output
        recognition_results = results['recognition']
        assert recognition_results is not None

    def test_pipeline_performance(self, sample_image, tmp_path):
        """Test pipeline performance under load"""
        # Create multiple test images
        test_images = []
        for i in range(5):
            test_image_path = os.path.join(
                tmp_path, f'test_sheet_music_{i}.png')
            cv2.imwrite(test_image_path, sample_image)
            test_images.append(test_image_path)

        # Initialize pipeline
        pipeline = OMRPipeline()

        # Process multiple images
        all_results = []
        for image_path in test_images:
            results = pipeline.process_sheet_music(image_path)
            all_results.append(results)

        # Validate performance results
        assert len(all_results) == 5
        for results in all_results:
            assert 'preprocessing' in results
            assert 'segmentation' in results
            assert 'recognition' in results

    def test_error_scenarios(self, tmp_path):
        """Test pipeline error handling scenarios"""
        pipeline = OMRPipeline()

        # Test non-image file
        with pytest.raises(Exception):
            pipeline.process_sheet_music(
                os.path.join(tmp_path, 'non_image_file.txt'))

        # Test corrupted image
        corrupted_image_path = os.path.join(tmp_path, 'corrupted_image.png')
        with open(corrupted_image_path, 'wb') as f:
            f.write(b'corrupted_data')

        with pytest.raises(Exception):
            pipeline.process_sheet_music(corrupted_image_path)
