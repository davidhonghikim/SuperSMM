import os
import logging
import tempfile

import numpy as np
import pytest
import cv2

# Absolute imports to ensure reliability
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from core.omr_pipeline import OMRPipeline, AdvancedPreprocessor, SymbolSegmenter, SymbolRecognizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='tests/test_omr_suite.log',
    filemode='w'
)
logger = logging.getLogger(__name__)


class TestOMRSuite:
    @pytest.fixture(scope='class')
    def sample_image(self):
        """Create a sample sheet music image for testing"""
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
            # Create a simple test image
            test_image = np.zeros((500, 700, 3), dtype=np.uint8)

            # Draw staff lines
            for y in range(100, 300, 50):
                cv2.line(test_image, (50, y), (650, y), (255, 255, 255), 2)

            # Draw some simple symbols
            cv2.rectangle(test_image, (100, 150), (150, 200),
                          (255, 255, 255), -1)  # Note
            cv2.rectangle(test_image, (200, 200), (250, 250),
                          (255, 255, 255), -1)  # Rest

            cv2.imwrite(temp_file.name, test_image)
            yield temp_file.name

        # Clean up
        os.unlink(temp_file.name)

    def test_omr_pipeline_full_process(self, sample_image):
        """Comprehensive test of the entire OMR pipeline"""
        logger.info(f"Testing OMR pipeline with image: {sample_image}")

        # Initialize pipeline
        pipeline = OMRPipeline()

        try:
            # Process sheet music
            omr_results = pipeline.process_sheet_music(sample_image)

            # Validate core results structure
            assert 'preprocessing' in omr_results, "Preprocessing results missing"
            assert 'segmentation' in omr_results, "Segmentation results missing"
            assert 'recognition' in omr_results, "Recognition results missing"

            # Validate recognition results
            recognition = omr_results['recognition']
            assert 'symbols' in recognition, "No symbols recognized"
            assert 'confidence_scores' in recognition, "No confidence scores"

            # Validate symbols
            symbols = recognition['symbols']
            assert len(symbols) > 0, "No symbols were recognized"

            # Check confidence scores
            confidence_scores = recognition['confidence_scores']
            assert len(confidence_scores) == len(
                symbols), "Confidence scores do not match number of symbols"

            # Validate preprocessing stages
            preprocessing = omr_results['preprocessing']
            assert 'original' in preprocessing, "Original image not preserved"
            assert 'normalized' in preprocessing, "Normalized image not created"
            assert 'binary' in preprocessing, "Binary image not created"
            assert 'no_staff_lines' in preprocessing, "Staff line removal not performed"

            # Validate segmentation
            segmentation = omr_results['segmentation']
            assert 'symbols' in segmentation, "No symbols found in segmentation"
            assert len(segmentation['symbols']) > 0, "No symbols segmented"

            # Basic sanity checks on symbol features
            for symbol in segmentation['symbols']:
                assert symbol is not None, "Symbol cannot be None"
                assert symbol.size > 0, "Symbol image cannot be empty"

            # Validate music theory analysis
            assert 'music_theory' in omr_results, "Music theory analysis missing"
            music_theory = omr_results['music_theory']

            expected_categories = [
                'notes',
                'rests',
                'clefs',
                'accidentals',
                'unknown']
            assert 'symbol_distribution' in music_theory, "Symbol distribution missing"
            assert 'confidence_metrics' in music_theory, "Confidence metrics missing"

            # Log successful processing
            logger.info("OMR pipeline processing completed successfully")

        except Exception as e:
            logger.error(f"OMR pipeline test failed: {e}", exc_info=True)
            raise

    def test_export_debug_images(self, sample_image):
        """Test debug image export functionality"""
        logger.info("Testing debug image export")

        # Initialize pipeline
        pipeline = OMRPipeline()

        # Process sheet music
        omr_results = pipeline.process_sheet_music(sample_image)

        # Create temporary debug directory
        with tempfile.TemporaryDirectory() as debug_dir:
            # Export debug images
            pipeline.export_debug_images(omr_results, output_dir=debug_dir)

            # Check for expected debug images
            expected_images = [
                'original.png',
                'normalized.png',
                'binary.png',
                'no_staff_lines.png'
            ]

            for img_name in expected_images:
                img_path = os.path.join(debug_dir, img_name)
                assert os.path.exists(
                    img_path), f"Debug image {img_name} not created"

            # Log successful debug image export
            logger.info("Debug image export completed successfully")

    def test_error_handling(self):
        """Test error handling for invalid inputs"""
        logger.info("Testing error handling")

        pipeline = OMRPipeline()

        # Test non-existent file
        with pytest.raises(ValueError, match="Invalid image path"):
            pipeline.process_sheet_music("/path/to/nonexistent/image.png")

        # Test None input
        with pytest.raises(TypeError):
            pipeline.process_sheet_music(None)

    def test_component_integration(self):
        """Test integration of individual components"""
        logger.info("Testing component integration")

        # Initialize individual components
        from preprocessing.advanced_preprocessor import AdvancedPreprocessor
        from segmentation.symbol_segmenter import SymbolSegmenter
        from recognition.symbol_recognizer import SymbolRecognizer

        # Create a test image
        test_image = np.zeros((500, 700, 3), dtype=np.uint8)
        cv2.rectangle(test_image, (100, 200), (200, 300), (255, 255, 255), -1)

        # Test preprocessing
        try:
            preprocessed = preprocessor.process_page(test_image)
            assert 'no_staff_lines' in preprocessed, "Preprocessing failed"
        except Exception as e:
            logger.error(f"Preprocessing failed: {e}", exc_info=True)
            raise

        # Test segmentation
        try:
            segmented = segmenter.segment_page(preprocessed['no_staff_lines'])
            assert 'symbols' in segmented, "Segmentation failed"
        except Exception as e:
            logger.error(f"Segmentation failed: {e}", exc_info=True)
            raise

        # Test recognition
        try:
            recognized = recognizer.recognize_symbols(segmented['symbols'])
            assert len(recognized) > 0, "No symbols recognized"
        except Exception as e:
            logger.error(f"Recognition failed: {e}", exc_info=True)
            raise

        logger.info("Component integration test completed successfully")

# Optional: Performance and stress testing


def test_performance():
    """Comprehensive performance test for the OMR pipeline

    Measures:
    - Processing time per image
    - Memory usage
    - Consistency of results
    """
    logger.info("Starting performance test")

    import time
    import psutil
    import tracemalloc

    # Configuration for performance test
    test_config = {
        'num_images': 10,  # Number of test images
        'image_size': (500, 700),  # Standard image size
        'max_processing_time': 2.0,  # Max time per image (seconds)
        'max_memory_usage': 200 * 1024 * 1024,  # 200 MB
    }

    # Create multiple test images with varying complexity
    def create_test_image(complexity=1):
        """Create a test image with varying symbol complexity"""
        img = np.zeros((test_config['image_size'][0],
                        test_config['image_size'][1], 3), dtype=np.uint8)

        # Draw staff lines
        for y in range(100, 400, 50):
            cv2.line(img, (50, y), (650, y), (255, 255, 255), 2)

        # Add symbols based on complexity
        for i in range(complexity):
            x_offset = 100 + i * 100
            # Different symbol types
            if i % 3 == 0:  # Rectangle (note)
                cv2.rectangle(
                    img, (x_offset, 200), (x_offset + 50, 250), (255, 255, 255), -1)
            elif i % 3 == 1:  # Circle (rest)
                cv2.circle(img, (x_offset, 250), 25, (255, 255, 255), -1)
            else:  # Triangle (clef)
                pts = np.array([
                    [x_offset, 200],
                    [x_offset + 50, 250],
                    [x_offset, 300]
                ], np.int32)
                cv2.fillPoly(img, [pts], (255, 255, 255))

        return img

    # Performance tracking variables
    processing_times = []
    memory_usages = []

    # Initialize pipeline
    pipeline = OMRPipeline()

    # Start memory tracking
    tracemalloc.start()

    # Process multiple images with increasing complexity
    for complexity in range(1, 4):  # 1, 2, 3 symbols
        test_images = [
            create_test_image(complexity) for _ in range(
                test_config['num_images'])]

        for idx, img in enumerate(test_images):
            # Save temporary image
            temp_image_path = os.path.join(
                tempfile.gettempdir(),
                f'test_image_{complexity}_{idx}.png')
            cv2.imwrite(temp_image_path, img)

            try:
                # Measure processing time
                start_time = time.time()

                # Measure memory before processing
                mem_before = tracemalloc.get_traced_memory()[0]

                # Process image
                results = pipeline.process_sheet_music(temp_image_path)

                # Measure processing time
                processing_time = time.time() - start_time
                processing_times.append(processing_time)

                # Measure memory usage
                mem_after = tracemalloc.get_traced_memory()[0]
                memory_usage = mem_after - mem_before
                memory_usages.append(memory_usage)

                # Validate results
                assert 'recognition' in results, f"No recognition results for image {idx}"
                assert len(results['recognition']['symbols']
                           ) > 0, f"No symbols recognized in image {idx}"

                # Log performance details
                logger.info(f"Image {idx} (Complexity {complexity}):")
                logger.info(
                    f"  Processing Time: {processing_time:.4f} seconds")
                logger.info(f"  Memory Usage: {memory_usage / 1024:.2f} KB")
                logger.info(
                    f"  Recognized Symbols: {len(results['recognition']['symbols'])}")

                # Clean up temporary image
                os.unlink(temp_image_path)

            except Exception as e:
                logger.error(f"Performance test failed for image {idx}: {e}")
                logger.error(traceback.format_exc())
                raise

    # Stop memory tracking
    tracemalloc.stop()

    # Performance assertions
    avg_processing_time = np.mean(processing_times)
    avg_memory_usage = np.mean(memory_usages)

    logger.info("Performance Test Summary:")
    logger.info(f"Average Processing Time: {avg_processing_time:.4f} seconds")
    logger.info(f"Average Memory Usage: {avg_memory_usage / 1024:.2f} KB")

    # Assertions to ensure performance meets minimum requirements
    assert avg_processing_time < test_config['max_processing_time'], \
        f"Average processing time {avg_processing_time:.4f}s exceeds limit"

    assert avg_memory_usage < test_config['max_memory_usage'], \
        f"Average memory usage {avg_memory_usage/1024:.2f} KB exceeds limit"

    logger.info("Performance test completed successfully")
