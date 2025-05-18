import os
import pytest
import numpy as np
import cv2

import sys, os
sys.path.insert(0, '/ml/models/resources/tf-deep-omr/src')

from src.core.omr_pipeline import OMRPipeline
from src.preprocessing.advanced_preprocessor import AdvancedPreprocessor
from src.segmentation.symbol_segmenter import SymbolSegmenter
from src.recognition.symbol_recognizer import SymbolRecognizer


class TestOMRPipelineIntegration:
    @pytest.fixture
    def sample_sheet_music_path(self):
        """Path to the sample sheet music PDF"""
        return "/ml/models/resources/tf-deep-omr/imports/Somewhere_Over_the_Rainbow.pdf"

    @pytest.fixture
    def pdf_to_image(self, sample_sheet_music_path):
        """Convert PDF to image for testing"""
        print("DEBUG: pdf_to_image fixture started.")
        print(f"DEBUG: Attempting to import pdf2image for {sample_sheet_music_path}")
        import pdf2image
        print("DEBUG: pdf2image imported. Calling convert_from_path...")

        # Convert first page of PDF to image
        images = pdf2image.convert_from_path(sample_sheet_music_path, dpi=300, poppler_path="/usr/local/opt/poppler/bin")
        print(f"DEBUG: convert_from_path returned {len(images)} image(s).")

        # Convert to numpy array
        print("DEBUG: Converting image to numpy array...")
        image_array = np.array(images[0])
        print("DEBUG: Numpy array conversion complete.")

        # Convert to grayscale if needed
        print("DEBUG: Checking image_array shape for grayscale conversion...")
        if len(image_array.shape) == 3:
            image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)

        print("DEBUG: Grayscale conversion check complete. Returning image_array.")
        return image_array

    def test_omr_pipeline_full_process(self, pdf_to_image):
        import sys
        print(f"DEBUG_FULL_PROCESS: sys.path: {sys.path}")
        """
        Full integration test of OMR pipeline

        Validates that the entire pipeline can process a real sheet music image
        """
        # Initialize pipeline
        pipeline = OMRPipeline()

        # Preprocess the image
        preprocessed = pipeline.preprocessor.process_page(pdf_to_image)

        # ---- START NEW DEBUG ----
        print(f"DEBUG_FULL_PROCESS: pipeline.segmenter object: {pipeline.segmenter}")
        print(f"DEBUG_FULL_PROCESS: type(pipeline.segmenter): {type(pipeline.segmenter)}")
        print(f"DEBUG_FULL_PROCESS: pipeline.segmenter.__module__: {pipeline.segmenter.__module__}") # ADD THIS
        print(f"DEBUG_FULL_PROCESS: hasattr(pipeline.segmenter, 'segment_symbols'): {hasattr(pipeline.segmenter, 'segment_symbols')}")
        if hasattr(pipeline.segmenter, 'segment_symbols'):
            print(f"DEBUG_FULL_PROCESS: pipeline.segmenter.segment_symbols: {pipeline.segmenter.segment_symbols}")
        print(f"DEBUG_FULL_PROCESS: dir(pipeline.segmenter): {dir(pipeline.segmenter)}")
        import inspect
        print(f"DEBUG_FULL_PROCESS: inspect.getmembers(pipeline.segmenter): {inspect.getmembers(pipeline.segmenter)}")
        # Also, let's check the class itself directly from the imported module
        from src.segmentation.symbol_segmenter import SymbolSegmenter as DirectSymbolSegmenter
        print(f"DEBUG_FULL_PROCESS: inspect.getmembers(DirectSymbolSegmenter): {inspect.getmembers(DirectSymbolSegmenter)}")
        print(f"DEBUG_FULL_PROCESS: 'segment_symbols' in DirectSymbolSegmenter.__dict__: {'segment_symbols' in DirectSymbolSegmenter.__dict__}")
        print(f"DEBUG_FULL_PROCESS: inspect.getfile(DirectSymbolSegmenter): {inspect.getfile(DirectSymbolSegmenter)}")
        try:
            print(f"DEBUG_FULL_PROCESS: inspect.getsource(DirectSymbolSegmenter):\\n{inspect.getsource(DirectSymbolSegmenter)}")
        except Exception as e:
            print(f"DEBUG_FULL_PROCESS: Could not get source for DirectSymbolSegmenter: {e}")
        # ---- END NEW DEBUG ----
        print(f"DEBUG_FULL_PROCESS: preprocessed.keys(): {preprocessed.keys()}")
        # Segment symbols
        segmented_symbols = pipeline.segmenter.segment_symbols(
            preprocessed['image_without_staffs'])

        # Recognize symbols
        recognition_results = pipeline.recognizer.recognize_symbols(
            segmented_symbols['symbol_candidates']
        )

        # Validate results
        assert len(segmented_symbols['symbol_candidates']
                   ) > 0, "No symbols were segmented"
        assert len(recognition_results) > 0, "No symbols were recognized"

        # Optional: Export debug images
        pipeline.export_debug_images({
            'preprocessing': preprocessed,
            'segmentation': segmented_symbols,
            'recognition': {
                'symbols': recognition_results
            }
        }, output_dir='/Users/danger/CascadeProjects/LOO/SuperSMM/exports/debug_images')

    def test_music_theory_analysis(self, sample_sheet_music_path):
        """
        Test music theory analysis generation
        """
        # Initialize pipeline
        pipeline = OMRPipeline()

        # Full processing
        omr_results = pipeline.process_sheet_music(
            sample_sheet_music_path  # Use the fixture path
        )

        # Validate music theory analysis
        assert 'music_theory' in omr_results, "Music theory analysis not generated"

        music_theory = omr_results['music_theory']
        assert isinstance(
            music_theory, dict), "Music theory analysis should be a dictionary"

        # Check basic metrics
        assert 'total_symbols' in music_theory, "Total symbols count missing"
        assert 'symbol_distribution' in music_theory, "Symbol distribution missing"
        assert 'confidence_metrics' in music_theory, "Confidence metrics missing"
