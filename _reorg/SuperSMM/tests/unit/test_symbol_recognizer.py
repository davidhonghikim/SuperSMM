import os
import pytest
import numpy as np
import tensorflow as tf
from unittest.mock import patch, MagicMock

from recognition.symbol_recognizer import SymbolRecognizer


class TestSymbolRecognizer:
    @pytest.fixture
    def mock_model_path(self, tmp_path):
        """Create a temporary directory for model path and a dummy vocab file."""
        model_dir = tmp_path / "test_model_resources"
        model_dir.mkdir(exist_ok=True) # Allow it to exist if called multiple times
        
        # Define model file path (even if it's not always created/valid for all tests)
        model_file_path = model_dir / "symbol_recognition.h5"
        model_file_path.touch() # Optional: create dummy model file if tests need it to exist

        # Create a dummy vocabulary file in the same directory
        vocab_file = model_dir / "vocabulary_semantic.txt"
        expected_classes = [
            'quarter_note', 'half_note', 'whole_note',
            'eighth_note', 'sixteenth_note',
            'quarter_rest', 'half_rest', 'whole_rest',
            'treble_clef', 'bass_clef',
            'sharp', 'flat', 'natural'
        ]
        with open(vocab_file, 'w') as f:
            for cls_name in expected_classes:
                f.write(f"{cls_name}\n")
        
        return str(model_file_path) # Return path to the .h5 file, vocab is alongside it

    @pytest.fixture
    def sample_symbols(self):
        """Generate sample symbol images for testing"""
        # Create 5 random symbol images
        return [
            np.random.randint(
                0, 255, (64, 64, 3), dtype=np.uint8) for _ in range(5)]

    def test_init_with_default_model(self, mock_model_path):
        """Test initializing SymbolRecognizer with a default model path"""
        with patch('tensorflow.keras.models.load_model', side_effect=OSError):
            with patch('recognition.symbol_recognizer.SymbolRecognizer._create_symbol_recognition_model') as mock_create:
                mock_model = MagicMock()
                mock_create.return_value = mock_model

                recognizer = SymbolRecognizer(model_path=mock_model_path)

                # Verify model creation was called
                mock_create.assert_called_once()
                assert recognizer.model == mock_model

    def test_recognize_symbols(self, mock_model_path, sample_symbols):
        """Test symbol recognition process"""
        # Create a mock model with predictable outputs
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([
            [0.9, 0.05, 0.05],  # Confident prediction for first class
            [0.1, 0.8, 0.1],    # Confident prediction for second class
            [0.05, 0.05, 0.9],  # Confident prediction for third class
            [0.6, 0.3, 0.1],    # Less confident prediction
            [0.2, 0.7, 0.1]     # Moderate confidence
        ])

        # Patch the model loading to return our mock model
        with patch('tensorflow.keras.models.load_model', return_value=mock_model):
            recognizer = SymbolRecognizer(model_path=mock_model_path)

            # Perform recognition
            output_data = recognizer.recognize_symbols(sample_symbols)
            recognized_symbols_list = output_data['recognized_symbols']

            # Verify results
            assert len(recognized_symbols_list) == len(sample_symbols)

            # Check structure of recognized symbols
            for symbol_info in recognized_symbols_list:
                assert 'class' in symbol_info  # Method returns 'class'
                assert 'confidence' in symbol_info
                # The test used to check for 'raw_image', which is not returned by current method
                assert 0 <= symbol_info['confidence'] <= 1

    def test_symbol_classes(self, mock_model_path):
        """Test that symbol classes are correctly defined"""
        # Patch model loading as this test only cares about vocabulary
        with patch('tensorflow.keras.models.load_model', MagicMock()):
            recognizer = SymbolRecognizer(model_path=mock_model_path)

        expected_classes = [
            'quarter_note', 'half_note', 'whole_note',
            'eighth_note', 'sixteenth_note',
            'quarter_rest', 'half_rest', 'whole_rest',
            'treble_clef', 'bass_clef',
            'sharp', 'flat', 'natural'
        ]

        assert set(recognizer.config['class_labels']) == set(expected_classes)

    def test_preprocessing(self, mock_model_path):
        """Test symbol preprocessing"""
        # Create a mock recognizer
        with patch('tensorflow.keras.models.load_model', MagicMock()) as mock_load:
            mock_model = MagicMock()
            mock_load.return_value = mock_model

            recognizer = SymbolRecognizer(model_path=mock_model_path)

            # Create a sample symbol image
            sample_symbol = np.random.randint(
                0, 255, (100, 100, 3), dtype=np.uint8)

            # Preprocess the symbol
            preprocessed = recognizer.preprocess_symbol(sample_symbol)

            # Verify preprocessing
            assert preprocessed is not None
            assert len(preprocessed.shape) == 3  # Batch dimension is not added by preprocess_symbol
            # Resized to standard input size
            assert preprocessed.shape[1:3] == (64, 64)
