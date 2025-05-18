"""Tests for the symbol classifier model."""

import numpy as np
import pytest
import tensorflow as tf
from unittest.mock import Mock, patch

from src.core.models.symbol_classifier import SymbolClassifier


@pytest.fixture
def mock_model():
    """Create a mock TensorFlow model."""
    model = Mock(spec=tf.keras.Model)
    model.predict.return_value = np.array([
        [0.1, 0.8, 0.1],  # Class 1
        [0.7, 0.2, 0.1],  # Class 0
    ])
    return model


class TestSymbolClassifier:
    """Test suite for SymbolClassifier class."""
    
    @patch('src.core.models.base_model.BaseModel._load_model')
    def test_init(self, mock_load):
        """Test classifier initialization."""
        # Setup
        mock_load.return_value = Mock(spec=tf.keras.Model)
        
        # Create classifier
        classifier = SymbolClassifier("test/path")
        
        assert classifier.input_shape == (64, 64, 1)
        assert classifier.num_classes > 0
        assert len(classifier.class_labels) > 0
        assert classifier.model is not None
        
    def test_preprocess_image(self):
        """Test image preprocessing."""
        # Setup
        classifier = SymbolClassifier()
        test_image = np.random.rand(100, 100, 3)
        
        # Process image
        processed = classifier.preprocess_image(test_image)
        
        assert processed.shape[0] == 1  # Batch dimension
        assert processed.shape[1:3] == (64, 64)  # Target size
        assert processed.shape[-1] == 1  # Grayscale
        assert processed.dtype == np.float32
        assert 0 <= processed.min() <= processed.max() <= 1  # Normalized
        
    def test_predict_success(self, mock_model):
        """Test successful symbol prediction."""
        # Setup
        classifier = SymbolClassifier()
        classifier.model = mock_model
        test_images = [np.random.rand(100, 100, 3) for _ in range(2)]
        
        # Make predictions
        results = classifier.predict(test_images)
        
        assert len(results) == 2
        for result in results:
            assert 'label' in result
            assert 'confidence' in result
            assert 'class_index' in result
            assert isinstance(result['confidence'], float)
            assert 0 <= result['confidence'] <= 1
            
    def test_predict_error(self, mock_model):
        """Test prediction error handling."""
        # Setup
        classifier = SymbolClassifier()
        classifier.model = mock_model
        classifier.model.predict.side_effect = Exception("Test error")
        test_images = [np.random.rand(100, 100, 3)]
        
        # Make predictions
        results = classifier.predict(test_images)
        
        assert len(results) == 1
        assert 'error' in results[0]
        assert str(results[0]['error']) == "Test error"
