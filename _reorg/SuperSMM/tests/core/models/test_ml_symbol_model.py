"""Tests for the ML Symbol Recognition Model."""

import pytest
import numpy as np
from src.core.models.ml_symbol_model import MLSymbolModel

@pytest.fixture
def model():
    """Create a test instance of MLSymbolModel."""
    return MLSymbolModel()

def test_model_initialization(model):
    """Test that model initializes with mock model when no path provided."""
    assert model.model is not None

def test_predict_symbols(model):
    """Test symbol prediction returns expected format."""
    # Create dummy image data
    test_images = [np.zeros((64, 64, 1)) for _ in range(3)]
    
    predictions = model.predict_symbols(test_images)
    
    assert len(predictions) == 3
    for pred in predictions:
        assert 'label' in pred
        assert 'confidence' in pred
        assert isinstance(pred['label'], str)
        assert isinstance(pred['confidence'], float)

def test_get_symbol_label(model):
    """Test symbol label mapping."""
    # Create dummy prediction array
    prediction = np.zeros(10)
    prediction[4] = 1  # Index 4 is treble_clef
    
    label = model.get_symbol_label(prediction)
    assert label == 'treble_clef'

def test_get_symbol_label_unknown(model):
    """Test handling of unknown predictions."""
    # Create invalid prediction array
    prediction = np.zeros(15)  # Too many classes
    prediction[-1] = 1
    
    label = model.get_symbol_label(prediction)
    assert label == 'unknown'
