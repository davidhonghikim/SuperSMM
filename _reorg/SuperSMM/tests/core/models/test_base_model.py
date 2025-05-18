"""Tests for the base model class."""

import pytest
import tensorflow as tf
from pathlib import Path
from unittest.mock import Mock, patch

from src.core.models.base_model import BaseModel


class TestBaseModel:
    """Test suite for BaseModel class."""
    
    def test_init(self):
        """Test model initialization."""
        model_path = "test/path/model.h5"
        model = BaseModel(model_path)
        
        assert model.model_path == Path(model_path)
        assert model.model is None
        assert model.logger is not None
        
    @patch('tensorflow.keras.models.load_model')
    def test_load_model_success(self, mock_load):
        """Test successful model loading."""
        # Setup
        model_path = "test/path/model.h5"
        mock_model = Mock(spec=tf.keras.Model)
        mock_load.return_value = mock_model
        
        # Create model and mock path exists
        model = BaseModel(model_path)
        with patch.object(Path, 'exists', return_value=True):
            loaded = model._load_model()
            
        assert loaded == mock_model
        mock_load.assert_called_once_with(str(model_path))
        
    def test_load_model_not_found(self):
        """Test model loading when file doesn't exist."""
        model_path = "test/path/model.h5"
        model = BaseModel(model_path)
        
        with pytest.raises(FileNotFoundError):
            model._load_model()
            
    @patch('tensorflow.keras.models.load_model')
    def test_load_model_error(self, mock_load):
        """Test model loading when an error occurs."""
        # Setup
        model_path = "test/path/model.h5"
        mock_load.side_effect = Exception("Test error")
        
        # Create model and mock path exists
        model = BaseModel(model_path)
        with patch.object(Path, 'exists', return_value=True):
            with pytest.raises(ValueError):
                model._load_model()
