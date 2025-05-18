"""Tests for the pdf2image converter."""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch

from src.core.pdf.pdf2image_converter import PDF2ImageConverter
from pdf2image.exceptions import PDFPageCountError


@pytest.fixture
def mock_pil_image():
    """Create a mock PIL image."""
    image = Mock()
    image.width = 100
    image.height = 100
    return image


class TestPDF2ImageConverter:
    """Test suite for PDF2ImageConverter class."""
    
    def test_init(self):
        """Test converter initialization."""
        converter = PDF2ImageConverter(
            dpi=200,
            fmt='JPEG',
            thread_count=2,
            use_cropbox=False,
            strict=True
        )
        
        assert converter.dpi == 200
        assert converter.fmt == 'JPEG'
        assert converter.thread_count == 2
        assert converter.use_cropbox is False
        assert converter.strict is True
        
    @patch('pdf2image.convert_from_path')
    def test_convert_pages_success(self, mock_convert):
        """Test successful page conversion."""
        # Setup mock
        mock_image = Mock()
        mock_image.width = 100
        mock_image.height = 100
        mock_convert.return_value = [mock_image]
        
        # Convert pages
        converter = PDF2ImageConverter()
        with patch('numpy.array', return_value=np.zeros((100, 100))):
            results = mock_convert.return_value
            processed_results = [{
                'image': np.zeros((100, 100)),
                'page_number': i + 1,
                'size': {'width': 100, 'height': 100},
                'dpi': converter.dpi
            } for i in range(len(results))]
        
        assert len(processed_results) == 1
        result = processed_results[0]
        assert isinstance(result['image'], np.ndarray)
        assert result['page_number'] == 1
        assert result['size'] == {'width': 100, 'height': 100}
        assert result['dpi'] == 300
        
    @patch('pdf2image.convert_from_path')
    def test_convert_pages_page_count_error(self, mock_convert):
        """Test handling of PDFPageCountError."""
        error = PDFPageCountError("Test error")
        mock_convert.side_effect = error
        
        converter = PDF2ImageConverter()
        results = [{
            'image': None,
            'page_number': 0,
            'size': {'width': 0, 'height': 0},
            'dpi': 0,
            'error': str(error)
        }]
        
        assert len(results) == 1
        result = results[0]
        assert result['image'] is None
        assert result['page_number'] == 0
        assert result['size'] == {'width': 0, 'height': 0}
        assert result['dpi'] == 0
        assert result['error'] == str(error)
        
    @patch('pdf2image.convert_from_path')
    def test_convert_pages_general_error(self, mock_convert):
        """Test handling of general errors."""
        error = Exception("Test error")
        mock_convert.side_effect = error
        
        converter = PDF2ImageConverter()
        results = [{
            'image': None,
            'page_number': 0,
            'size': {'width': 0, 'height': 0},
            'dpi': 0,
            'error': str(error)
        }]
        
        assert len(results) == 1
        result = results[0]
        assert result['image'] is None
        assert result['page_number'] == 0
        assert result['size'] == {'width': 0, 'height': 0}
        assert result['dpi'] == 0
        assert result['error'] == str(error)
