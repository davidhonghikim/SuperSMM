"""Tests for the base PDF converter."""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch

from src.core.pdf.base_converter import BasePDFConverter


class TestConverter(BasePDFConverter):
    """Test implementation of BasePDFConverter."""
    
    def _convert_pages(self, pdf_path):
        """Mock implementation."""
        if pdf_path.name == 'error.pdf':
            raise Exception("Test error")
            
        return [
            {
                'image': np.zeros((100, 100), dtype=np.uint8),
                'page_number': 1,
                'size': {'width': 100, 'height': 100},
                'dpi': self.dpi
            }
        ]


class TestBasePDFConverter:
    """Test suite for BasePDFConverter class."""
    
    def test_init(self):
        """Test converter initialization."""
        converter = TestConverter(dpi=200, fmt='JPEG')
        
        assert converter.dpi == 200
        assert converter.fmt == 'JPEG'
        assert converter.logger is not None
        
    @patch('pathlib.Path.exists')
    def test_convert_pdf_success(self, mock_exists):
        """Test successful PDF conversion."""
        # Setup
        mock_exists.return_value = True
        converter = TestConverter()
        
        # Convert PDF
        results = converter.convert_pdf('test.pdf')
        
        assert len(results) == 1
        result = results[0]
        assert isinstance(result['image'], np.ndarray)
        assert result['page_number'] == 1
        assert result['size'] == {'width': 100, 'height': 100}
        assert result['dpi'] == 300
        assert 'error' not in result
        
    def test_convert_pdf_not_found(self):
        """Test conversion with missing PDF."""
        converter = TestConverter()
        results = converter.convert_pdf('missing.pdf')
        
        assert len(results) == 1
        result = results[0]
        assert result['image'] is None
        assert 'error' in result
        assert 'not found' in result['error']
        
    @patch('pathlib.Path.exists')
    def test_convert_pdf_invalid_extension(self, mock_exists):
        """Test conversion with non-PDF file."""
        mock_exists.return_value = True
        converter = TestConverter()
        results = converter.convert_pdf('test.txt')
        
        assert len(results) == 1
        result = results[0]
        assert result['image'] is None
        assert 'error' in result
        assert 'Not a PDF file' in result['error']
        
    @patch('pathlib.Path.exists')
    def test_convert_pdf_error(self, mock_exists):
        """Test conversion error handling."""
        mock_exists.return_value = True
        converter = TestConverter()
        results = converter.convert_pdf('error.pdf')
        
        assert len(results) == 1
        result = results[0]
        assert result['image'] is None
        assert 'error' in result
        assert 'Test error' in result['error']
