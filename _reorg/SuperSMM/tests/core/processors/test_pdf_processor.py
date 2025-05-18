"""Tests for the PDF processor."""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch

from src.core.processors.pdf_processor import PDFProcessor
from src.core.pdf.base_converter import BasePDFConverter


class MockConverter(BasePDFConverter):
    """Mock converter for testing."""
    
    def convert_pdf(self, pdf_path: str):
        """Override to bypass file checks."""
        if Path(pdf_path).name == 'error.pdf':
            return [{
                'image': None,
                'page_number': 0,
                'size': {'width': 0, 'height': 0},
                'dpi': 0,
                'error': 'Test error'
            }]
            
        return [
            {
                'image': np.zeros((100, 100), dtype=np.uint8),
                'page_number': i + 1,
                'size': {'width': 100, 'height': 100},
                'dpi': self.dpi
            }
            for i in range(2)
        ]
        
    def _convert_pages(self, pdf_path: Path):
        """Not used in tests."""
        return []


class TestPDFProcessor:
    """Test suite for PDFProcessor class."""
    
    def test_init(self):
        """Test processor initialization."""
        processor = PDFProcessor(converter_class=MockConverter)
        assert isinstance(processor.converter, MockConverter)
        assert processor.logger is not None
        
    def test_extract_pages_success(self):
        """Test successful page extraction."""
        processor = PDFProcessor(converter_class=MockConverter)
        pages = processor.extract_pages('test.pdf')
        
        assert len(pages) == 2
        for page in pages:
            assert isinstance(page, np.ndarray)
            assert page.shape == (100, 100)
            
    def test_extract_pages_error(self):
        """Test page extraction error handling."""
        processor = PDFProcessor(converter_class=MockConverter)
        
        with pytest.raises(ValueError, match='Test error'):
            processor.extract_pages('error.pdf')
            
    def test_get_page_metadata_success(self):
        """Test successful metadata retrieval."""
        processor = PDFProcessor(converter_class=MockConverter)
        metadata = processor.get_page_metadata('test.pdf')
        
        assert len(metadata) == 2
        for i, page in enumerate(metadata, 1):
            assert page['page_number'] == i
            assert page['size'] == {'width': 100, 'height': 100}
            assert page['dpi'] == 300
            assert page['error'] is None
            
    def test_get_page_metadata_error(self):
        """Test metadata retrieval error handling."""
        processor = PDFProcessor(converter_class=MockConverter)
        metadata = processor.get_page_metadata('error.pdf')
        
        assert len(metadata) == 1
        page = metadata[0]
        assert page['page_number'] == 0
        assert page['size'] == {'width': 0, 'height': 0}
        assert page['dpi'] == 0
        assert page['error'] == 'Test error'
