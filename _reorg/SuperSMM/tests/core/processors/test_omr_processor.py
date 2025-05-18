"""Tests for the main OMR Processor."""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.core.processors.omr_processor import LocalOMRProcessor
from src.core.pdf.pdf2image_converter import PDF2ImageConverter
from src.core.processors.image_preprocessor import ImagePreprocessor
from src.core.processors.staff_remover import StaffRemover
from src.core.processors.symbol_segmenter import SymbolSegmenter
from src.core.models.symbol_classifier import SymbolClassifier

@pytest.fixture
def processor():
    """Create a test instance of LocalOMRProcessor."""
    with patch('src.core.models.symbol_classifier.SymbolClassifier._load_model'):
        return LocalOMRProcessor('test/model/path')

@pytest.fixture
def mock_pdf_pages():
    """Create mock PDF pages."""
    return [
        {
            'image': np.ones((100, 100, 3), dtype=np.uint8),
            'page_number': i + 1,
            'size': {'width': 100, 'height': 100},
            'dpi': 300
        } for i in range(2)
    ]

@pytest.fixture
def mock_preprocessed():
    """Create mock preprocessing result."""
    return {
        'binary': np.ones((100, 100), dtype=np.uint8),
        'grayscale': np.ones((100, 100), dtype=np.uint8),
        'denoised': np.ones((100, 100), dtype=np.uint8),
        'threshold_params': {'block_size': 11, 'c_value': 2}
    }

@pytest.fixture
def mock_staff_result():
    """Create mock staff removal result."""
    return {
        'image': np.ones((100, 100), dtype=np.uint8),
        'staff_lines': [
            {'x': 0, 'y': 10, 'width': 100, 'height': 2},
            {'x': 0, 'y': 30, 'width': 100, 'height': 2}
        ],
        'line_count': 2
    }

def test_processor_initialization(processor):
    """Test processor initialization."""
    assert isinstance(processor.preprocessor, ImagePreprocessor)
    assert isinstance(processor.staff_remover, StaffRemover)
    assert isinstance(processor.segmenter, SymbolSegmenter)
    assert isinstance(processor.classifier, SymbolClassifier)
    assert isinstance(processor.pdf_processor.converter, PDF2ImageConverter)

def test_process_sheet_music(
    processor,
    mock_pdf_pages,
    mock_preprocessed,
    mock_staff_result
):
    """Test full sheet music processing pipeline."""
    # Mock PDF processor
    processor.pdf_processor.extract_pages = MagicMock(return_value=mock_pdf_pages)
    
    # Mock component methods
    processor.preprocessor.preprocess = MagicMock(return_value=mock_preprocessed)
    processor.staff_remover.remove_staff = MagicMock(return_value=mock_staff_result)
    processor.segmenter.segment = MagicMock(return_value=[
        {
            'image': np.zeros((64, 64), dtype=np.uint8),
            'position': {'x': 10, 'y': 20},
            'size': {'width': 64, 'height': 64},
            'area': 4096
        }
    ])
    processor.classifier.predict = MagicMock(return_value=[
        {
            'label': 'quarter_note',
            'confidence': 0.95,
            'class_index': 2
        }
    ])
    
    # Process test PDF
    results = processor.process_sheet_music('test.pdf')
    
    # Verify results
    assert len(results) == 2  # Two pages
    for result in results:
        assert result['page_number'] > 0
        assert 'symbols' in result
        assert 'staff_lines' in result
        assert 'preprocessing' in result
        assert 'error' not in result
        
        # Check symbol data
        assert len(result['symbols']) > 0
        symbol = result['symbols'][0]
        assert 'image' in symbol
        assert 'position' in symbol
        assert 'label' in symbol
        assert 'confidence' in symbol

def test_process_sheet_music_error(processor):
    """Test error handling in processing pipeline."""
    pdf_path = '/data/input/scores/Somewhere_Over_the_Rainbow.pdf'
    
    # Test file not found
    processor.pdf_processor.extract_pages = MagicMock(
        return_value=[{
            'error': 'PDF not found',
            'page_number': 0
        }]
    )
    results = processor.process_sheet_music(pdf_path)
    assert len(results) == 1
    assert 'error' in results[0]
    assert 'PDF not found' in results[0]['error']
    
    # Test extraction error
    processor.pdf_processor.extract_pages = MagicMock(
        return_value=[{
            'error': 'PDF processing failed',
            'page_number': 0
        }]
    )
    results = processor.process_sheet_music(pdf_path)
    assert len(results) == 1
    assert 'error' in results[0]
    assert 'PDF processing failed' in results[0]['error']
    
def test_process_page(processor, mock_pdf_pages):
    """Test single page processing."""
    # Test preprocessing error
    processor.preprocessor.preprocess = MagicMock(return_value={'error': 'Test error'})
    result = processor._process_page(mock_pdf_pages[0]['image'])
    assert 'error' in result
    assert result['error'] == 'Test error'
    
    # Test staff removal error
    processor.preprocessor.preprocess = MagicMock(return_value={
        'binary': np.ones((100, 100), dtype=np.uint8),
        'threshold_params': {'block_size': 11, 'c_value': 2}
    })
    processor.staff_remover.remove_staff = MagicMock(return_value={'error': 'Test error'})
    result = processor._process_page(mock_pdf_pages[0])
    assert 'error' in result
    assert result['error'] == 'Test error'
    
    # Test no symbols found
    processor.staff_remover.remove_staff = MagicMock(return_value={
        'image': np.ones((100, 100), dtype=np.uint8),
        'staff_lines': []
    })
    processor.segmenter.segment = MagicMock(return_value=[])
    result = processor._process_page(mock_pdf_pages[0])
    assert 'error' in result
    assert result['error'] == 'No symbols detected'
