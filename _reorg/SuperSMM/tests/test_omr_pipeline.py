import pytest
import numpy as np
from pathlib import Path
from src.core.omr_pipeline import OMRPipeline
from src.config.omr_config import OMRConfig, PreprocessingConfig

@pytest.fixture
def test_config():
    return OMRConfig(
        preprocessing=PreprocessingConfig(
            kernel_size=(3, 3),
            min_image_size=(50, 50),
            max_image_size=(1000, 1000)
        ),
        debug_mode=True
    )

@pytest.fixture
def pipeline(test_config):
    return OMRPipeline(custom_config=test_config)

def test_pipeline_initialization(pipeline, test_config):
    assert pipeline._config.preprocessing.kernel_size == (3, 3)
    assert pipeline._config.debug_mode is True
    assert isinstance(pipeline.output_dir, Path)
    assert isinstance(pipeline.cache_dir, Path)

def test_process_image(pipeline):
    # Create a dummy test image
    test_image = np.zeros((100, 100), dtype=np.uint8)
    test_image[40:60, 40:60] = 255  # Add a white square
    
    results = pipeline.process_sheet_music(test_image)
    
    assert 'preprocessing' in results
    assert 'segmentation' in results
    assert 'recognition' in results
    assert 'metadata' in results

def test_export_results(pipeline, tmp_path):
    # Create dummy results
    results = {
        'preprocessing': np.zeros((100, 100), dtype=np.uint8),
        'segmentation': [np.zeros((10, 10), dtype=np.uint8)],
        'recognition': [{'symbol': 'note', 'confidence': 0.9}]
    }
    
    # Export results
    pipeline.export_results(results, tmp_path)
    
    # Check exported files
    assert (tmp_path / 'preprocessed.png').exists()
    assert (tmp_path / 'segmentation.png').exists()
    assert (tmp_path / 'recognition_results.json').exists()
    
    # In debug mode, should also have performance report
    assert (tmp_path / 'performance_report.json').exists()
