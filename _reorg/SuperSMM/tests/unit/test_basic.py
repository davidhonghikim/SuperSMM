import pytest
from src.core.omr_processor import LocalOMRProcessor


def test_processor_initialization():
    processor = LocalOMRProcessor()
    assert processor is not None
