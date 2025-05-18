import os
import pytest
import numpy as np
import cv2

from src.core.omr_processor import LocalOMRProcessor

# Sample PDF for testing (you'll need to replace with an actual test PDF)
TEST_PDF_PATH = '/Users/danger/CascadeProjects/LOO/SheetMasterMusic/sample_files/pdfs/Somewhere Over the Rainbow.pdf'


@pytest.fixture
def omr_processor():
    return LocalOMRProcessor()


def test_pdf_extraction(omr_processor):
    pages = omr_processor.extract_pdf_pages(TEST_PDF_PATH)

    assert len(pages) > 0, "No pages extracted from PDF"
    assert all(isinstance(page, np.ndarray)
               for page in pages), "Pages are not numpy arrays"


def test_image_preprocessing(omr_processor):
    pages = omr_processor.extract_pdf_pages(TEST_PDF_PATH)
    preprocessed = omr_processor.preprocess_image(pages[0])

    assert preprocessed.shape == pages[0].shape[:
                                                2], "Preprocessing changed image dimensions"
    assert preprocessed.dtype == np.uint8, "Preprocessed image is not uint8"
    assert len(preprocessed.shape) == 2, "Preprocessed image is not grayscale"


def test_staff_line_detection(omr_processor):
    pages = omr_processor.extract_pdf_pages(TEST_PDF_PATH)
    preprocessed = omr_processor.preprocess_image(pages[0])
    staff_lines = omr_processor.detect_staff_lines(preprocessed)

    assert isinstance(
        staff_lines, dict), "Staff line detection did not return a dictionary"
    assert 'total_lines' in staff_lines, "Missing total lines in staff line detection"
    assert 'horizontal_lines' in staff_lines, "Missing horizontal lines in staff line detection"
    assert 'staff_line_spacing' in staff_lines, "Missing staff line spacing in staff line detection"


def test_full_sheet_music_processing(omr_processor):
    results = omr_processor.process_sheet_music(TEST_PDF_PATH)

    assert len(results) > 0, "No pages processed"

    for result in results:
        assert 'preprocessed_image' in result, "Missing preprocessed image in result"
        assert 'staff_line_detection' in result, "Missing staff line detection in result"

        preprocessed = result['preprocessed_image']
        staff_lines = result['staff_line_detection']

        assert preprocessed is not None, "Preprocessed image is None"
        assert staff_lines is not None, "Staff line detection is None"


def test_error_handling(omr_processor):
    # Test with non-existent file
    with pytest.raises(Exception):
        omr_processor.process_sheet_music('/path/to/nonexistent/file.pdf')

# Optional: Performance test


def test_processing_performance(omr_processor):
    import time

    start_time = time.time()
    results = omr_processor.process_sheet_music(TEST_PDF_PATH)
    end_time = time.time()

    processing_time = end_time - start_time
    assert processing_time < 10, f"Processing took too long: {processing_time} seconds"
