import os
import sys
import pytest
import numpy as np
import cv2

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
# print('DEBUG sys.path:', sys.path)


@pytest.fixture(scope='session')
def sample_image():
    """Generate a sample sheet music image for testing"""
    # Create a blank white image
    image = np.ones((500, 800, 3), dtype=np.uint8) * 255

    # Draw some mock musical elements
    cv2.line(image, (50, 250), (750, 250), (0, 0, 0), 2)  # Staff line
    cv2.line(image, (50, 270), (750, 270), (0, 0, 0), 2)  # Staff line

    # Draw a mock note
    cv2.circle(image, (200, 240), 20, (0, 0, 0), -1)

    return image


# @pytest.fixture(scope='session')
# def config_path():
#     """Provide path to test configuration"""
#     return os.path.join(os.path.dirname(__file__), '..', '..', 'config.yml')


# @pytest.fixture(scope='module')
# def omr_pipeline():
#     """Create OMR Pipeline instance for testing"""
#     from core.omr_pipeline import OMRPipeline
#     return OMRPipeline()


# def pytest_configure(config):
#     """Configure pytest settings"""
#     config.addinivalue_line(
#         'markers',
#         'integration: mark test as an integration test'
#     )
#     config.addinivalue_line(
#         'markers',
#         'performance: mark test related to performance'
#     )


# def pytest_terminal_summary(terminalreporter, exitstatus, config):
#     """Custom terminal summary reporter"""
#     terminalreporter.write_line('')
#     terminalreporter.write_line('SuperSMM Test Summary')
#     terminalreporter.write_line('-' * 20)
#     terminalreporter.write_line(
#         f'Total tests: {terminalreporter.stats.get("call", [])}')
#     terminalreporter.write_line(
#         f'Passed: {len(terminalreporter.stats.get("passed", []))}')
#     terminalreporter.write_line(
#         f'Failed: {len(terminalreporter.stats.get("failed", []))}')
#     terminalreporter.write_line(
#         f'Skipped: {len(terminalreporter.stats.get("skipped", []))}')
