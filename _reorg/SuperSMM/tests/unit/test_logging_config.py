import os
import logging
import pytest

from logging_config import setup_logging, log_performance


class TestLoggingConfig:
    def test_logging_setup(self, tmp_path):
        """Test logging configuration setup"""
        log_dir = tmp_path / 'logs'
        log_file = setup_logging(log_dir=str(log_dir))

        # Check log file creation
        assert os.path.exists(log_file)

        # Check logger configuration
        logger = logging.getLogger()
        assert logger.level == logging.INFO

        # Verify log handlers
        assert len(logger.handlers) > 0

    def test_performance_logging_decorator(self, tmp_path):
        """Test performance logging decorator"""
        log_dir = tmp_path / 'logs'
        setup_logging(log_dir=str(log_dir))

        @log_performance
        def mock_function(x, y):
            return x + y

        # Call the decorated function
        result = mock_function(3, 4)

        # Check result
        assert result == 7

        # Check log file for performance metrics
        log_files = os.listdir(log_dir)
        performance_logs = [f for f in log_files if 'performance' in f]
        assert len(performance_logs) > 0
