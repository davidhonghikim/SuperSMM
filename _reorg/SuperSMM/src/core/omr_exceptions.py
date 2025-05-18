import logging
import traceback
from typing import Optional, Any, Callable, Dict
import functools
import re
import datetime
from collections import defaultdict


class OMRBaseError(Exception):
    """Base exception for OMR pipeline errors"""

    def __init__(self, message: str, context: Optional[dict] = None):
        super().__init__(message)
        self.context = context or {}
        self.logger = logging.getLogger(__name__)
        self._log_error()

    def _log_error(self):
        """Log error with additional context"""
        error_details = {
            "error_type": self.__class__.__name__,
            "message": str(self),
            "context": self.context,
        }
        self.logger.error(f"OMR Error: {error_details}")


class PreprocessingError(OMRBaseError):
    """Raised when image preprocessing fails"""

    pass


class SegmentationError(OMRBaseError):
    """Raised when symbol segmentation fails"""

    pass


class RecognitionError(OMRBaseError):
    """Raised when symbol recognition fails"""

    pass


class ConfigurationError(OMRBaseError):
    """Raised when configuration is invalid"""

    pass


class TimeoutError(OMRBaseError):
    """Raised when processing exceeds time limit"""

    pass


def create_error_handler(default_return: Any = None):
    """
    Decorator for error handling and logging

    Args:
        default_return: Value to return if an error occurs

    Returns:
        Callable: Decorated function with error handling
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Use self.logger if available, else use module logger
            logger = None
            if args and hasattr(args[0], "logger"):
                logger = getattr(args[0], "logger")
            else:
                logger = logging.getLogger(func.__module__)
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Log the error
                logger.error(f"Error in {func.__name__}: {str(e)}")
                logger.error(f"Args: {args}, Kwargs: {kwargs}")

                # Determine specific error type
                if isinstance(e, PreprocessingError):
                    logger.error("Preprocessing stage error")
                elif isinstance(e, SegmentationError):
                    logger.error("Segmentation stage error")
                elif isinstance(e, RecognitionError):
                    logger.error("Recognition stage error")

                # Optional: Add error traceback
                logger.error(traceback.format_exc())

                # Return default or re-raise
                if default_return is not None:
                    return default_return
                raise

        return wrapper

    return decorator


def generate_error_report(error_log_path: str) -> dict:
    """
    Generate comprehensive error report from log file

    Args:
        error_log_path (str): Path to log file containing errors

    Returns:
        dict: Comprehensive error analysis report
    """
    import re
    import datetime
    from collections import defaultdict

    # Initialize error report
    error_report = {
        "timestamp": datetime.datetime.now().isoformat(),
        "log_file": error_log_path,
        "error_summary": {
            "total_errors": 0,
            "error_types": defaultdict(int),
            "error_stages": defaultdict(int),
        },
        "detailed_errors": [],
    }

    # Error parsing regex patterns
    error_pattern = re.compile(
        r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) - " r"(\w+) - ERROR - (.+)"
    )

    # Predefined error stages
    error_stages = {
        "PreprocessingError": "preprocessing",
        "SegmentationError": "segmentation",
        "RecognitionError": "recognition",
        "ConfigurationError": "configuration",
    }

    # Read and analyze log file
    with open(error_log_path, "r") as log_file:
        for line in log_file:
            # Parse error entry
            match = error_pattern.match(line)
            if match:
                timestamp, logger, message = match.groups()

                # Identify error type and stage
                error_type = "Unknown"
                error_stage = "unknown"

                for stage_key, stage_name in error_stages.items():
                    if stage_key in message:
                        error_type = stage_key
                        error_stage = stage_name
                        break

                # Update error summary
                error_report["error_summary"]["total_errors"] += 1
                error_report["error_summary"]["error_types"][error_type] += 1
                error_report["error_summary"]["error_stages"][error_stage] += 1

                # Store detailed error
                error_report["detailed_errors"].append(
                    {
                        "timestamp": timestamp,
                        "logger": logger,
                        "message": message,
                        "error_type": error_type,
                        "error_stage": error_stage,
                    }
                )

    # Compute error severity
    error_report["error_severity"] = {
        "critical_errors": error_report["error_summary"]["error_types"].get(
            "ConfigurationError", 0
        ),
        "high_severity_errors": (
            error_report["error_summary"]["error_types"].get("PreprocessingError", 0)
            + error_report["error_summary"]["error_types"].get("SegmentationError", 0)
        ),
        "medium_severity_errors": error_report["error_summary"]["error_types"].get(
            "RecognitionError", 0
        ),
    }

    return error_report
