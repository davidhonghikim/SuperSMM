"""Custom exceptions for the OMR system."""

from typing import Any, Dict, Optional


class OMRException(Exception):
    """Base exception for all OMR-related errors."""

    def __init__(
        self,
        message: str = "An error occurred in the OMR system",
        context: Optional[Dict[str, Any]] = None,
    ):
        self.message = message
        self.context = context or {}
        super().__init__(self.message)

    def __str__(self) -> str:
        if self.context:
            return f"{self.message} | Context: {self.context}"
        return self.message


class ConfigurationError(OMRException):
    """Raised when there is a configuration error."""

    def __init__(
        self,
        message: str = "Configuration error",
        context: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(f"Configuration error: {message}", context)


class PreprocessingError(OMRException):
    """Raised when an error occurs during image preprocessing."""

    def __init__(
        self,
        message: str = "Preprocessing error",
        context: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(f"Preprocessing error: {message}", context)


class SegmentationError(OMRException):
    """Raised when an error occurs during staff or symbol segmentation."""

    def __init__(
        self,
        message: str = "Segmentation error",
        context: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(f"Segmentation error: {message}", context)


class RecognitionError(OMRException):
    """Raised when an error occurs during symbol recognition."""

    def __init__(
        self,
        message: str = "Recognition error",
        context: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(f"Recognition error: {message}", context)


class ExportError(OMRException):
    """Raised when an error occurs during result export."""

    def __init__(
        self,
        message: str = "Export error",
        context: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(f"Export error: {message}", context)


def create_error_handler(logger):
    """Create a decorator that handles exceptions and logs them.

    Args:
        logger: Logger instance to use for error logging

    Returns:
        Decorator function that handles exceptions
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except OMRException as e:
                logger.error(str(e), exc_info=True)
                raise
            except Exception as e:
                logger.error(
                    f"Unexpected error in {func.__name__}: {str(e)}", exc_info=True
                )
                raise OMRException(
                    f"Unexpected error in {func.__name__}",
                    context={"error": str(e), "function": func.__name__},
                ) from e

        return wrapper

    return decorator
