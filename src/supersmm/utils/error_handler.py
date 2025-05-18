import traceback
from typing import Callable, Optional
from pathlib import Path

# Use the centralized logger
from .logger import setup_logger


class SuperSMMError(Exception):
    """Base exception for SuperSMM application"""

    pass


class OMRProcessingError(SuperSMMError):
    """Raised when OMR processing encounters an error"""

    pass


class ExportError(SuperSMMError):
    """Raised during export operations"""

    pass


class MLModelError(SuperSMMError):
    """Raised for machine learning model related errors"""

    pass


class ErrorHandler:
    def __init__(self, log_dir: Optional[str] = None):
        """
        Advanced error handling and logging

        Args:
            log_dir: Directory to store error logs (relative to project root).
                    If None, uses the default from the centralized logger.
        """
        # Set up logger using the centralized utility
        self.logger = setup_logger(
            name="error_handler",
            log_type="debug",  # Errors go to debug log directory
            log_level=logging.ERROR  # Only log errors by default
        )
        
        # Store the log file path for reference
        self.error_log_path = (
            Path(log_dir) / "error" / "supersmm_errors.log" 
            if log_dir 
            else None
        )

    def handle_error(self, error: Exception, context: str = None):
        """
        Comprehensive error handling

        Args:
            error (Exception): The exception to handle
            context (str, optional): Additional context about the error
        """
        error_details = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
            "context": context,
        }

        # Log detailed error information
        self.logger.error(
            f"Error Details:\n"
            f"Type: {error_details['type']}\n"
            f"Message: {error_details['message']}\n"
            f"Context: {error_details.get('context', 'N/A')}\n"
            f"Traceback:\n{error_details['traceback']}"
        )

    def retry_on_error(
        self, func: Callable, max_retries: int = 3, retry_delay: float = 1.0
    ) -> Callable:
        """
        Decorator to retry a function on specific errors

        Args:
            func (Callable): Function to retry
            max_retries (int): Maximum number of retry attempts
            retry_delay (float): Delay between retries in seconds

        Returns:
            Callable: Wrapped function with retry logic
        """
        import time
        import functools

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except (OMRProcessingError, ExportError) as e:
                    retries += 1
                    self.handle_error(e, f"Retry attempt {retries}")

                    if retries == max_retries:
                        raise

                    time.sleep(retry_delay)

        return wrapper


def main():
    error_handler = ErrorHandler()

    @error_handler.retry_on_error
    def risky_function():
        """Simulated function with potential errors"""
        import random

        if random.random() < 0.7:
            raise OMRProcessingError("Simulated processing error")
        return "Success!"

    try:
        result = risky_function()
        print(result)
    except Exception as e:
        error_handler.handle_error(e)


if __name__ == "__main__":
    main()
