"""Decorators for the SuperSMM project.

This module contains utility decorators for common tasks like timing, logging, and more.
"""

import time
import functools
import logging
from typing import Callable, Any, TypeVar, cast

# Create a logger for this module
logger = logging.getLogger(__name__)

def log_performance(func: Callable) -> Callable:
    """
    Decorator to log the execution time of a function.
    
    Args:
        func: The function to be decorated
        
    Returns:
        The wrapped function with performance logging
    """
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time = time.perf_counter()
        result = None
        try:
            logger.debug(f"Starting {func.__name__}")
            result = func(*args, **kwargs)
            return result
        finally:
            end_time = time.perf_counter()
            exec_time = (end_time - start_time) * 1000  # Convert to milliseconds
            logger.debug(f"Finished {func.__name__} in {exec_time:.2f}ms")
    
    return wrapper

def log_exceptions(func: Callable) -> Callable:
    """
    Decorator to log exceptions raised by a function.
    
    Args:
        func: The function to be decorated
        
    Returns:
        The wrapped function with exception logging
    """
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}", exc_info=True)
            raise
    
    return wrapper
