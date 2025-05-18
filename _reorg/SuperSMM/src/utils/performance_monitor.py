import time
import psutil
import logging
import functools
from typing import Callable


class PerformanceMonitor:
    def __init__(self):
        self.logger = logging.getLogger("performance_monitor")

    def measure_execution_time(self, func: Callable) -> Callable:
        """
        Decorator to measure function execution time and log performance

        Args:
            func (Callable): Function to be monitored

        Returns:
            Callable: Wrapped function with performance measurement
        """

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            start_memory = psutil.Process().memory_info().rss / (1024 * 1024)

            result = func(*args, **kwargs)

            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss / (1024 * 1024)

            execution_time = end_time - start_time
            memory_usage = end_memory - start_memory

            self.logger.info(
                f"Function: {func.__name__}\n"
                f"Execution Time: {execution_time:.4f} seconds\n"
                f"Memory Usage: {memory_usage:.2f} MB"
            )

            return result

        return wrapper

    def profile_system_resources(self):
        """
        Log current system resource utilization
        """
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()

        self.logger.info(
            f"System Resources:\n"
            f"CPU Usage: {cpu_percent}%\n"
            f"Total Memory: {memory.total / (1024*1024):.2f} MB\n"
            f"Available Memory: {memory.available / (1024*1024):.2f} MB\n"
            f"Memory Usage: {memory.percent}%"
        )


def main():
    monitor = PerformanceMonitor()

    @monitor.measure_execution_time
    def sample_function():
        time.sleep(1)  # Simulate work

    sample_function()
    monitor.profile_system_resources()


if __name__ == "__main__":
    main()
