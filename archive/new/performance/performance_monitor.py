import logging
import time
import functools
from ..utils.logger import setup_logger
import datetime
import psutil
import tracemalloc
import multiprocessing
from typing import Callable, Any, List, Dict, Union
import json
import matplotlib.pyplot as plt
import seaborn as sns


class PerformanceMonitor:
    @staticmethod
    def track_performance(func: Callable) -> Callable:
        """
        Decorator to monitor function performance and resource usage

        Args:
            func (Callable): Function to monitor

        Returns:
            Wrapped function with performance tracking
        """

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Set up logger using the centralized configuration
            logger = setup_logger(
                name="performance_monitor",
                log_type="performance",  # Log to the performance log directory
                log_level=logging.INFO,
                log_to_console=True,
                log_to_file=True
            )

            # Start memory tracking
            tracemalloc.start()

            # Track start time and system resources
            start_time = time.time()
            start_memory = psutil.Process().memory_info().rss / (1024 * 1024)  # MB
            cpu_start = psutil.cpu_percent()

            try:
                # Execute function
                result = func(*args, **kwargs)

                # Calculate performance metrics
                end_time = time.time()
                processing_time = end_time - start_time

                # Memory tracking
                current, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()

                # System resource tracking
                end_memory = psutil.Process().memory_info().rss / (1024 * 1024)
                cpu_end = psutil.cpu_percent()

                # Log performance metrics
                logger.info(f"Function: {func.__name__}")
                logger.info(f"Processing Time: {processing_time:.4f} seconds")
                logger.info(f"Memory Usage: {end_memory - start_memory:.2f} MB")
                logger.info(f"Peak Memory: {peak / (1024 * 1024):.2f} MB")
                logger.info(f"CPU Usage: {cpu_end - cpu_start:.2f}%")

                return result

            except Exception as e:
                logger.error(f"Performance tracking error in {func.__name__}: {e}")
                logger.error(traceback.format_exc())
                raise

        return wrapper

    @staticmethod
    def adaptive_resource_allocation(
        max_memory_mb: float = 500, max_cpu_percent: float = 70
    ):
        """
        Dynamically adjust processing strategy based on system resources

        Args:
            max_memory_mb (float): Maximum allowed memory usage
            max_cpu_percent (float): Maximum allowed CPU utilization

        Returns:
            Adaptive processing configuration
        """
        current_memory = psutil.Process().memory_info().rss / (1024 * 1024)
        current_cpu = psutil.cpu_percent()

        config = {
            "parallel_processing": current_cpu < max_cpu_percent,
            "gpu_acceleration": current_memory < max_memory_mb,
            "compression_level": (
                "high" if current_memory > max_memory_mb * 0.8 else "normal"
            ),
        }

        return config

    @staticmethod
    def comprehensive_performance_analysis(test_functions: list = None) -> dict:
        """
        Perform comprehensive performance analysis across multiple functions

        Args:
            test_functions (list, optional): List of functions to analyze

        Returns:
            dict: Comprehensive performance analysis report
        """
        import datetime
        import multiprocessing

        # Default test functions if none provided
        if test_functions is None:

            def test_func1(x):
                time.sleep(0.1)
                return x * x

            def test_func2(x):
                time.sleep(0.2)
                return x**3

            test_functions = [test_func1, test_func2]

        # Performance tracking results
        performance_results = {
            "timestamp": datetime.datetime.now().isoformat(),
            "system_config": PerformanceMonitor.adaptive_resource_allocation(),
            "function_performance": [],
        }

        # Track performance for each function
        for func in test_functions:
            # Track individual function performance
            start_time = time.time()
            start_memory = psutil.Process().memory_info().rss / (1024 * 1024)
            start_cpu = psutil.cpu_percent()

            # Run function multiple times
            with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
                results = pool.map(func, range(10))

            # Calculate performance metrics
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss / (1024 * 1024)
            end_cpu = psutil.cpu_percent()

            # Store function performance details
            performance_results["function_performance"].append(
                {
                    "function_name": func.__name__,
                    "processing_time": end_time - start_time,
                    "memory_usage": end_memory - start_memory,
                    "cpu_usage": end_cpu - start_cpu,
                    "results_count": len(results),
                }
            )

        # System-wide performance metrics
        performance_results["system_performance"] = {
            "total_memory_gb": psutil.virtual_memory().total / (1024**3),
            "available_memory_gb": psutil.virtual_memory().available / (1024**3),
            "cpu_cores": multiprocessing.cpu_count(),
            "cpu_frequency_mhz": psutil.cpu_freq().current,
        }

        return performance_results


# Example usage
def example_performance_tracking():
    @PerformanceMonitor.track_performance
    def example_function(data):
        # Simulated processing
        time.sleep(1)
        return data * 2

    result = example_function(10)
    print(f"Adaptive Config: {PerformanceMonitor.adaptive_resource_allocation()}")


@staticmethod
def visualize_performance_metrics(
    performance_report: Dict[str, Any], output_dir: str = None
) -> Dict[str, str]:
    """
    Visualize performance metrics from comprehensive performance analysis

    Args:
        performance_report (Dict[str, Any]): Performance analysis report
        output_dir (str, optional): Directory to save visualization outputs

    Returns:
        Dict[str, str]: Paths to generated visualization files
    """
    import os
    import json
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Set default output directory
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), "performance_viz")
    os.makedirs(output_dir, exist_ok=True)

    # Visualization outputs
    viz_outputs = {}

    # 1. Function Performance Comparison
    if performance_report.get("function_performance"):
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 3, 1)
        plt.title("Processing Time")
        plt.bar(
            [f["function_name"] for f in performance_report["function_performance"]],
            [f["processing_time"] for f in performance_report["function_performance"]],
        )
        plt.ylabel("Time (seconds)")
        plt.xticks(rotation=45)

        plt.subplot(1, 3, 2)
        plt.title("Memory Usage")
        plt.bar(
            [f["function_name"] for f in performance_report["function_performance"]],
            [f["memory_usage"] for f in performance_report["function_performance"]],
        )
        plt.ylabel("Memory (MB)")
        plt.xticks(rotation=45)

        plt.subplot(1, 3, 3)
        plt.title("CPU Usage")
        plt.bar(
            [f["function_name"] for f in performance_report["function_performance"]],
            [f["cpu_usage"] for f in performance_report["function_performance"]],
        )
        plt.ylabel("CPU (%)")
        plt.xticks(rotation=45)

        plt.tight_layout()
        function_perf_path = os.path.join(
            output_dir, "function_performance_comparison.png"
        )
        plt.savefig(function_perf_path)
        plt.close()
        viz_outputs["function_performance"] = function_perf_path

    # 2. System Performance Radar Chart
    system_perf = performance_report.get("system_performance", {})
    if system_perf:
        plt.figure(figsize=(8, 8))
        plt.subplot(polar=True)

        # Metrics to plot
        metrics = {
            "Total Memory (GB)": system_perf.get("total_memory_gb", 0),
            "Available Memory (GB)": system_perf.get("available_memory_gb", 0),
            "CPU Cores": system_perf.get("cpu_cores", 0),
            "CPU Frequency (MHz)": system_perf.get("cpu_frequency_mhz", 0),
        }

        # Normalize metrics
        max_values = {
            "Total Memory (GB)": max(metrics["Total Memory (GB)"], 1),
            "Available Memory (GB)": max(metrics["Available Memory (GB)"], 1),
            "CPU Cores": max(metrics["CPU Cores"], 1),
            "CPU Frequency (MHz)": max(metrics["CPU Frequency (MHz)"], 1),
        }

        normalized_metrics = {k: v / max_values[k] for k, v in metrics.items()}

        # Radar chart
        angles = [
            n / float(len(normalized_metrics)) * 2 * 3.14159
            for n in range(len(normalized_metrics))
        ]
        angles += angles[:1]

        values = list(normalized_metrics.values())
        values += values[:1]

        plt.polar(angles[:-1], values[:-1], "o-", linewidth=2)
        plt.fill(angles, values, alpha=0.25)
        plt.xticks(angles[:-1], list(normalized_metrics.keys()))
        plt.title("System Performance Overview")

        system_perf_path = os.path.join(output_dir, "system_performance_radar.png")
        plt.savefig(system_perf_path)
        plt.close()
        viz_outputs["system_performance"] = system_perf_path

    # 3. Adaptive Resource Allocation
    if "system_config" in performance_report:
        plt.figure(figsize=(8, 6))
        config = performance_report["system_config"]
        plt.bar(
            config.keys(),
            [
                int(val) if isinstance(val, bool) else float(val)
                for val in config.values()
            ],
        )
        plt.title("Adaptive Resource Allocation")
        plt.ylabel("Value (0-1)")
        plt.xticks(rotation=45)
        adaptive_config_path = os.path.join(
            output_dir, "adaptive_resource_allocation.png"
        )
        plt.savefig(adaptive_config_path)
        plt.close()
        viz_outputs["adaptive_config"] = adaptive_config_path

    # 4. Generate JSON summary
    summary_path = os.path.join(output_dir, "performance_visualization_summary.json")
    with open(summary_path, "w") as f:
        json.dump(
            {
                "timestamp": performance_report.get("timestamp"),
                "visualizations": viz_outputs,
                "metrics_summary": {
                    "function_performance": performance_report.get(
                        "function_performance", []
                    ),
                    "system_performance": system_perf,
                    "system_config": performance_report.get("system_config", {}),
                },
            },
            f,
            indent=4,
        )

    viz_outputs["summary"] = summary_path

    return viz_outputs
