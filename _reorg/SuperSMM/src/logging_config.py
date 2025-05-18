import os
import json
import logging
import logging.handlers
from datetime import datetime
from typing import Dict, Any, List
import matplotlib.pyplot as plt
import seaborn as sns


def setup_logging(log_level=logging.INFO, log_dir="logs", logger_name=None):
    """
    Configure comprehensive logging for the OMR pipeline

    Args:
        log_level (int): Logging level
        log_dir (str): Directory to store log files
    """
    # Create logs directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)

    # Generate unique log filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"omr_pipeline_{timestamp}.log")

    # Configure root logger
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            # Console handler
            logging.StreamHandler(),
            # File handler with rotation
            logging.handlers.RotatingFileHandler(
                log_file, maxBytes=10 * 1024 * 1024, backupCount=5  # 10 MB
            ),
        ],
    )

    # Configure a specific logger for performance metrics if needed, separate from root
    perf_logger = logging.getLogger("performance")
    # Ensure performance logger also gets handlers if not propagated to root or if root has different level
    if not perf_logger.handlers:
        # If you want performance logs to go to the same handlers as root:
        # for handler in logging.root.handlers:
        #     perf_logger.addHandler(handler)
        # Or configure specific handlers for performance if desired.
        # For now, assume it will use root handlers if propagation is enabled (default).
        pass
    perf_logger.setLevel(logging.DEBUG)  # Allow DEBUG level for performance logs

    # Get the root logger to confirm setup (optional to return)
    root_logger = logging.getLogger()
    root_logger.info(
        f"Global logging configured. Log level: {logging.getLevelName(root_logger.getEffectiveLevel())}. Log file: {log_file}"
    )

    # The function's main purpose is setup, so returning None or the root logger is fine.
    # Returning log_file path might still be useful for some contexts.
    return log_file


def log_performance(func):
    """
    Decorator to log function performance
    """
    import time
    import functools

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = logging.getLogger("performance")
        start_time = time.time()
        memory_before = get_memory_usage()

        try:
            result = func(*args, **kwargs)

            # Log performance metrics
            end_time = time.time()
            memory_after = get_memory_usage()

            logger.info(
                f"Performance: {func.__name__} "
                f"Time: {end_time - start_time:.4f}s "
                f"Memory Change: {memory_after - memory_before} MB"
            )

            return result
        except Exception as e:
            logger.error(f"Performance error in {func.__name__}: {e}")
            raise

    return wrapper


def get_memory_usage():
    """
    Get current memory usage in MB
    """
    import psutil
    import os

    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # Convert to MB


def analyze_log_file(log_path: str, log_level: str = "INFO") -> dict:
    """
    Perform comprehensive log file analysis

    Args:
        log_path (str): Path to log file
        log_level (str): Minimum log level to analyze

    Returns:
        dict: Comprehensive log analysis report
    """
    import re
    from collections import defaultdict
    import datetime

    # Log level mapping
    log_levels = {"DEBUG": 0, "INFO": 1, "WARNING": 2, "ERROR": 3, "CRITICAL": 4}

    # Initialize analysis results
    analysis_results = {
        "timestamp": datetime.datetime.now().isoformat(),
        "log_file": log_path,
        "log_level_counts": defaultdict(int),
        "performance_metrics": {
            "total_processing_time": 0.0,
            "total_memory_change": 0.0,
        },
        "error_details": [],
        "performance_entries": [],
    }

    # Log parsing regex patterns
    log_pattern = re.compile(
        r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) - " r"(\w+) - (\w+) - (.+)"
    )
    performance_pattern = re.compile(
        r"Performance: (\w+) " r"Time: (\d+\.\d+)s " r"Memory Change: ([-\d\.]+) MB"
    )

    # Read and analyze log file
    with open(log_path, "r") as log_file:
        for line in log_file:
            # Parse log entry
            match = log_pattern.match(line)
            if match:
                timestamp, logger, level, message = match.groups()

                # Count log levels
                if log_levels.get(level, 0) >= log_levels.get(log_level.upper(), 0):
                    analysis_results["log_level_counts"][level] += 1

                # Capture error details
                if level in ["ERROR", "CRITICAL"]:
                    analysis_results["error_details"].append(
                        {"timestamp": timestamp, "logger": logger, "message": message}
                    )

            # Parse performance entries
            perf_match = performance_pattern.search(line)
            if perf_match:
                func_name, processing_time, memory_change = perf_match.groups()

                # Track performance metrics
                analysis_results["performance_metrics"][
                    "total_processing_time"
                ] += float(processing_time)
                analysis_results["performance_metrics"]["total_memory_change"] += float(
                    memory_change
                )

                analysis_results["performance_entries"].append(
                    {
                        "function_name": func_name,
                        "processing_time": float(processing_time),
                        "memory_change": float(memory_change),
                    }
                )

    return analysis_results


def generate_logging_report(log_path: str, output_dir: str = None) -> dict:
    """
    Generate a comprehensive logging report with visualizations

    Args:
        log_path (str): Path to log file
        output_dir (str, optional): Directory to save visualization outputs

    Returns:
        dict: Comprehensive logging report with visualization paths
    """
    import os
    import json
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Set default output directory
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(log_path), "logging_reports")
    os.makedirs(output_dir, exist_ok=True)

    # Analyze log file
    log_analysis = analyze_log_file(log_path)

    # Prepare report with visualizations
    report = {
        "timestamp": log_analysis["timestamp"],
        "log_file": log_path,
        "visualizations": {},
    }

    # 1. Log Level Distribution
    plt.figure(figsize=(10, 6))
    plt.bar(
        log_analysis["log_level_counts"].keys(),
        log_analysis["log_level_counts"].values(),
    )
    plt.title("Log Level Distribution")
    plt.xlabel("Log Level")
    plt.ylabel("Count")
    log_level_path = os.path.join(output_dir, "log_level_distribution.png")
    plt.savefig(log_level_path)
    plt.close()
    report["visualizations"]["log_level_distribution"] = log_level_path

    # 2. Performance Metrics
    perf_entries = log_analysis["performance_entries"]
    if perf_entries:
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.title("Processing Time Distribution")
        plt.boxplot([entry["processing_time"] for entry in perf_entries])
        plt.ylabel("Processing Time (s)")

        plt.subplot(1, 2, 2)
        plt.title("Memory Change Distribution")
        plt.boxplot([entry["memory_change"] for entry in perf_entries])
        plt.ylabel("Memory Change (MB)")

        performance_metrics_path = os.path.join(output_dir, "performance_metrics.png")
        plt.tight_layout()
        plt.savefig(performance_metrics_path)
        plt.close()
        report["visualizations"]["performance_metrics"] = performance_metrics_path

    # 3. Error Details Visualization
    if log_analysis["error_details"]:
        plt.figure(figsize=(10, 6))
        error_loggers = [error["logger"] for error in log_analysis["error_details"]]
        error_counts = {
            logger: error_loggers.count(logger) for logger in set(error_loggers)
        }
        plt.bar(error_counts.keys(), error_counts.values())
        plt.title("Errors by Logger")
        plt.xlabel("Logger")
        plt.ylabel("Error Count")
        plt.xticks(rotation=45)
        error_distribution_path = os.path.join(output_dir, "error_distribution.png")
        plt.savefig(error_distribution_path)
        plt.close()
        report["visualizations"]["error_distribution"] = error_distribution_path

    # 4. Save detailed JSON report
    report_path = os.path.join(output_dir, "logging_report.json")
    with open(report_path, "w") as f:
        json.dump(
            {"analysis": log_analysis, "visualizations": report["visualizations"]},
            f,
            indent=4,
        )
    report["report_file"] = report_path

    return report


# To use this logging configuration:
# 1. Import the setup_logging function in your main application script.
#    from src.logging_config import setup_logging
# 2. Call setup_logging() early in your application's execution, before other modules that use logging.
#    setup_logging(log_level=logging.INFO, log_dir='logs/pipeline') # Example call
