"""Logging configuration for the SuperSMM project.

This module provides configuration for logging across the application.
It sets up loggers, handlers, and formatters consistently.
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Import the centralized logger
from .logger import setup_logger

def setup_logging(log_level: int = logging.INFO, log_dir: str = "logs", 
                 logger_name: Optional[str] = None, log_type: str = "app") -> logging.Logger:
    """
    Configure comprehensive logging for the OMR pipeline using the centralized logger.

    Args:
        log_level: Logging level (default: logging.INFO)
        log_dir: Directory to store log files (relative to project root)
        logger_name: Name for the logger (default: None for root logger)
        log_type: Type of log (app, debug, ml, etc.) - determines subdirectory
        
    Returns:
        Configured logger instance
    """
    # Convert log level from string if needed (for backward compatibility)
    if isinstance(log_level, str):
        log_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Set up the logger using the centralized utility
    logger = setup_logger(
        name=logger_name or "omr_pipeline",
        log_type=log_type,
        log_level=log_level,
        log_dir=os.path.join(Path(__file__).parent.parent, log_dir, log_type)
    )
    
    return logger


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
