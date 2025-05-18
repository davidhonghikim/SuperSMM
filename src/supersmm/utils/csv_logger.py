"""
Centralized CSV logging utility for model training metrics and other data.

This module provides a consistent way to log structured data to CSV files,
with support for appending to existing logs and automatic file rotation.
"""

import os
import csv
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
import pandas as pd
from .logger import setup_logger
import logging


class CSVLogger:
    """A utility class for logging structured data to CSV files.
    
    This class handles:
    - Creating and appending to CSV files
    - Automatic directory creation
    - File rotation based on size or time
    - Consistent formatting of log entries
    """
    
    def __init__(
        self,
        log_dir: str = "logs/metrics",
        filename: str = "training_metrics",
        max_file_size: int = 10 * 1024 * 1024,  # 10MB
        max_files: int = 5,
        log_type: str = "metrics"
    ):
        """Initialize the CSV logger.
        
        Args:
            log_dir: Base directory for log files
            filename: Base name for log files (without extension)
            max_file_size: Maximum file size in bytes before rotation
            max_files: Maximum number of files to keep
            log_type: Type of log (metrics, training, evaluation, etc.)
        """
        self.log_dir = Path(log_dir)
        self.filename = filename
        self.max_file_size = max_file_size
        self.max_files = max_files
        self.log_type = log_type
        
        # Ensure log directory exists
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up logger for CSV logging operations
        self.logger = setup_logger(
            name=f"csv_logger.{log_type}",
            log_type=log_type,
            log_level=logging.INFO,
            log_to_console=True,
            log_to_file=True
        )
    
    def _get_current_logfile(self) -> Path:
        """Get the path to the current log file."""
        return self.log_dir / f"{self.filename}.csv"
    
    def _rotate_logs(self):
        """Rotate log files if current log exceeds max size."""
        log_file = self._get_current_logfile()
        
        if not log_file.exists():
            return
            
        if log_file.stat().st_size < self.max_file_size:
            return
        
        # Find all existing log files
        log_files = sorted(self.log_dir.glob(f"{self.filename}*.csv"))
        
        # Remove oldest files if we've reached max_files
        while len(log_files) >= self.max_files:
            os.remove(log_files.pop(0))
        
        # Rename existing files
        for i in range(len(log_files), 0, -1):
            src = self.log_dir / f"{self.filename}_{i-1}.csv"
            if src.exists():
                src.rename(self.log_dir / f"{self.filename}_{i}.csv")
        
        # Rename current log file
        log_file.rename(self.log_dir / f"{self.filename}_1.csv")
    
    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """Log a dictionary of metrics to CSV.
        
        Args:
            metrics: Dictionary of metrics to log
            step: Optional step or epoch number
        """
        self._rotate_logs()
        log_file = self._get_current_logfile()
        
        # Add timestamp and step if provided
        entry = {
            "timestamp": datetime.now().isoformat(),
            **metrics
        }
        if step is not None:
            entry["step"] = step
        
        # Convert all values to strings for CSV compatibility
        for k, v in entry.items():
            if isinstance(v, (dict, list)):
                entry[k] = json.dumps(v)
            elif not isinstance(v, (str, int, float, bool)) and v is not None:
                entry[k] = str(v)
        
        # Write to CSV
        file_exists = log_file.exists()
        
        try:
            with open(log_file, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=entry.keys())
                if not file_exists:
                    writer.writeheader()
                writer.writerow(entry)
            
            self.logger.debug(f"Logged metrics to {log_file}")
        except Exception as e:
            self.logger.error(f"Failed to log metrics: {e}", exc_info=True)
    
    def read_metrics(self) -> pd.DataFrame:
        """Read all metrics from the log file into a pandas DataFrame.
        
        Returns:
            DataFrame containing all logged metrics
        """
        log_file = self._get_current_logfile()
        
        if not log_file.exists():
            self.logger.warning(f"No log file found at {log_file}")
            return pd.DataFrame()
        
        try:
            return pd.read_csv(log_file)
        except Exception as e:
            self.logger.error(f"Failed to read metrics: {e}", exc_info=True)
            return pd.DataFrame()


def get_csv_logger(
    name: str,
    log_dir: Optional[str] = None,
    **kwargs
) -> CSVLogger:
    """Get or create a CSV logger instance.
    
    Args:
        name: Name for the logger (used in filenames)
        log_dir: Optional custom log directory
        **kwargs: Additional arguments to pass to CSVLogger
        
    Returns:
        Configured CSVLogger instance
    """
    if log_dir is None:
        log_dir = f"logs/ml/{name}"
    
    return CSVLogger(
        log_dir=log_dir,
        filename=name,
        log_type="metrics",
        **kwargs
    )


# Example usage
if __name__ == "__main__":
    # Create a logger for training metrics
    logger = get_csv_logger("training")
    
    # Log some metrics
    for epoch in range(3):
        metrics = {
            "epoch": epoch,
            "loss": 0.1 * (10 - epoch),
            "accuracy": 0.9 + (0.01 * epoch),
            "learning_rate": 0.001 * (0.9 ** epoch),
            "batch_size": 32,
            "custom_metric": {"key": f"value_{epoch}"}
        }
        logger.log_metrics(metrics, step=epoch)
    
    # Read back the metrics
    df = logger.read_metrics()
    print("\nLogged metrics:")
    print(df)
