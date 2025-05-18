#!/usr/bin/env python3
"""
Training script with centralized logging for the CTC model.
"""

import os
import sys
import yaml
import argparse
import tensorflow as tf
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import our logging utilities
from utils.logger import setup_logger
from utils.csv_logger import get_csv_logger
from utils.performance_monitor import PerformanceMonitor
from utils.error_handler import ErrorHandler

# Import CTC training components
from src.tf_deep_omr.src import ctc_training
from src.tf_deep_omr.src.ctc_utils import sparse_tuple_from

# Set up logging
logger = setup_logger("training", log_type="training")
metrics_logger = get_csv_logger("training_metrics")
performance_monitor = PerformanceMonitor()
error_handler = ErrorHandler()

def load_config(config_path):
    """Load training configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded configuration from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Failed to load config from {config_path}: {e}")
        raise

def setup_directories(config):
    """Create necessary directories for outputs."""
    dirs = [
        config.get('log_dir', 'logs'),
        config.get('checkpoint_dir', 'checkpoints'),
        os.path.dirname(config.get('save_model', 'models/model')),
    ]
    
    for d in dirs:
        if d:
            os.makedirs(d, exist_ok=True)
            logger.debug(f"Created directory: {d}")

def log_environment():
    """Log environment information."""
    import platform
    import tensorflow as tf
    
    env_info = {
        "Platform": platform.platform(),
        "Python Version": platform.python_version(),
        "TensorFlow Version": tf.__version__,
        "CUDA Available": tf.test.is_built_with_cuda(),
        "GPUs Available": tf.config.list_physical_devices('GPU'),
        "CPU Cores": os.cpu_count(),
        "Working Directory": os.getcwd(),
    }
    
    logger.info("Environment Information:")
    for key, value in env_info.items():
        logger.info(f"  {key}: {value}")

def main():
    parser = argparse.ArgumentParser(description="Train CTC model with centralized logging")
    parser.add_argument("--config", type=str, default="config/training_config.yaml",
                       help="Path to training config file")
    parser.add_argument("--resume", action="store_true",
                       help="Resume training from latest checkpoint")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug logging")
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        config = load_config(args.config)
        
        # Set up directories
        setup_directories(config)
        
        # Log environment info
        log_environment()
        
        # Log training start
        logger.info("Starting CTC model training")
        logger.info(f"Configuration: {config}")
        
        # Convert config to command line args for the CTC trainer
        ctc_args = argparse.Namespace(
            corpus=config.get('corpus', './data'),
            set=config.get('set_file', './data/train.txt'),
            save_model=config.get('save_model', './models/model'),
            voc=config.get('vocabulary', './data/vocabulary.txt'),
            semantic=config.get('semantic', False),
            resume=args.resume,
            max_epochs=config.get('epochs', 50),
            epochs_per_run=config.get('epochs_per_run', 10),
            batch_size=config.get('batch_size', 16),
            img_height=config.get('img_height', 128),
            dropout=config.get('dropout', 0.5)
        )
        
        # Start training
        ctc_training.run_training(ctc_args)
        
        logger.info("Training completed successfully")
        
    except Exception as e:
        error_handler.handle_error(e, "Training failed")
        sys.exit(1)

if __name__ == "__main__":
    main()
