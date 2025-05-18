#!/usr/bin/env python3
"""
Main training script for the Deep OMR system.
This script serves as an entry point to run the training process.
"""

import os
import sys
import argparse
from typing import Dict, Any, Optional

# Add the parent directory to the path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import modules directly
try:
    # Try importing directly
    import yaml
    from ctc_training import run_training

    def get_training_config(config_path=None):
        """
        Load training configuration from a YAML file.

        Args:
            config_path: Path to the YAML configuration file. If None, uses default path.

        Returns:
            dict: The loaded configuration

        Raises:
            FileNotFoundError: If the config file doesn't exist
            yaml.YAMLError: If there's an error parsing the YAML
        """
        if config_path is None:
            # Default config path relative to project root
            config_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "config",
                "training_config.yaml",
            )

        # Convert to absolute path if it's relative
        if not os.path.isabs(config_path):
            config_path = os.path.abspath(config_path)

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
                if config is None:
                    config = {}
                return config
        except yaml.YAMLError as e:
            print(f"Error parsing YAML configuration: {e}")
            raise

    def validate_training_config(config):
        """Validate the training configuration."""
        # Ensure required fields are present
        required_fields = ["corpus", "set_file", "vocabulary", "save_model"]
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required field '{field}' in configuration")
        return config

except ImportError as e:
    print(f"Error: {e}")
    print("Could not import required modules. Check your installation.")
    sys.exit(1)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train the Deep OMR system")
    parser.add_argument(
        "-config", dest="config_path", type=str, help="Path to the configuration file"
    )
    parser.add_argument(
        "-corpus", dest="corpus", type=str, help="Path to the corpus directory"
    )
    parser.add_argument("-set", dest="set_file", type=str, help="Path to the set file")
    parser.add_argument(
        "-save_model", dest="save_model", type=str, help="Path to save the model"
    )
    parser.add_argument(
        "-vocabulary", dest="vocabulary", type=str, help="Path to the vocabulary file"
    )
    parser.add_argument(
        "-semantic",
        dest="semantic",
        action="store_true",
        help="Use semantic vocabulary",
    )
    parser.add_argument(
        "-agnostic",
        dest="agnostic",
        action="store_true",
        help="Use agnostic vocabulary",
    )
    parser.add_argument(
        "-epochs", dest="epochs", type=int, help="Number of epochs to train for"
    )
    parser.add_argument(
        "-batch_size", dest="batch_size", type=int, help="Batch size for training"
    )

    return parser.parse_args()


def args_to_config(args) -> Dict[str, Any]:
    """Convert command line arguments to a configuration dictionary."""
    config = {}

    # Only add arguments that were explicitly provided
    if args.corpus is not None:
        config["corpus"] = args.corpus
    if args.set_file is not None:
        config["set_file"] = args.set_file
    if args.save_model is not None:
        config["save_model"] = args.save_model
    if args.vocabulary is not None:
        config["vocabulary"] = args.vocabulary
    if args.epochs is not None:
        config["epochs"] = args.epochs
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size

    # Boolean flags
    if args.semantic:
        config["semantic"] = True
    if args.agnostic:
        config["agnostic"] = True
        config["semantic"] = False

    return config


def main():
    """Main entry point for the training script."""
    args = parse_args()
    print(f"Loading configuration from: {args.config_path}")

    # Load and validate configuration
    config = get_training_config(args.config_path)

    # Override config with command line arguments
    cmd_config = args_to_config(args)
    config.update(cmd_config)

    # Validate the configuration
    config = validate_training_config(config)

    # Get project root directory (one level up from src)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def make_absolute(path, base=project_root):
        """Convert a path to absolute, ensuring it exists."""
        if not path:
            return path

        # Skip if already absolute
        if os.path.isabs(path):
            return path

        # Join with base directory and normalize
        abs_path = os.path.abspath(os.path.join(base, path))

        # For directories, check existence of parent
        if key in ["corpus", "save_model"] and not os.path.exists(
            os.path.dirname(abs_path)
        ):
            os.makedirs(os.path.dirname(abs_path), exist_ok=True)

        return abs_path

    # Process each path in the config
    for key in ["corpus", "set_file", "vocabulary", "save_model"]:
        if key in config and config[key]:
            config[key] = make_absolute(config[key])

    # Handle vocabulary selection if not explicitly set
    if "vocabulary" not in config or not config["vocabulary"]:
        vocab_file = (
            "vocabulary_semantic.txt"
            if config.get("semantic", True)
            else "vocabulary_agnostic.txt"
        )
        config["vocabulary"] = os.path.join(project_root, "src", "Data", vocab_file)

    print("Using the following paths:")
    print(f"Corpus: {config.get('corpus')}")
    print(f"Set file: {config.get('set_file')}")
    print(f"Vocabulary: {config.get('vocabulary')}")
    print(f"Save model: {config.get('save_model')}")

    # Run training
    success = run_training(config)

    if success:
        print("Training completed successfully!")
    else:
        print("Training failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
