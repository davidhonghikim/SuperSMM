#!/usr/bin/env python3
from src.logging_config import setup_logging
from src.config_manager import ConfigManager
from src.core.omr_pipeline import OMRPipeline
import os
import sys
import argparse
import logging

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))


def process_sheet_music(args):
    """
    Process sheet music from input file

    Args:
        args (argparse.Namespace): Command-line arguments
    """
    # Setup logging
    log_file = setup_logging(
        log_level=getattr(logging, args.log_level.upper()), log_dir=args.log_dir
    )

    # Load configuration
    config_manager = ConfigManager(config_path=args.config, default_config=None)

    # Initialize pipeline
    pipeline = OMRPipeline(
        config_path=args.config,
    )

    try:
        # Process sheet music
        results = pipeline.process_sheet_music(args.input)

        # Export results
        if args.output:
            os.makedirs(args.output, exist_ok=True)
            pipeline.export_results(results, args.output)

        print(f"Processing complete. Log file: {log_file}")

    except Exception as e:
        logging.error(f"Sheet music processing failed: {e}")
        sys.exit(1)


def main():
    """
    Main CLI entry point
    """
    parser = argparse.ArgumentParser(description="SuperSMM: Sheet Music Processing CLI")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Process sheet music command
    process_parser = subparsers.add_parser("process", help="Process sheet music file")
    process_parser.add_argument("input", help="Input sheet music file path")
    process_parser.add_argument(
        "-o", "--output", help="Output directory for processed files"
    )
    process_parser.add_argument(
        "-c", "--config", help="Path to configuration file", default=None
    )
    process_parser.add_argument(
        "-l",
        "--log-level",
        choices=["debug", "info", "warning", "error", "critical"],
        default="info",
        help="Logging level",
    )
    process_parser.add_argument(
        "--log-dir", default="logs", help="Directory for log files"
    )
    process_parser.set_defaults(func=process_sheet_music)

    # Parse arguments
    args = parser.parse_args()

    # Execute command
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
