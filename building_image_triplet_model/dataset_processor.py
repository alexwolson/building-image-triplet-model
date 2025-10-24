#!/usr/bin/env python3

"""CLI wrapper for dataset preprocessing.

This module provides a backward-compatible CLI interface for the dataset preprocessing
functionality. The actual processing logic has been refactored into the preprocessing
submodule for better organization and maintainability.
"""

import argparse
from pathlib import Path

from building_image_triplet_model.preprocessing import (
    DatasetProcessor,
    load_processing_config,
    update_config_file,
)


def main() -> None:
    """Main CLI entry point for dataset processing."""
    parser = argparse.ArgumentParser(description="Process building typology dataset")
    parser.add_argument(
        "--config", type=Path, required=True, help="Path to YAML config file (required)"
    )
    args = parser.parse_args()

    # Load configuration
    config = load_processing_config(args.config)

    # Process dataset
    processor = DatasetProcessor(config)
    processor.process_dataset()

    # Update config file with processed values
    update_config_file(args.config, config)


if __name__ == "__main__":
    main()
