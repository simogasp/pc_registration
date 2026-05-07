#!/usr/bin/env python3
"""Plot histogram distributions of validation errors.

This script reads a validation JSON file and creates histogram plots
for rotation and translation errors across all validated pairs.
"""

import sys
import argparse
import json
import logging
from pathlib import Path
from typing import List, Tuple

from registration.utils.logging import setup_logging

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from plotting_common import create_error_histograms  # noqa: E402

logger = logging.getLogger(__name__)


def load_validation_results(json_file: Path) -> dict:
    """Load validation results from a JSON file.

    Args:
        json_file: Path to the validation JSON file.

    Returns:
        Dictionary containing validation results.

    Raises:
        FileNotFoundError: If the JSON file does not exist.
        json.JSONDecodeError: If the file is not valid JSON.
    """
    if not json_file.exists():
        raise FileNotFoundError(f"Validation file not found: {json_file}")

    with open(json_file, "r") as f:
        data = json.load(f)

    logger.info(f"Loaded validation results from {json_file.name}")
    logger.info(f"  Dataset: {data.get('dataset', 'N/A')}")
    logger.info(f"  Number of pairs: {data.get('num_pairs', 0)}")
    logger.info(f"  Settings: {data.get('settings', {})}")

    return data


def extract_error_values(data: dict) -> Tuple[List[float], List[float]]:
    """Extract rotation and translation error values from validation results.

    Args:
        data: Validation results dictionary.

    Returns:
        Tuple of (rotation_errors, translation_errors) lists.
    """
    rotation_errors = []
    translation_errors = []

    pairs = data.get("pairs", [])

    for pair in pairs:
        errors = pair.get("errors", {})
        rotation_errors.append(errors.get("rotation_degrees", 0.0))
        translation_errors.append(errors.get("translation", 0.0))

    logger.info(f"Extracted {len(rotation_errors)} error pairs")

    return rotation_errors, translation_errors


def main(args: argparse.Namespace):
    """Main function to generate error histogram plots.

    Args:
        args: Namespace object containing command-line arguments.
    """
    input_file = Path(args.input)
    output_dir = Path(args.output)

    # Load validation results
    logger.info(f"Loading validation results from {input_file}")
    data = load_validation_results(input_file)

    # Extract error values
    rotation_errors, translation_errors = extract_error_values(data)

    # Create histograms
    logger.info(f"Generating error histograms in {output_dir}")
    create_error_histograms(
        rotation_errors,
        translation_errors,
        output_dir,
        rotation_title="Distribution of Validation Rotation Errors",
        translation_title="Distribution of Validation Translation Errors",
    )

    logger.info("All plots generated successfully")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate histogram plots from validation error data"
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="Path to validation JSON file (e.g., validation_no-gt_step80_vs450.json)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="output/validation_histograms",
        help="Output directory for histogram plots (default: output/validation_histograms)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set logging level (default: INFO)",
    )

    input_args = parser.parse_args()

    setup_logging(level=getattr(logging, input_args.log_level))

    main(input_args)
