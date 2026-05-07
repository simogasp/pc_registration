#!/usr/bin/env python3
"""Plot histogram distributions of localization errors.

This script reads a localization JSON file (from localize_against_map.py)
and creates histogram plots for rotation and translation errors across all scans.
"""

import sys
import argparse
import json
import logging
from pathlib import Path
from typing import Tuple, List

from registration.utils.logging import setup_logging

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from plotting_common import create_error_histograms  # noqa: E402

logger = logging.getLogger(__name__)


def load_localization_results(json_file: Path) -> dict:
    """Load localization results from a JSON file.

    Args:
        json_file: Path to the localization JSON file.

    Returns:
        Dictionary containing localization results.

    Raises:
        FileNotFoundError: If the JSON file does not exist.
        json.JSONDecodeError: If the file is not valid JSON.
    """
    if not json_file.exists():
        raise FileNotFoundError(f"Localization file not found: {json_file}")

    with open(json_file, "r") as f:
        data = json.load(f)

    logger.info(f"Loaded localization results from {json_file.name}")

    params = data.get("parameters", {})
    logger.info(f"  Data directory: {params.get('data_dir', 'N/A')}")
    logger.info(f"  Map file: {params.get('map_file', 'N/A')}")
    logger.info(f"  Number of scans: {data.get('num_scans', 0)}")
    logger.info(f"  Method: {params.get('method', 'N/A')}")
    logger.info(f"  Voxel size: {params.get('voxel_size', 'N/A')}")
    logger.info(
        f"  Max correspondence distance: {params.get('max_correspondence_distance', 'N/A')}"
    )

    return data


def extract_error_values(data: dict) -> Tuple[List[float], List[float]]:
    """Extract rotation and translation error values from localization results.

    Args:
        data: Localization results dictionary.

    Returns:
        Tuple of (rotation_errors, translation_errors) lists.
    """
    # Localization results already have error lists at the top level
    rotation_errors = data.get("rotation_errors", [])
    translation_errors = data.get("translation_errors", [])

    logger.info(f"Extracted {len(rotation_errors)} error values")

    if len(rotation_errors) != len(translation_errors):
        logger.warning(
            f"Mismatch in error counts: {len(rotation_errors)} rotation, "
            f"{len(translation_errors)} translation"
        )

    return rotation_errors, translation_errors


def main(args: argparse.Namespace):
    """Main function to generate error histogram plots.

    Args:
        args: Namespace object containing command-line arguments.
    """
    input_file = Path(args.input)
    output_dir = Path(args.output)

    # Load localization results
    logger.info(f"Loading localization results from {input_file}")
    data = load_localization_results(input_file)

    # Extract error values
    rotation_errors, translation_errors = extract_error_values(data)

    # Create histograms
    logger.info(f"Generating error histograms in {output_dir}")
    create_error_histograms(
        rotation_errors,
        translation_errors,
        output_dir,
        rotation_title="Distribution of Localization Rotation Errors",
        translation_title="Distribution of Localization Translation Errors",
    )

    logger.info("All plots generated successfully")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate histogram plots from localization error data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="Path to localization JSON file (e.g., localization_results.json from localize_against_map.py)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="output/localization_histograms",
        help="Output directory for histogram plots",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set logging level",
    )

    input_args = parser.parse_args()

    setup_logging(level=getattr(logging, input_args.log_level))

    main(input_args)
