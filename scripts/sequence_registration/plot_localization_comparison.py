#!/usr/bin/env python3
"""Plot comparison of localization results across different voxel sizes and methods.

This script analyzes the results from running localize_against_map.py with different
configurations (voxel sizes and methods) and creates comparative box plots showing
the distribution of rotation and translation errors.
"""

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Dict, Tuple, Optional, List

from registration.utils.logging import setup_logging

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from plotting_common import create_grouped_barplot, create_grouped_boxplot  # noqa: E402

logger = logging.getLogger(__name__)

# Method name mapping from directory name to display name
METHOD_MAP = {
    "ransac": "RANSAC only",
    "ransac_icp": "RANSAC + ICP",
    "ransac_gicp": "RANSAC + GICP",
}

# Color scheme for methods
METHOD_COLORS = {
    "RANSAC only": "#90f3f4",  # Red
    "RANSAC + ICP": "#f39c12",  # Orange
    "RANSAC + GICP": "#27ae60",  # Green
}


def parse_directory_name(dirname: str) -> Tuple[str, float]:
    """Extract method and voxel size from directory name.

    Args:
        dirname: Directory name (e.g., 'ransac_vs50', 'ransac_gicp_vs450').

    Returns:
        Tuple of (method_display_name, voxel_size).

    Raises:
        ValueError: If the directory name does not match the expected pattern.
    """
    pattern = r"(ransac(?:_icp|_gicp)?)_vs(\d+(?:\.\d+)?)"
    match = re.search(pattern, dirname)
    if not match:
        raise ValueError(f"Directory name '{dirname}' does not match expected pattern")

    method_key = match.group(1)
    voxel_size = float(match.group(2))

    if method_key not in METHOD_MAP:
        raise ValueError(f"Unknown method: {method_key}")

    return METHOD_MAP[method_key], voxel_size


def load_localization_results(base_dir: Path) -> Dict[float, Dict[str, dict]]:
    """Load all localization results from a base directory.

    Args:
        base_dir: Base directory containing result subdirectories.

    Returns:
        Nested dictionary: {voxel_size: {method: statistics_dict}}.
    """
    results = {}

    # Find all subdirectories matching the pattern
    for subdir in base_dir.iterdir():
        if not subdir.is_dir():
            continue

        try:
            method, voxel_size = parse_directory_name(subdir.name)
        except ValueError as e:
            logger.debug(f"Skipping directory {subdir.name}: {e}")
            continue

        # Load the localization_results.json file
        json_file = subdir / "localization_results.json"
        if not json_file.exists():
            logger.warning(
                f"Missing localization_results.json in {subdir.name}, skipping"
            )
            continue

        try:
            with open(json_file, "r") as f:
                data = json.load(f)

            if "statistics" not in data:
                logger.warning(
                    f"No statistics found in {json_file.relative_to(base_dir)}"
                )
                continue

            # Organize by voxel size and method
            if voxel_size not in results:
                results[voxel_size] = {}

            results[voxel_size][method] = {
                "statistics": data["statistics"],
                "rotation_errors": data.get("rotation_errors", []),
                "translation_errors": data.get("translation_errors", []),
            }

            logger.info(
                f"Loaded {subdir.name}: voxel_size={voxel_size}, method={method}"
            )

        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Error loading {json_file.relative_to(base_dir)}: {e}")
            continue

    return results


def extract_metric_data(
    results: Dict[float, Dict[str, dict]], metric_name: str
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Extract data for a specific metric organized for box plot creation.

    Args:
        results: Dictionary of localization results.
        metric_name: Name of the metric to extract (e.g., 'rotation_error_degrees').

    Returns:
        Nested dictionary: {voxel_size_str: {method: {stat_name: value}}}.
    """
    organized_data = {}

    for voxel_size, methods_data in results.items():
        voxel_key = str(int(voxel_size))  # Convert to string for plotting

        if voxel_key not in organized_data:
            organized_data[voxel_key] = {}

        for method, method_data in methods_data.items():
            stats = method_data["statistics"]
            if metric_name not in stats:
                logger.warning(f"Metric {metric_name} not found for {method}")
                continue

            organized_data[voxel_key][method] = stats[metric_name]

    return organized_data


def compute_success_rate(
    rotation_errors: List[float],
    translation_errors: List[float],
    max_rot_err: float,
    max_transl_err: float,
) -> float:
    """Compute the fraction of scans that satisfy both error thresholds.

    A scan is considered successfully localized when its rotation error is strictly
    below max_rot_err AND its translation error is strictly below max_transl_err.

    Args:
        rotation_errors: Per-scan rotation errors in degrees.
        translation_errors: Per-scan translation errors.
        max_rot_err: Rotation error threshold in degrees.
        max_transl_err: Translation error threshold.

    Returns:
        Success rate as a fraction in [0, 1]. Returns 0.0 for empty inputs.
    """
    if not rotation_errors or not translation_errors:
        return 0.0

    n = min(len(rotation_errors), len(translation_errors))
    successes = sum(
        rot < max_rot_err and transl < max_transl_err
        for rot, transl in zip(rotation_errors[:n], translation_errors[:n])
    )
    return successes / n


def extract_success_rate_data(
    results: Dict[float, Dict[str, dict]],
    max_rot_err: float,
    max_transl_err: float,
) -> Dict[str, Dict[str, float]]:
    """Extract success rate data organized for bar plot creation.

    Args:
        results: Dictionary of localization results {voxel_size: {method: data}}.
        max_rot_err: Rotation error threshold in degrees.
        max_transl_err: Translation error threshold.

    Returns:
        Nested dictionary: {voxel_size_str: {method: success_rate}} where success
        rate is a fraction in [0, 1].
    """
    organized_data: Dict[str, Dict[str, float]] = {}

    for voxel_size, methods_data in results.items():
        voxel_key = str(int(voxel_size))
        organized_data[voxel_key] = {}

        for method, method_data in methods_data.items():
            rot_errors = method_data.get("rotation_errors", [])
            transl_errors = method_data.get("translation_errors", [])

            if not rot_errors or not transl_errors:
                logger.warning(
                    f"No per-scan error data for {method} at voxel size {voxel_size}"
                )
                continue

            organized_data[voxel_key][method] = compute_success_rate(
                rot_errors, transl_errors, max_rot_err, max_transl_err
            )

    return organized_data


def _create_boxplots_for_metric(
    results: Dict[float, Dict[str, dict]],
    metric_key: str,
    metric_label: str,
    filename_prefix: str,
    output_dir: Path,
    voxel_sizes: List[float],
    methods: List[str],
    voxel_suffix: str,
    combined_title: str,
    single_title_template: str,
) -> None:
    """Generate a combined boxplot and one per-voxel-size boxplot for a metric.

    Args:
        results: Dictionary of localization results {voxel_size: {method: data}}.
        metric_key: Key of the metric in the statistics dictionary.
        metric_label: Y-axis label for the plots.
        filename_prefix: Output filename prefix (e.g. 'comparison_rotation_error').
        output_dir: Directory where to save the plots.
        voxel_sizes: Sorted list of voxel sizes to include.
        methods: Ordered list of method display names.
        voxel_suffix: Filename suffix derived from the voxel size filter.
        combined_title: Title for the combined (all voxel sizes) plot.
        single_title_template: Title template for per-voxel plots; receives {voxel_int}.
    """
    voxel_labels = [str(int(vs)) for vs in voxel_sizes]
    all_data = extract_metric_data(results, metric_key)

    create_grouped_boxplot(
        data=all_data,
        metric_label=metric_label,
        output_path=output_dir / f"{filename_prefix}{voxel_suffix}.png",
        group_labels=voxel_labels,
        category_labels=methods,
        colors=METHOD_COLORS,
        title=combined_title,
    )

    for voxel_size in voxel_sizes:
        voxel_int = int(voxel_size)
        single_data = extract_metric_data({voxel_size: results[voxel_size]}, metric_key)
        create_grouped_boxplot(
            data=single_data,
            metric_label=metric_label,
            output_path=output_dir / f"{filename_prefix}_{voxel_int}.png",
            group_labels=[str(voxel_int)],
            category_labels=methods,
            colors=METHOD_COLORS,
            title=single_title_template.format(voxel_int=voxel_int),
        )
        logger.debug(f"Created {filename_prefix} plot for voxel size {voxel_int}")


def _create_barplots_for_success_rate(
    results: Dict[float, Dict[str, dict]],
    max_rot_err: float,
    max_transl_err: float,
    output_dir: Path,
    voxel_sizes: List[float],
    methods: List[str],
    voxel_suffix: str,
) -> None:
    """Generate a combined barplot and one per-voxel-size barplot for success rate.

    Args:
        results: Dictionary of localization results {voxel_size: {method: data}}.
        max_rot_err: Rotation error threshold in degrees.
        max_transl_err: Translation error threshold.
        output_dir: Directory where to save the plots.
        voxel_sizes: Sorted list of voxel sizes to include.
        methods: Ordered list of method display names.
        voxel_suffix: Filename suffix derived from the voxel size filter.
    """
    voxel_labels = [str(int(vs)) for vs in voxel_sizes]
    threshold_str = f"rot < {max_rot_err}deg, transl < {max_transl_err}"
    all_data = extract_success_rate_data(results, max_rot_err, max_transl_err)

    create_grouped_barplot(
        data=all_data,
        metric_label="Success Rate (%)",
        output_path=output_dir / f"comparison_success_rate{voxel_suffix}.png",
        group_labels=voxel_labels,
        category_labels=methods,
        colors=METHOD_COLORS,
        title=(
            f"Localization Success Rate Across Voxel Sizes and Methods"
            f" ({threshold_str})"
        ),
    )

    for voxel_size in voxel_sizes:
        voxel_int = int(voxel_size)
        voxel_label = str(voxel_int)
        single_data = {voxel_label: all_data.get(voxel_label, {})}
        create_grouped_barplot(
            data=single_data,
            metric_label="Success Rate (%)",
            output_path=output_dir / f"comparison_success_rate_{voxel_int}.png",
            group_labels=[voxel_label],
            category_labels=methods,
            colors=METHOD_COLORS,
            title=(f"Localization Success Rate - Voxel {voxel_int} ({threshold_str})"),
        )
        logger.debug(f"Created success rate plot for voxel size {voxel_int}")


def create_comparison_plots(
    results: Dict[float, Dict[str, dict]],
    output_dir: Path,
    voxel_sizes_filter: Optional[List[float]] = None,
    max_rot_err: Optional[float] = None,
    max_transl_err: Optional[float] = None,
):
    """Create comparison plots for rotation error, translation error, and success rate.

    Args:
        results: Dictionary of localization results.
        output_dir: Directory where to save the plots.
        voxel_sizes_filter: Optional list of voxel sizes being plotted (used for
            deriving output filenames).
        max_rot_err: Rotation error threshold for counting a scan as a success
            (degrees). If None, success rate plots are skipped.
        max_transl_err: Translation error threshold for counting a scan as a success.
            If None, success rate plots are skipped.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    voxel_sizes = sorted(results.keys())
    methods = list(METHOD_MAP.values())

    voxel_suffix = ""
    if voxel_sizes_filter:
        voxel_suffix = "_" + "_".join(str(int(vs)) for vs in sorted(voxel_sizes_filter))

    logger.info("Creating rotation error comparison plots...")
    _create_boxplots_for_metric(
        results=results,
        metric_key="rotation_error_degrees",
        metric_label="Rotation Error (degrees)",
        filename_prefix="comparison_rotation_error",
        output_dir=output_dir,
        voxel_sizes=voxel_sizes,
        methods=methods,
        voxel_suffix=voxel_suffix,
        combined_title="Localization Rotation Error Comparison Across Voxel Sizes and Methods",
        single_title_template="Localization Rotation Error Comparison (Voxel Size: {voxel_int})",
    )

    logger.info("Creating translation error comparison plots...")
    _create_boxplots_for_metric(
        results=results,
        metric_key="translation_error",
        metric_label="Translation Error",
        filename_prefix="comparison_translation_error",
        output_dir=output_dir,
        voxel_sizes=voxel_sizes,
        methods=methods,
        voxel_suffix=voxel_suffix,
        combined_title="Localization Translation Error Comparison Across Voxel Sizes and Methods",
        single_title_template="Localization Translation Error Comparison (Voxel Size: {voxel_int})",
    )

    if max_rot_err is not None and max_transl_err is not None:
        logger.info(
            f"Creating success rate comparison plots "
            f"(rot < {max_rot_err}deg, transl < {max_transl_err})..."
        )
        _create_barplots_for_success_rate(
            results=results,
            max_rot_err=max_rot_err,
            max_transl_err=max_transl_err,
            output_dir=output_dir,
            voxel_sizes=voxel_sizes,
            methods=methods,
            voxel_suffix=voxel_suffix,
        )


def filter_voxel_sizes(
    results: Dict[float, Dict[str, dict]], requested_sizes: Optional[List[float]]
) -> Dict[float, Dict[str, dict]]:
    """Filter results to only include requested voxel sizes.

    Args:
        results: Dictionary of localization results.
        requested_sizes: List of voxel sizes to include, or None for all.

    Returns:
        Filtered results dictionary.

    Raises:
        ValueError: If any requested voxel size is not found in results.
    """
    if requested_sizes is None:
        return results

    available_sizes = set(results.keys())
    requested_set = set(requested_sizes)

    missing_sizes = requested_set - available_sizes
    if missing_sizes:
        missing_str = ", ".join(str(int(vs)) for vs in sorted(missing_sizes))
        available_str = ", ".join(str(int(vs)) for vs in sorted(available_sizes))
        raise ValueError(
            f"Requested voxel size(s) not found: {missing_str}. "
            f"Available voxel sizes: {available_str}"
        )

    # Filter to only include requested sizes
    filtered = {vs: data for vs, data in results.items() if vs in requested_set}

    logger.info(
        f"Filtered to {len(filtered)} voxel sizes: "
        f"{', '.join(str(int(vs)) for vs in sorted(filtered.keys()))}"
    )

    return filtered


def main(args: argparse.Namespace):
    """Main function to generate localization comparison plots.

    Args:
        args: Namespace object containing command-line arguments.
    """
    base_dir = Path(args.input)
    output_dir = Path(args.output)

    if not base_dir.exists():
        logger.error(f"Input directory does not exist: {base_dir}")
        return

    logger.info(f"Loading localization results from {base_dir}")
    results = load_localization_results(base_dir)

    if not results:
        logger.error("No localization results found")
        return

    num_configs = sum(len(methods) for methods in results.values())
    logger.info(f"Found {num_configs} configurations across {len(results)} voxel sizes")

    # Parse voxel sizes filter if provided
    voxel_sizes_filter = None
    if args.voxel_sizes:
        try:
            voxel_sizes_filter = [
                float(vs.strip()) for vs in args.voxel_sizes.split(",")
            ]
            logger.info(
                f"Filtering to voxel sizes: "
                f"{', '.join(str(int(vs)) for vs in voxel_sizes_filter)}"
            )
        except ValueError as e:
            logger.error(f"Invalid voxel size format: {e}")
            return

    # Filter results to requested voxel sizes
    try:
        results = filter_voxel_sizes(results, voxel_sizes_filter)
    except ValueError as e:
        logger.error(str(e))
        return

    logger.info(f"Generating comparison plots in {output_dir}")

    create_comparison_plots(
        results=results,
        output_dir=output_dir,
        voxel_sizes_filter=voxel_sizes_filter,
        max_rot_err=args.max_rot_err,
        max_transl_err=args.max_transl_err,
    )

    logger.info("All plots generated successfully")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate comparison plots from localization results across different configurations",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="Base directory containing localization result subdirectories (e.g., output/sequence_registration/localization/comparison/)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="output/localization_comparison_plots",
        help="Output directory for comparison plots",
    )
    parser.add_argument(
        "--voxel-sizes",
        type=str,
        default=None,
        help="Comma-separated list of voxel sizes to plot (e.g., '50,100,450'). If not specified, all voxel sizes are plotted.",
    )
    parser.add_argument(
        "--max-rot-err",
        type=float,
        default=2,
        help="Rotation error threshold (degrees) for the success rate plots. "
        "A scan is counted as successful when both its rotation and translation errors are below their respective thresholds. "
        "If not set, success rate plots are skipped.",
    )
    parser.add_argument(
        "--max-transl-err",
        type=float,
        default=200,
        help="Translation error threshold for the success rate plots. "
        "A scan is counted as successful when both its rotation and translation errors are below their respective thresholds. "
        "If not set, success rate plots are skipped.",
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
