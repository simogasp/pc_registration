#!/usr/bin/env python3
"""Plot validation statistics comparison across different parameters.

This script reads validation JSON files and creates comparison plots
for different voxel sizes across various step values.
"""

import argparse
import json
import logging
import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from registration.utils.logging import setup_logging

logger = logging.getLogger(__name__)

FILENAME_ROOT = "validation_no-gt_gicp_step"


def parse_filename(filename: str) -> tuple[int, int]:
    """Extract step and voxel size from validation result filename.

    Args:
        filename: Name of the validation JSON file.

    Returns:
        Tuple of (step, voxel_size) extracted from the filename.

    Raises:
        ValueError: If the filename does not match the expected pattern.
    """
    pattern = r"_step(\d+)_vs(\d+)\.json"
    match = re.search(pattern, filename)
    if not match:
        raise ValueError(f"Filename '{filename}' does not match expected pattern")

    step = int(match.group(1))
    voxel_size = int(match.group(2))
    return step, voxel_size


def load_validation_results(directory: Path) -> dict[tuple[int, int], dict]:
    """Load all validation JSON files from a directory.

    Args:
        directory: Path to directory containing validation JSON files.

    Returns:
        Dictionary mapping (step, voxel_size) tuples to their statistics.
    """
    results: dict[tuple[int, int], dict] = {}

    for json_file in directory.glob("*_step*_vs*.json"):
        try:
            step, voxel_size = parse_filename(json_file.name)

            with open(json_file, "r") as f:
                data = json.load(f)

            results[(step, voxel_size)] = data["summary_statistics"]
            logger.info(
                f"Loaded {json_file.name}: step={step}, voxel_size={voxel_size}"
            )

        except (ValueError, KeyError, json.JSONDecodeError) as e:
            logger.warning(f"Skipping {json_file.name}: {e}")

    return results


def extract_metric_data(
    results: dict[tuple[int, int], dict], metric_name: str
) -> dict[int, dict[int, dict[str, float]]]:
    """Extract data for a specific metric organized by step and voxel size.

    Args:
        results: Dictionary of validation results.
        metric_name: Name of the metric to extract (e.g., 'rotation_error_degrees').

    Returns:
        Nested dictionary: {step: {voxel_size: {stat_name: value}}}
    """
    organized_data = {}

    for (step, voxel_size), stats in results.items():
        if step not in organized_data:
            organized_data[step] = {}

        organized_data[step][voxel_size] = stats[metric_name]

    return organized_data


def create_comparison_boxplot(
    data: dict[int, dict[int, dict[str, float]]],
    metric_name: str,
    metric_label: str,
    output_path: Path,
):
    """Create a box plot comparing voxel sizes across different steps.

    Args:
        data: Organized data for the metric.
        metric_name: Name of the metric being plotted.
        metric_label: Label for the y-axis.
        output_path: Path where to save the plot.
    """
    steps = sorted(data.keys())
    voxel_sizes = sorted({vs for step_data in data.values() for vs in step_data.keys()})

    _, ax = plt.subplots(figsize=(12, 6))

    # Prepare box plot data using pre-computed statistics
    positions = []
    bxp_stats = []
    colors = []

    palette = plt.colormaps["tab10"]
    color_map = {
        vs: palette(i / max(len(voxel_sizes) - 1, 1))
        for i, vs in enumerate(voxel_sizes)
    }

    for i, step in enumerate(steps):
        for j, voxel_size in enumerate(voxel_sizes):
            if voxel_size not in data[step]:
                continue

            stats = data[step][voxel_size]

            # Calculate Q1 and Q3 from mean and std (assuming normal distribution)
            q1 = stats["mean"] - 0.675 * stats["std"]
            q3 = stats["mean"] + 0.675 * stats["std"]

            # Ensure proper ordering
            q1 = max(stats["min"], min(q1, stats["median"]))
            q3 = min(stats["max"], max(q3, stats["median"]))

            # Create box plot statistics dictionary
            box_stats = {
                "med": stats["median"],
                "q1": q1,
                "q3": q3,
                "whislo": stats["min"],
                "whishi": stats["max"],
                "mean": stats["mean"],
                "label": f"VS={voxel_size}",
            }

            logger.debug(
                f"{metric_name} Step {step}, Voxel Size {voxel_size}: {box_stats}"
            )

            position = i * (len(voxel_sizes) + 0.5) + j
            positions.append(position)
            bxp_stats.append(box_stats)
            colors.append(color_map.get(voxel_size, "#95a5a6"))

    # Create box plot using pre-computed statistics
    bp = ax.bxp(
        bxp_stats,
        positions=positions,
        widths=0.6,
        patch_artist=True,
        showfliers=False,
        showmeans=False,
    )

    # Color the boxes
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Set x-axis labels
    tick_positions = [
        i * (len(voxel_sizes) + 0.5) + (len(voxel_sizes) - 1) / 2
        for i in range(len(steps))
    ]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([f"Step {step}" for step in steps])

    # Labels and title
    ax.set_ylabel(metric_label)
    ax.set_xlabel("Step Size")
    vs_labels = " vs ".join(str(int(vs)) for vs in voxel_sizes)
    ax.set_title(f"{metric_label} Comparison: Voxel Size {vs_labels}")
    ax.grid(True, alpha=0.3, axis="y")

    # Legend
    legend_elements = [
        Rectangle(
            (0, 0),
            1,
            1,
            facecolor=color_map[vs],
            alpha=0.7,
            label=f"Voxel Size {int(vs)}",
        )
        for vs in voxel_sizes
    ]
    ax.legend(handles=legend_elements, loc="upper right")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    logger.info(f"Saved plot to {output_path}")
    plt.close()


def create_all_plots(results: dict[tuple[int, int], dict], output_dir: Path):
    """Create all comparison plots for different metrics.

    Args:
        results: Dictionary of validation results.
        output_dir: Directory where to save the plots.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = [
        ("rotation_error_degrees", "Rotation Error (degrees)"),
        ("translation_error", "Translation Error"),
        ("fitness", "Registration Fitness"),
        ("rmse", "Registration RMSE"),
    ]

    for metric_name, metric_label in metrics:
        logger.info(f"Creating plot for {metric_name}...")
        data = extract_metric_data(results, metric_name)

        output_path = output_dir / f"comparison_{metric_name}.png"
        create_comparison_boxplot(data, metric_name, metric_label, output_path)


def main(args: argparse.Namespace):
    """Main function to generate validation comparison plots.

    Args:
        args: Namespace object containing command-line arguments.
    """
    input_dir = Path(args.input)
    output_dir = Path(args.output)

    if not input_dir.exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        return

    logger.info(f"Loading validation results from {input_dir}")
    results = load_validation_results(input_dir)

    if not results:
        logger.error("No validation results found")
        return

    logger.info(f"Found {len(results)} validation result files")
    logger.info(f"Generating comparison plots in {output_dir}")

    create_all_plots(results, output_dir)

    logger.info("All plots generated successfully")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate comparison plots from validation results"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Directory containing validation JSON files",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output/validation_plots",
        help="Output directory for plots (default: output/validation_plots)",
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
