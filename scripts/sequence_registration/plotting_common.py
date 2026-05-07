"""Common plotting utilities for registration error visualization.

This module provides shared functions for creating error distribution plots
used by validation and localization analysis scripts.
"""

import logging
from pathlib import Path
from typing import List, Dict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter

logger = logging.getLogger(__name__)


def create_histogram(
    values: List[float],
    title: str,
    xlabel: str,
    output_path: Path,
    bins: int = 30,
    color: str = "#3498db",
):
    """Create a histogram plot for a given set of values.

    Args:
        values: List of values to plot.
        title: Plot title.
        xlabel: Label for x-axis.
        output_path: Path where to save the plot.
        bins: Number of histogram bins.
        color: Color for histogram bars.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create histogram
    n, bins_edges, patches = ax.hist(
        values, bins=bins, color=color, alpha=0.7, edgecolor="black", linewidth=0.5
    )

    # Add vertical lines for mean and median
    mean_val = float(np.mean(values))
    median_val = float(np.median(values))

    ax.axvline(
        mean_val,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {mean_val:.2f}",
    )
    ax.axvline(
        median_val,
        color="green",
        linestyle="-.",
        linewidth=2,
        label=f"Median: {median_val:.2f}",
    )

    # Add statistics text box
    stats_text = (
        f"Statistics:\n"
        f"Mean:   {mean_val:.2f}\n"
        f"Median: {median_val:.2f}\n"
        f"Std:    {np.std(values):.2f}\n"
        f"Min:    {np.min(values):.2f}\n"
        f"Max:    {np.max(values):.2f}"
    )

    ax.text(
        0.98,
        0.97,
        stats_text,
        transform=ax.transAxes,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        fontsize=10,
        family="monospace",
    )

    # Labels and title
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(loc="upper left", fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    logger.info(f"Saved histogram to {output_path.name}")
    plt.close()


def create_error_histograms(
    rotation_errors: List[float],
    translation_errors: List[float],
    output_dir: Path,
    rotation_title: str = "Distribution of Rotation Errors",
    translation_title: str = "Distribution of Translation Errors",
):
    """Create histogram plots for rotation and translation errors.

    Args:
        rotation_errors: List of rotation errors in degrees.
        translation_errors: List of translation errors.
        output_dir: Directory where to save the plots.
        rotation_title: Title for rotation error histogram.
        translation_title: Title for translation error histogram.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if not rotation_errors or not translation_errors:
        logger.error("No error data provided")
        return

    # Create rotation error histogram
    logger.info("Creating rotation error histogram...")
    rotation_output = output_dir / "histogram_rotation_error.png"
    create_histogram(
        rotation_errors,
        title=rotation_title,
        xlabel="Rotation Error (degrees)",
        output_path=rotation_output,
        bins=30,
        color="#e74c3c",
    )

    # Create translation error histogram
    logger.info("Creating translation error histogram...")
    translation_output = output_dir / "histogram_translation_error.png"
    create_histogram(
        translation_errors,
        title=translation_title,
        xlabel="Translation Error",
        output_path=translation_output,
        bins=30,
        color="#3498db",
    )


def create_grouped_boxplot(
    data: Dict[str, Dict[str, Dict[str, float]]],
    metric_label: str,
    output_path: Path,
    group_labels: List[str],
    category_labels: List[str],
    colors: Dict[str, str],
    title: str,
):
    """Create a grouped box plot comparing multiple categories across groups.

    Args:
        data: Nested dictionary {group: {category: {stat_name: value}}}.
              Statistics should include 'mean', 'median', 'std', 'min', 'max'.
        metric_label: Label for the y-axis.
        output_path: Path where to save the plot.
        group_labels: Labels for the groups (x-axis).
        category_labels: Labels for the categories (different boxes per group).
        colors: Dictionary mapping category labels to colors.
        title: Plot title.
    """
    fig, ax = plt.subplots(figsize=(14, 7))

    positions = []
    bxp_stats = []
    box_colors = []

    num_categories = len(category_labels)
    group_width = num_categories + 0.5

    for i, group in enumerate(group_labels):
        if group not in data:
            logger.warning(f"Group {group} not found in data")
            continue

        for j, category in enumerate(category_labels):
            if category not in data[group]:
                logger.warning(f"Category {category} not found for group {group}")
                continue

            stats = data[group][category]

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
                "label": category,
            }

            position = i * group_width + j
            positions.append(position)
            bxp_stats.append(box_stats)
            box_colors.append(colors.get(category, "#95a5a6"))

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
    for patch, color in zip(bp["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Set x-axis labels
    tick_positions = [
        i * group_width + (num_categories - 1) / 2 for i in range(len(group_labels))
    ]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(group_labels, rotation=0)

    # Labels and title
    ax.set_ylabel(metric_label, fontsize=12)
    ax.set_xlabel("Voxel Size", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    # Legend
    legend_elements = [
        Rectangle((0, 0), 1, 1, facecolor=colors[cat], alpha=0.7, label=cat)
        for cat in category_labels
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    logger.info(f"Saved grouped box plot to {output_path.name}")
    plt.close()


def create_grouped_barplot(
    data: Dict[str, Dict[str, float]],
    metric_label: str,
    output_path: Path,
    group_labels: List[str],
    category_labels: List[str],
    colors: Dict[str, str],
    title: str,
):
    """Create a grouped bar plot comparing a scalar metric across groups and categories.

    Args:
        data: Nested dict {group_label: {category_label: value}} where values are
              fractions in [0, 1] (displayed as percentages).
        metric_label: Label for the y-axis.
        output_path: Path where to save the plot.
        group_labels: Labels for the groups (x-axis).
        category_labels: Labels for the categories (different bars per group).
        colors: Dictionary mapping category labels to colors.
        title: Plot title.
    """
    fig, ax = plt.subplots(figsize=(14, 7))

    num_categories = len(category_labels)
    group_width = num_categories + 0.5
    bar_width = 0.8

    for i, group in enumerate(group_labels):
        if group not in data:
            logger.warning(f"Group {group} not found in data")
            continue

        for j, category in enumerate(category_labels):
            if category not in data[group]:
                logger.warning(f"Category {category} not found for group {group}")
                continue

            value_pct = data[group][category] * 100.0
            position = i * group_width + j

            ax.bar(
                position,
                value_pct,
                width=bar_width,
                color=colors.get(category, "#95a5a6"),
                alpha=0.7,
                edgecolor="black",
                linewidth=0.5,
            )

            ax.text(
                position,
                value_pct + 0.5,
                f"{value_pct:.1f}%",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    tick_positions = [
        i * group_width + (num_categories - 1) / 2 for i in range(len(group_labels))
    ]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(group_labels, rotation=0)
    ax.set_ylim(0, 115)
    ax.set_ylabel(metric_label, fontsize=12)
    ax.set_xlabel("Voxel Size", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.0f}%"))

    legend_elements = [
        Rectangle((0, 0), 1, 1, facecolor=colors[cat], alpha=0.7, label=cat)
        for cat in category_labels
        if cat in colors
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    logger.info(f"Saved grouped bar plot to {output_path.name}")
    plt.close()
