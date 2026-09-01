#!/usr/bin/env python3
"""Generate a markdown report from localization comparison results.

This script reads the output of a localization comparison experiment (produced by
localize_against_map.py and plot_localization_comparison.py) and generates a
structured markdown report suitable for viewing on GitHub or any markdown renderer.

The report includes:
- An overview table with key metrics across all voxel sizes and methods.
- Aggregate comparison plots (all voxel sizes combined).
- Per-voxel-size detail sections (collapsible with HTML details/summary tags).

Usage examples::

    # Single comparison directory:
    python generate_localization_report.py --input comparison_ref_vs10/

    # Top-level directory (one report per comparison_ref_vsXX subfolder + index):
    python generate_localization_report.py --input comparison_fix_ransac/
"""

import argparse
import json
import logging
import re
from datetime import UTC, datetime
from pathlib import Path

from registration.utils.logging import setup_logging

logger = logging.getLogger(__name__)

# Method name mapping from directory prefix to display name.
METHOD_DISPLAY: dict[str, str] = {
    "ransac": "RANSAC only",
    "ransac_icp": "RANSAC + ICP",
    "ransac_gicp": "RANSAC + GICP",
}

# Canonical column order used throughout the report.
METHOD_ORDER: list[str] = ["ransac", "ransac_icp", "ransac_gicp"]

# Default success rate thresholds (matching plot_localization_comparison.py defaults).
DEFAULT_MAX_ROT_ERR_DEG: float = 2.0
DEFAULT_MAX_TRANSL_ERR_MM: float = 200.0

_DIR_PATTERN = re.compile(r"^(ransac(?:_icp|_gicp)?)_vs(\d+)$")
_COMPARISON_DIR_PATTERN = re.compile(r"^comparison_ref_vs(\d+)$")


def parse_method_and_voxel(dirname: str) -> tuple[str, int] | None:
    """Parse the method key and RANSAC voxel size from a directory name.

    Args:
        dirname: Directory name, e.g. 'ransac_icp_vs100'.

    Returns:
        Tuple of (method_key, voxel_size), or None if the name does not match.
    """
    match = _DIR_PATTERN.match(dirname)
    if not match:
        return None
    return match.group(1), int(match.group(2))


def find_method_dirs(comparison_dir: Path) -> dict[int, dict[str, Path]]:
    """Discover method subdirectories grouped by RANSAC voxel size.

    Args:
        comparison_dir: Path to a comparison_ref_vsXX directory.

    Returns:
        Nested dict mapping voxel_size to a dict of method_key -> path.
    """
    result: dict[int, dict[str, Path]] = {}
    for child in sorted(comparison_dir.iterdir()):
        if not child.is_dir():
            continue
        parsed = parse_method_and_voxel(child.name)
        if parsed is None:
            continue
        method_key, voxel_size = parsed
        if voxel_size not in result:
            result[voxel_size] = {}
        result[voxel_size][method_key] = child
    return result


def load_results_json(path: Path) -> dict | None:
    """Load and return a localization_results.json file.

    Args:
        path: Path to the JSON file.

    Returns:
        Parsed JSON dict, or None if the file is missing or malformed.
    """
    if not path.exists():
        logger.warning(f"Missing results file: {path}")
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning(f"Failed to load {path}: {exc}")
        return None


def compute_success_rate(
    rot_errors: list[float],
    transl_errors: list[float],
    max_rot: float,
    max_transl: float,
) -> float:
    """Compute the fraction of scans where both errors are below the thresholds.

    Args:
        rot_errors: Per-scan rotation errors in degrees.
        transl_errors: Per-scan translation errors.
        max_rot: Rotation error threshold in degrees.
        max_transl: Translation error threshold.

    Returns:
        Success rate as a value in [0, 1]. Returns 0.0 for empty inputs.
    """
    if not rot_errors or not transl_errors:
        return 0.0
    n = min(len(rot_errors), len(transl_errors))
    successes = sum(
        r < max_rot and t < max_transl
        for r, t in zip(rot_errors[:n], transl_errors[:n])
    )
    return successes / n


def parse_refinement_voxel(comparison_dir: Path) -> int | None:
    """Extract the refinement voxel size from a comparison directory name.

    Args:
        comparison_dir: Path whose name matches 'comparison_ref_vsXX'.

    Returns:
        Integer voxel size, or None if the name does not match.
    """
    match = _COMPARISON_DIR_PATTERN.match(comparison_dir.name)
    if not match:
        return None
    return int(match.group(1))


def find_comparison_dirs(input_dir: Path) -> list[Path]:
    """Find all comparison_ref_vsXX subdirectories within input_dir.

    Directories are sorted by their numeric voxel size, not lexicographically.

    Args:
        input_dir: Directory to search.

    Returns:
        List of matching subdirectories sorted by ascending voxel size.
    """
    pattern = re.compile(r"^comparison_ref_vs\d+$")
    dirs = [d for d in input_dir.iterdir() if d.is_dir() and pattern.match(d.name)]
    return sorted(dirs, key=lambda d: parse_refinement_voxel(d) or 0)


def _fmt(value: float | None, decimals: int = 2) -> str:
    """Format a float value for display, returning 'N/A' if None.

    Args:
        value: Value to format.
        decimals: Number of decimal places.

    Returns:
        Formatted string.
    """
    if value is None:
        return "N/A"
    return f"{value:.{decimals}f}"


def _get_stat(data: dict, metric: str, stat: str) -> float | None:
    """Safely retrieve a nested statistic from a results dict.

    Args:
        data: Parsed localization_results.json dict.
        metric: Metric key (e.g., 'rotation_error_degrees').
        stat: Statistic key (e.g., 'mean', 'median').

    Returns:
        Float value, or None if the key path is missing.
    """
    try:
        return data["statistics"][metric][stat]
    except (KeyError, TypeError):
        return None


def _relative_if_exists(path: Path, base: Path) -> str | None:
    """Return a relative path string if the file exists, or None.

    Args:
        path: Absolute path to the file.
        base: Base directory for the relative path.

    Returns:
        Relative path string or None.
    """
    if path.exists():
        return str(path.relative_to(base))
    return None


def build_stats_table(
    method_results: dict[str, dict | None],
    max_rot: float,
    max_transl: float,
) -> str:
    """Build a markdown statistics table comparing methods side by side.

    Args:
        method_results: Dict mapping method_key to its localization_results.json data.
        max_rot: Rotation error threshold for success rate.
        max_transl: Translation error threshold for success rate.

    Returns:
        Markdown table string.
    """
    headers = [METHOD_DISPLAY.get(m, m) for m in METHOD_ORDER]
    lines = ["| Metric | " + " | ".join(headers) + " |"]
    lines.append("|---|" + ":-:|" * len(METHOD_ORDER))

    row_specs = [
        ("Rotation error - mean (deg)", "rotation_error_degrees", "mean"),
        ("Rotation error - median (deg)", "rotation_error_degrees", "median"),
        ("Rotation error - std (deg)", "rotation_error_degrees", "std"),
        ("Translation error - mean", "translation_error", "mean"),
        ("Translation error - median", "translation_error", "median"),
        ("Translation error - std", "translation_error", "std"),
        ("Fitness - mean", "fitness", "mean"),
        ("Fitness - median", "fitness", "median"),
        ("Inlier RMSE - mean", "inlier_rmse", "mean"),
        ("Inlier RMSE - median", "inlier_rmse", "median"),
    ]

    for label, metric_key, stat_key in row_specs:
        row = [label]
        for method in METHOD_ORDER:
            data = method_results.get(method)
            val = _get_stat(data, metric_key, stat_key) if data else None
            row.append(_fmt(val))
        lines.append("| " + " | ".join(row) + " |")

    sr_label = f"Success rate (rot<{max_rot}deg, transl<{max_transl})"
    sr_row = [sr_label]
    for method in METHOD_ORDER:
        data = method_results.get(method)
        if data is None:
            sr_row.append("N/A")
        else:
            rot_errors = data.get("rotation_errors", [])
            transl_errors = data.get("translation_errors", [])
            sr = compute_success_rate(rot_errors, transl_errors, max_rot, max_transl)
            sr_row.append(f"{sr * 100:.1f}%")
    lines.append("| " + " | ".join(sr_row) + " |")

    return "\n".join(lines)


def build_overview_table(
    all_data: dict[int, dict[str, dict | None]],
    voxel_sizes: list[int],
    max_rot: float,
    max_transl: float,
) -> str:
    """Build a compact overview table with key metrics for all configurations.

    Each row corresponds to one RANSAC voxel size. Columns group the median
    rotation error, median translation error, and success rate for each method.

    Args:
        all_data: Nested dict of voxel_size -> method_key -> results dict.
        voxel_sizes: Sorted list of RANSAC voxel sizes to include.
        max_rot: Rotation error threshold for success rate.
        max_transl: Translation error threshold for success rate.

    Returns:
        Markdown table string.
    """
    method_labels = [METHOD_DISPLAY.get(m, m) for m in METHOD_ORDER]

    header_parts = ["RANSAC voxel"]
    for label in method_labels:
        header_parts += [
            f"{label}<br>rot. med (deg)",
            f"{label}<br>transl. med",
            f"{label}<br>SR",
        ]

    lines = ["| " + " | ".join(header_parts) + " |"]
    lines.append("|---|" + ":-:|" * (len(METHOD_ORDER) * 3))

    for vs in voxel_sizes:
        method_data = all_data.get(vs, {})
        row = [str(vs)]
        for method in METHOD_ORDER:
            data = method_data.get(method)
            rot_med = (
                _get_stat(data, "rotation_error_degrees", "median") if data else None
            )
            transl_med = (
                _get_stat(data, "translation_error", "median") if data else None
            )
            if data is not None:
                rot_errors = data.get("rotation_errors", [])
                transl_errors = data.get("translation_errors", [])
                sr = compute_success_rate(
                    rot_errors, transl_errors, max_rot, max_transl
                )
                sr_str = f"{sr * 100:.1f}%"
            else:
                sr_str = "N/A"
            row += [_fmt(rot_med), _fmt(transl_med), sr_str]
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


def build_image_table(
    image_paths: list[str | None],
    column_headers: list[str],
) -> str:
    """Build a markdown table row with inline images, one image per column.

    Args:
        image_paths: Relative image paths (None for a missing image).
        column_headers: Column header labels.

    Returns:
        Markdown string for a three-row table (header, separator, images).
    """
    header = "| " + " | ".join(column_headers) + " |"
    separator = "|" + ":-:|" * len(column_headers)
    cells = []
    for path, alt in zip(image_paths, column_headers):
        cells.append(f"![]({path})" if path else f"_{alt} not available_")
    image_row = "| " + " | ".join(cells) + " |"
    return f"{header}\n{separator}\n{image_row}"


def build_per_voxel_section(
    voxel_size: int,
    method_dirs: dict[str, Path],
    comparison_dir: Path,
    max_rot: float,
    max_transl: float,
) -> str:
    """Build a collapsible HTML/markdown section for one RANSAC voxel size.

    Args:
        voxel_size: RANSAC voxel size.
        method_dirs: Dict mapping method_key to its result directory path.
        comparison_dir: Root comparison directory (for relative path computation).
        max_rot: Rotation error threshold for success rate.
        max_transl: Translation error threshold for success rate.

    Returns:
        Markdown string enclosed in HTML <details>/<summary> tags.
    """
    method_results: dict[str, dict | None] = {}
    for key in METHOD_ORDER:
        path = method_dirs.get(key)
        method_results[key] = (
            load_results_json(path / "localization_results.json") if path else None
        )

    header_labels = [METHOD_DISPLAY.get(m, m) for m in METHOD_ORDER]
    stats_table = build_stats_table(method_results, max_rot, max_transl)

    rot_paths = []
    transl_paths = []
    for key in METHOD_ORDER:
        path = method_dirs.get(key)
        if path is not None:
            rot_paths.append(
                _relative_if_exists(
                    path / "plots" / "histogram_rotation_error.png", comparison_dir
                )
            )
            transl_paths.append(
                _relative_if_exists(
                    path / "plots" / "histogram_translation_error.png", comparison_dir
                )
            )
        else:
            rot_paths.append(None)
            transl_paths.append(None)

    rot_table = build_image_table(rot_paths, header_labels)
    transl_table = build_image_table(transl_paths, header_labels)

    comp_plots_dir = comparison_dir / "comparison_plots"
    comp_plots_block = _build_per_voxel_comp_plots(
        comp_plots_dir, voxel_size, comparison_dir
    )

    param_block = _build_param_block(method_dirs, comparison_dir)

    return (
        f"<details>\n"
        f"<summary><strong>RANSAC Voxel Size: {voxel_size}</strong></summary>\n"
        f"\n"
        f"#### Statistics\n"
        f"\n"
        f"{stats_table}\n"
        f"\n"
        f"#### Rotation Error Distributions\n"
        f"\n"
        f"{rot_table}\n"
        f"\n"
        f"#### Translation Error Distributions\n"
        f"\n"
        f"{transl_table}\n"
        f"{comp_plots_block}"
        f"{param_block}\n"
        f"\n"
        f"</details>"
    )


def _build_per_voxel_comp_plots(
    comp_plots_dir: Path, voxel_size: int, comparison_dir: Path
) -> str:
    """Build a markdown block embedding per-voxel comparison plots.

    Args:
        comp_plots_dir: Path to the comparison_plots directory.
        voxel_size: RANSAC voxel size to look up.
        comparison_dir: Root comparison directory for relative paths.

    Returns:
        Markdown string, empty if no plots are found.
    """
    images = [
        (
            comp_plots_dir / f"comparison_rotation_error_{voxel_size}.png",
            "Rotation Error Comparison",
        ),
        (
            comp_plots_dir / f"comparison_translation_error_{voxel_size}.png",
            "Translation Error Comparison",
        ),
        (
            comp_plots_dir / f"comparison_success_rate_{voxel_size}.png",
            "Success Rate Comparison",
        ),
    ]
    parts = []
    for img_path, label in images:
        rel = _relative_if_exists(img_path, comparison_dir)
        if rel:
            parts.append(f"![{label}]({rel})")
    if not parts:
        return ""
    return "\n\n#### Comparison Plots\n\n" + "\n\n".join(parts) + "\n"


def _build_param_block(method_dirs: dict[str, Path], comparison_dir: Path) -> str:
    """Build a bullet list with per-method run parameters.

    Args:
        method_dirs: Dict mapping method_key to its result directory path.
        comparison_dir: Root directory (used only for logging context).

    Returns:
        Markdown string, empty if no parameters files are found.
    """
    notes = []
    for key in METHOD_ORDER:
        path = method_dirs.get(key)
        if path is None:
            continue
        params_path = path / "parameters.json"
        if not params_path.exists():
            continue
        try:
            with open(params_path) as f:
                params = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            logger.debug(f"Could not load {params_path}: {exc}")
            continue
        timestamp = params.get("timestamp", "N/A")
        method_name = params.get("method", "N/A")
        label = METHOD_DISPLAY.get(key, key)
        notes.append(f"- **{label}**: {method_name} (run: {timestamp})")
    if not notes:
        return ""
    return "\n\n#### Run Parameters\n\n" + "\n".join(notes) + "\n"


def embed_aggregate_plots(comparison_dir: Path) -> str:
    """Build a section embedding the aggregate comparison plots.

    Args:
        comparison_dir: Path to the comparison_ref_vsXX directory.

    Returns:
        Markdown string, or a placeholder message if no plots are found.
    """
    plots_dir = comparison_dir / "comparison_plots"
    images = [
        (
            plots_dir / "comparison_rotation_error.png",
            "Rotation Error Comparison (All Voxel Sizes)",
        ),
        (
            plots_dir / "comparison_translation_error.png",
            "Translation Error Comparison (All Voxel Sizes)",
        ),
        (
            plots_dir / "comparison_success_rate.png",
            "Success Rate Comparison (All Voxel Sizes)",
        ),
    ]
    parts = []
    for img_path, label in images:
        rel = _relative_if_exists(img_path, comparison_dir)
        if rel:
            parts.append(f"![{label}]({rel})")
        else:
            logger.debug(f"Aggregate plot not found: {img_path}")
    if not parts:
        return "_No aggregate comparison plots found._"
    return "\n\n".join(parts)


def generate_single_report(
    comparison_dir: Path,
    output_path: Path | None,
    max_rot: float,
    max_transl: float,
) -> None:
    """Generate a markdown report for a single comparison_ref_vsXX directory.

    Args:
        comparison_dir: Path to the comparison_ref_vsXX directory.
        output_path: Destination for the report file. Defaults to
            comparison_dir/report.md.
        max_rot: Rotation error threshold in degrees for success rate.
        max_transl: Translation error threshold for success rate.
    """
    if output_path is None:
        output_path = comparison_dir / "report.md"

    ref_voxel = parse_refinement_voxel(comparison_dir)
    ref_voxel_label = f"{ref_voxel}" if ref_voxel is not None else comparison_dir.name

    logger.info(f"Generating report for {comparison_dir.name} -> {output_path}")

    method_dirs_by_voxel = find_method_dirs(comparison_dir)
    if not method_dirs_by_voxel:
        logger.warning(f"No method directories found in {comparison_dir}")
        return

    voxel_sizes = sorted(method_dirs_by_voxel.keys())

    all_data: dict[int, dict[str, dict | None]] = {
        vs: {
            key: load_results_json(path / "localization_results.json")
            for key, path in method_dirs_by_voxel[vs].items()
        }
        for vs in voxel_sizes
    }

    now = datetime.now(UTC).astimezone().strftime("%Y-%m-%d %H:%M:%S")
    voxel_sizes_str = ", ".join(str(v) for v in voxel_sizes)

    blocks = [
        f"# Localization Comparison: Refinement Voxel Size = {ref_voxel_label}",
        "",
        f"**Generated:** {now}  ",
        f"**Directory:** `{comparison_dir.name}`  ",
        f"**RANSAC voxel sizes tested:** {voxel_sizes_str}  ",
        f"**Success rate thresholds:** rotation < {max_rot} deg, translation < {max_transl}",
        "",
        "---",
        "",
        (
            "This report compares localization accuracy across three methods applied to the "
            "same global map. The coarse registration step uses RANSAC with feature-based "
            "correspondences at varying voxel sizes; the refinement step optionally applies "
            "ICP or Generalized ICP. The three methods are:\n\n"
            "1. **RANSAC only** (no refinement),\n"
            "2. **RANSAC + ICP** (point-to-plane ICP refinement), and\n"
            "3. **RANSAC + GICP** (Generalized ICP refinement).\n\n"
            "The overview table below summarises median errors and success rate for each "
            "RANSAC voxel size. Expand a per-voxel section to see full statistics and "
            "per-scan error histograms."
        ),
        "",
        "## Overview",
        "",
        build_overview_table(all_data, voxel_sizes, max_rot, max_transl),
        "",
        "---",
        "",
        "## Aggregate Comparison Plots",
        "",
        "_These plots compare all RANSAC voxel sizes together for each method._",
        "",
        embed_aggregate_plots(comparison_dir),
        "",
        "---",
        "",
        "## Per-Voxel-Size Details",
        "",
        "_Click a section header to expand results for that RANSAC voxel size._",
        "",
    ]

    for vs in voxel_sizes:
        blocks.append(
            build_per_voxel_section(
                voxel_size=vs,
                method_dirs=method_dirs_by_voxel[vs],
                comparison_dir=comparison_dir,
                max_rot=max_rot,
                max_transl=max_transl,
            )
        )
        blocks.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(blocks), encoding="utf-8")
    logger.info(f"Report written to {output_path}")


def generate_index_report(
    top_dir: Path,
    comparison_dirs: list[Path],
    output_path: Path,
    max_rot: float,
    max_transl: float,
) -> None:
    """Generate a top-level index report linking to per-refinement-voxel reports.

    Args:
        top_dir: The top-level comparison folder.
        comparison_dirs: Sorted list of comparison_ref_vsXX directories.
        output_path: Destination for the index report file.
        max_rot: Rotation error threshold used for individual reports.
        max_transl: Translation error threshold used for individual reports.
    """
    now = datetime.now(UTC).astimezone().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        f"# Localization Comparison Index: {top_dir.name}",
        "",
        f"**Generated:** {now}  ",
        f"**Success rate thresholds:** rotation < {max_rot} deg, translation < {max_transl}",
        "",
        "---",
        "",
        "## Experiment Description",
        "",
        (
            "This experiment evaluates a coarse-to-fine scan localization pipeline "
            "against a pre-built point cloud map. "
            "In the coarse step, RANSAC-based global registration estimates an initial "
            "alignment between the query scan and the map. "
            "In the fine step, the coarse estimate is refined with one of three strategies: "
            "no refinement (RANSAC only), point-to-plane ICP (RANSAC + ICP), or "
            "Generalized ICP (RANSAC + GICP).\n\n"
            "The experiment sweeps over the voxel size used during the refinement step "
            "to study its effect on localization accuracy (rotation error, translation error) "
            "and registration quality (fitness score, RMSE). "
            "Each entry in the table below corresponds to one refinement voxel size; "
            "follow the linked report for per-method comparisons, per-scan statistics, "
            "and embedded diagnostic plots."
        ),
        "",
        "---",
        "",
        "## Refinement Voxel Sizes",
        "",
        "| Refinement Voxel Size | Report |",
        "|---|---|",
    ]
    for d in comparison_dirs:
        ref_voxel = parse_refinement_voxel(d)
        label = f"{ref_voxel}" if ref_voxel is not None else d.name
        rel_report = d.relative_to(top_dir) / "report.md"
        lines.append(f"| {label} | [{d.name}/report.md]({rel_report}) |")

    lines += ["", "---", ""]
    output_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Index report written to {output_path}")


def main() -> None:
    """CLI entry point for generating localization comparison reports."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate markdown reports from localization comparison results. "
            "Accepts either a single comparison_ref_vsXX directory or a top-level "
            "folder containing multiple such directories."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help=(
            "Input directory: either a comparison_ref_vsXX folder (one report) or "
            "a top-level folder with comparison_ref_vsXX subdirectories (one report "
            "per subfolder plus an index report)."
        ),
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help=(
            "Output path for the generated report. In single-directory mode defaults "
            "to <input>/report.md. In multi-directory mode this sets the index report "
            "path (default: <input>/report.md); individual reports are always written "
            "to <comparison_dir>/report.md."
        ),
    )
    parser.add_argument(
        "--max-rot-err",
        type=float,
        default=DEFAULT_MAX_ROT_ERR_DEG,
        help="Rotation error threshold in degrees for success rate computation.",
    )
    parser.add_argument(
        "--max-transl-err",
        type=float,
        default=DEFAULT_MAX_TRANSL_ERR_MM,
        help="Translation error threshold for success rate computation.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Logging level.",
    )

    args = parser.parse_args()
    setup_logging(level=getattr(logging, args.log_level))

    input_dir = Path(args.input)
    if not input_dir.exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        return

    max_rot: float = args.max_rot_err
    max_transl: float = args.max_transl_err

    if _COMPARISON_DIR_PATTERN.match(input_dir.name):
        output_path = Path(args.output) if args.output else None
        generate_single_report(input_dir, output_path, max_rot, max_transl)
    else:
        comparison_dirs = find_comparison_dirs(input_dir)
        if not comparison_dirs:
            logger.warning(
                f"No comparison_ref_vsXX directories found in {input_dir}. "
                "Treating the input directory itself as a comparison directory."
            )
            generate_single_report(input_dir, None, max_rot, max_transl)
            return

        logger.info(f"Found {len(comparison_dirs)} comparison directories")
        for comp_dir in comparison_dirs:
            generate_single_report(comp_dir, None, max_rot, max_transl)

        index_path = Path(args.output) if args.output else input_dir / "report.md"
        generate_index_report(
            input_dir, comparison_dirs, index_path, max_rot, max_transl
        )


if __name__ == "__main__":
    main()
