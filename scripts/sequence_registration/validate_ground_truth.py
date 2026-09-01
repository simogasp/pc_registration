#! /usr/bin/env python3
"""Validate ground truth transformations by comparing with pairwise ICP registration.

This script performs sanity checks on ground truth transformations by:
1. Running pairwise registration between consecutive scans
2. Computing the relative ground truth pose from H matrices
3. Comparing registration results with ground truth
4. Reporting statistics and errors for each pair
"""

import argparse
import json
import logging

# Add scripts directory to path for imports
import sys
from pathlib import Path

import numpy as np

from registration.utils.logging import setup_logging

scripts_dir = Path(__file__).parent
if str(scripts_dir) not in sys.path:
    sys.path.insert(0, str(scripts_dir))

from registration_common import (
    find_scan_pairs,
    load_point_cloud,
    load_transformation_matrix,
    pairwise_registration,
    rotation_error_degrees,
    translation_error,
)

logger = logging.getLogger(__name__)


def compute_relative_pose(H_source: np.ndarray, H_target: np.ndarray) -> np.ndarray:
    """Compute relative pose from source to target.

    Given absolute poses H_source and H_target in world frame,
    compute the transformation that brings source to target frame.

    Args:
        H_source: 4x4 transformation matrix of source scan (world frame).
        H_target: 4x4 transformation matrix of target scan (world frame).

    Returns:
        4x4 relative transformation matrix: H_rel = inv(H_target) @ H_source
    """
    return np.linalg.inv(H_target) @ H_source


def format_transformation_matrix(H: np.ndarray, indent: int = 4) -> str:
    """Format a 4x4 transformation matrix as a readable string.

    Args:
        H: 4x4 transformation matrix.
        indent: Number of spaces for indentation.

    Returns:
        Formatted string.
    """
    lines = []
    prefix = " " * indent
    for row in H:
        formatted_row = " ".join([f"{val:8.4f}" for val in row])
        lines.append(f"{prefix}[{formatted_row}]")
    return "\n".join(lines)


def validate_consecutive_pairs(
    data_dir: str,
    voxel_size: float = 50.0,
    max_correspondence_distance: float = 150.0,
    use_ground_truth_init: bool = True,
    output_file: str | None = None,
    start_scan: int | None = None,
    end_scan: int | None = None,
    step: int = 1,
    use_generalized_icp: bool = False,
):
    """Validate ground truth by comparing with pairwise registration.

    Args:
        data_dir: Directory containing .ply and .json scan pairs.
        voxel_size: Voxel size for downsampling input scans (0 = no downsampling).
        max_correspondence_distance: Maximum correspondence distance for ICP.
        use_ground_truth_init: If True, use ground truth as initialization for ICP.
        output_file: Optional file to save results (JSON format).
        start_scan: Index of first scan to process (0-based, inclusive). None = start from 0.
        end_scan: Index of last scan to process (0-based, inclusive). None = process until end.
        step: Step size for selecting pairs (1 = consecutive, 2 = every other scan, etc.).
        use_generalized_icp: If True, use GICP; otherwise use classic ICP.
    """
    data_path = Path(data_dir)

    # Validate input directory
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    # Find all scan pairs
    pairs = find_scan_pairs(data_path)

    if len(pairs) < 2:
        raise ValueError("Need at least 2 scans for consecutive pair validation")

    # Validate and apply scan range
    total_scans = len(pairs)
    start_idx = start_scan if start_scan is not None else 0
    end_idx = end_scan if end_scan is not None else total_scans - 1

    # Validation checks
    if start_idx < 0 or start_idx >= total_scans:
        raise ValueError(
            f"start_scan ({start_idx}) must be in range [0, {total_scans - 1}]"
        )
    if end_idx < 0 or end_idx >= total_scans:
        raise ValueError(
            f"end_scan ({end_idx}) must be in range [0, {total_scans - 1}]"
        )
    if start_idx > end_idx:
        raise ValueError(f"start_scan ({start_idx}) must be <= end_scan ({end_idx})")

    # Slice pairs (inclusive range)
    pairs = pairs[start_idx : end_idx + 1]
    num_scans = len(pairs)

    # Calculate number of pairs with given step
    num_pairs = len([i for i in range(num_scans) if i + step < num_scans])

    if num_scans < step + 1:
        raise ValueError(
            f"Need at least {step + 1} scans in the selected range for step={step} validation"
        )

    logger.info(
        f"Processing scans {start_idx} to {end_idx} (inclusive) - {num_scans} scans out of {total_scans} total"
    )
    logger.info(f"Validating {num_pairs} pairs with step={step}")
    if step == 1:
        logger.info("  (consecutive pairs: i and i+1)")
    else:
        logger.info(f"  (pairs with step: i and i+{step})")
    logger.info(f"Voxel size: {voxel_size}")
    logger.info(f"Max correspondence distance: {max_correspondence_distance}")
    logger.info(f"Use ground truth initialization: {use_ground_truth_init}")
    logger.info("=" * 80)

    # Statistics accumulators
    rotation_errors = []
    translation_errors = []
    fitness_scores = []
    rmse_values = []

    # Results for optional JSON output
    results = []

    # Process pairs with given step
    pair_count = 0
    for i, (source_ply, source_json) in enumerate(pairs):
        # Check if target index is within range
        target_idx = i + step
        if target_idx >= len(pairs):
            break

        target_ply, target_json = pairs[target_idx]

        logger.info(
            f"\nPair {pair_count}/{num_pairs - 1}: {source_ply.stem} -> {target_ply.stem} (step={step})"
        )
        logger.info("-" * 80)
        pair_count += 1

        # Load point clouds
        logger.info("  Loading point clouds...")
        source_pcd = load_point_cloud(source_ply, voxel_size=voxel_size)
        target_pcd = load_point_cloud(target_ply, voxel_size=voxel_size)
        logger.info(
            f"    Source: {len(source_pcd.points)} points, "
            f"Target: {len(target_pcd.points)} points"
        )

        # Load ground truth poses
        H_source = load_transformation_matrix(source_json)
        H_target = load_transformation_matrix(target_json)

        # Compute relative ground truth
        H_rel_gt = compute_relative_pose(H_source, H_target)

        # Perform registration
        logger.info("  Running ICP registration...")
        init_transform = H_rel_gt if use_ground_truth_init else np.eye(4)

        H_rel_icp, reg_result = pairwise_registration(
            source_pcd,
            target_pcd,
            max_correspondence_distance,
            init_transformation=init_transform,
            max_iteration=50,
            verbose=False,
            use_generalized_icp=use_generalized_icp,
        )

        # Extract rotation and translation
        R_gt = H_rel_gt[:3, :3]
        t_gt = H_rel_gt[:3, 3]
        R_icp = H_rel_icp[:3, :3]
        t_icp = H_rel_icp[:3, 3]

        # Compute errors
        rot_error = rotation_error_degrees(R_gt, R_icp)
        trans_error = translation_error(t_gt, t_icp)

        # Store statistics
        rotation_errors.append(rot_error)
        translation_errors.append(trans_error)
        fitness_scores.append(reg_result.fitness)
        rmse_values.append(reg_result.inlier_rmse)

        # Print results
        logger.info("  Registration Statistics:")
        logger.info(f"    Fitness:      {reg_result.fitness:.4f}")
        logger.info(f"    Inlier RMSE:  {reg_result.inlier_rmse:.4f}")

        logger.info("  Error vs Ground Truth:")
        logger.info(f"    Rotation error:    {rot_error:.4f}°")
        logger.info(f"    Translation error: {trans_error:.4f}")

        # Optional: print transformations for debugging
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("  Ground Truth Relative Pose (source -> target):")
            logger.debug(f"\n{format_transformation_matrix(H_rel_gt)}")
            logger.debug("  ICP Registration Result:")
            logger.debug(f"\n{format_transformation_matrix(H_rel_icp)}")

        # Store result for JSON output
        result_entry = {
            "pair_index": pair_count - 1,
            "source_index": i,
            "target_index": target_idx,
            "source": source_ply.stem,
            "target": target_ply.stem,
            "step": step,
            "registration": {
                "fitness": float(reg_result.fitness),
                "inlier_rmse": float(reg_result.inlier_rmse),
            },
            "errors": {
                "rotation_degrees": float(rot_error),
                "translation": float(trans_error),
            },
            "ground_truth_relative_pose": H_rel_gt.tolist(),
            "icp_relative_pose": H_rel_icp.tolist(),
        }
        results.append(result_entry)

    # Print summary statistics
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY STATISTICS")
    logger.info("=" * 80)

    logger.info("Rotation Error (degrees):")
    logger.info(f"  Mean:   {np.mean(rotation_errors):.4f}°")
    logger.info(f"  Median: {np.median(rotation_errors):.4f}°")
    logger.info(f"  Std:    {np.std(rotation_errors):.4f}°")
    logger.info(f"  Min:    {np.min(rotation_errors):.4f}°")
    logger.info(f"  Max:    {np.max(rotation_errors):.4f}°")

    logger.info("Translation Error:")
    logger.info(f"  Mean:   {np.mean(translation_errors):.4f}")
    logger.info(f"  Median: {np.median(translation_errors):.4f}")
    logger.info(f"  Std:    {np.std(translation_errors):.4f}")
    logger.info(f"  Min:    {np.min(translation_errors):.4f}")
    logger.info(f"  Max:    {np.max(translation_errors):.4f}")

    logger.info("Registration Fitness:")
    logger.info(f"  Mean:   {np.mean(fitness_scores):.4f}")
    logger.info(f"  Median: {np.median(fitness_scores):.4f}")
    logger.info(f"  Std:    {np.std(fitness_scores):.4f}")
    logger.info(f"  Min:    {np.min(fitness_scores):.4f}")
    logger.info(f"  Max:    {np.max(fitness_scores):.4f}")

    logger.info("Registration RMSE:")
    logger.info(f"  Mean:   {np.mean(rmse_values):.4f}")
    logger.info(f"  Median: {np.median(rmse_values):.4f}")
    logger.info(f"  Std:    {np.std(rmse_values):.4f}")
    logger.info(f"  Min:    {np.min(rmse_values):.4f}")
    logger.info(f"  Max:    {np.max(rmse_values):.4f}")

    # Save results to JSON if requested
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        summary = {
            "dataset": str(data_path),
            "num_pairs": len(results),
            "settings": {
                "voxel_size": voxel_size,
                "max_correspondence_distance": max_correspondence_distance,
                "use_ground_truth_init": use_ground_truth_init,
                "use_generalized_icp": use_generalized_icp,
                "step": step,
                "start_scan": start_idx,
                "end_scan": end_idx,
                "total_scans_available": total_scans,
            },
            "summary_statistics": {
                "rotation_error_degrees": {
                    "mean": float(np.mean(rotation_errors)),
                    "median": float(np.median(rotation_errors)),
                    "std": float(np.std(rotation_errors)),
                    "min": float(np.min(rotation_errors)),
                    "max": float(np.max(rotation_errors)),
                },
                "translation_error": {
                    "mean": float(np.mean(translation_errors)),
                    "median": float(np.median(translation_errors)),
                    "std": float(np.std(translation_errors)),
                    "min": float(np.min(translation_errors)),
                    "max": float(np.max(translation_errors)),
                },
                "fitness": {
                    "mean": float(np.mean(fitness_scores)),
                    "median": float(np.median(fitness_scores)),
                    "std": float(np.std(fitness_scores)),
                    "min": float(np.min(fitness_scores)),
                    "max": float(np.max(fitness_scores)),
                },
                "rmse": {
                    "mean": float(np.mean(rmse_values)),
                    "median": float(np.median(rmse_values)),
                    "std": float(np.std(rmse_values)),
                    "min": float(np.min(rmse_values)),
                    "max": float(np.max(rmse_values)),
                },
            },
            "pairs": results,
        }

        with open(output_path, "w") as f:
            json.dump(summary, f, indent=2)

        logger.info(f"✓ Results saved to: {output_path}")

    logger.info("\n" + "=" * 80)
    logger.info("✓ Validation complete!")
    logger.info("=" * 80)


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Validate ground truth transformations by comparing with pairwise ICP registration",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Input directory containing .ply and .json scan pairs",
    )

    parser.add_argument(
        "--voxel-size",
        type=float,
        default=50.0,
        help="Voxel size for downsampling input scans (0 = no downsampling)",
    )

    parser.add_argument(
        "--max-correspondence-distance",
        type=float,
        default=150.0,
        help="Maximum correspondence distance for ICP registration",
    )

    parser.add_argument(
        "--no-gt-init",
        action="store_true",
        help="Do not use ground truth as initialization (use identity instead)",
    )

    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Optional output JSON file to save detailed results",
    )

    parser.add_argument(
        "--start-scan",
        type=int,
        default=None,
        help="Index of first scan to process (0-based, inclusive)",
    )

    parser.add_argument(
        "--end-scan",
        type=int,
        default=None,
        help="Index of last scan to process (0-based, inclusive)",
    )

    parser.add_argument(
        "--step",
        type=int,
        default=1,
        help="Step size for selecting pairs (1=consecutive, 2=every other scan, etc.)",
    )

    parser.add_argument(
        "--use-gicp",
        action="store_true",
        help="Use Generalized ICP (GICP) instead of classic ICP for registration",
    )

    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set logging level (default: INFO)",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(level=getattr(logging, args.log_level))

    # Validate step parameter
    if args.step < 1:
        parser.error("--step must be a positive integer (>= 1)")

    try:
        validate_consecutive_pairs(
            data_dir=args.input,
            voxel_size=args.voxel_size,
            max_correspondence_distance=args.max_correspondence_distance,
            use_ground_truth_init=not args.no_gt_init,
            output_file=args.output,
            start_scan=args.start_scan,
            end_scan=args.end_scan,
            step=args.step,
            use_generalized_icp=args.use_gicp,
        )
    except Exception as e:
        logger.error(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()
