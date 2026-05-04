#! /usr/bin/env python3
"""Fuse multiple point cloud scans using ground truth transformations.

This script reads a folder containing paired .ply and .json files, applies the
ground truth transformations from the JSON files to each point cloud, and creates
two output maps:
1. Raw map: concatenation of all transformed point clouds (binary PLY)
2. Fused map: merged point clouds with voxel downsampling (binary PLY)
"""

import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Optional

import open3d as o3d

from registration.utils.logging import setup_logging

# Add scripts directory to path for imports
import sys

scripts_dir = Path(__file__).parent
if str(scripts_dir) not in sys.path:
    sys.path.insert(0, str(scripts_dir))

from registration_common import (  # noqa: E402
    find_scan_pairs,
    load_transformation_matrix,
    load_and_transform_scan,
    remove_outliers,
    filter_distant_points,
    save_point_cloud_binary,
    save_parameters,
)

logger = logging.getLogger(__name__)


def process_all_scans(pairs: List[Tuple[Path, Path]]) -> List[o3d.geometry.PointCloud]:
    """Load and transform all scan pairs.

    Args:
        pairs: List of (ply_path, json_path) tuples.

    Returns:
        List of transformed point clouds.
    """
    transformed_scans = []

    for i, (ply_path, json_path) in enumerate(pairs):
        logger.info(f"Processing scan {i + 1}/{len(pairs)}: {ply_path.name}")

        try:
            # Load transformation matrix
            H = load_transformation_matrix(json_path)

            # Load and transform point cloud
            pcd = load_and_transform_scan(ply_path, H)
            transformed_scans.append(pcd)

        except Exception as e:
            logger.error(f"Failed to process {ply_path.name}: {e}")
            raise

    return transformed_scans


def create_raw_map(
    transformed_scans: List[o3d.geometry.PointCloud],
) -> o3d.geometry.PointCloud:
    """Create raw map by concatenating all transformed point clouds.

    Args:
        transformed_scans: List of transformed point clouds.

    Returns:
        Combined point cloud containing all scans.
    """
    combined_pcd = o3d.geometry.PointCloud()

    for pcd in transformed_scans:
        combined_pcd += pcd

    logger.info(
        f"Created raw map with {len(combined_pcd.points)} points from "
        f"{len(transformed_scans)} scans"
    )

    return combined_pcd


def create_fused_map(
    transformed_scans: List[o3d.geometry.PointCloud], voxel_size: float = 10.0
) -> o3d.geometry.PointCloud:
    """Create fused map by merging transformed scans with voxel downsampling.

    Args:
        transformed_scans: List of transformed point clouds.
        voxel_size: Voxel size for downsampling (default: 10.0 mm).

    Returns:
        Fused and downsampled point cloud.
    """
    # First concatenate all scans
    combined_pcd = create_raw_map(transformed_scans)

    # Downsample to fuse overlapping points
    logger.info(f"Downsampling with voxel size {voxel_size}...")
    fused_pcd = combined_pcd.voxel_down_sample(voxel_size=voxel_size)

    logger.info(
        f"Created fused map with {len(fused_pcd.points)} points "
        f"(reduction: {len(combined_pcd.points) - len(fused_pcd.points)} points)"
    )

    return fused_pcd


def fuse_scans(
    data_dir: str,
    output_dir: str,
    voxel_size: float = 10.0,
    skip_raw: bool = False,
    skip_fused: bool = False,
    remove_outliers_flag: bool = False,
    outlier_nb_neighbors: int = 20,
    outlier_std_ratio: float = 2.0,
    filter_distant_flag: bool = False,
    max_distance: Optional[float] = None,
    distance_percentile: float = 99.0,
    start_scan: Optional[int] = None,
    end_scan: Optional[int] = None,
    step: int = 1,
):
    """Main function to fuse multiple scans into map files.

    Args:
        data_dir: Directory containing .ply and .json scan pairs.
        output_dir: Directory where output maps will be saved.
        voxel_size: Voxel size for downsampling in fused map (mm).
        skip_raw: If True, skip saving raw concatenated map.
        skip_fused: If True, skip saving fused map.
        remove_outliers_flag: If True, remove outliers from both raw and fused maps.
        outlier_nb_neighbors: Number of neighbors for statistical outlier removal.
        outlier_std_ratio: Standard deviation ratio threshold for outlier removal.
        filter_distant_flag: If True, filter out points that are too far from centroid.
        max_distance: Maximum distance from centroid (mm). If None, uses percentile.
        distance_percentile: Distance percentile threshold (default: 99.0).
        start_scan: Index of first scan to process (0-based, inclusive). None = start from 0.
        end_scan: Index of last scan to process (0-based, inclusive). None = process until end.
        step: Process every nth scan within the selected range (default: 1 = all scans).
    """
    data_path = Path(data_dir)
    output_path = Path(output_dir)

    # Validate input directory
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_path}")

    # Find all scan pairs
    pairs = find_scan_pairs(data_path)

    if not pairs:
        raise ValueError(f"No matching .ply/.json pairs found in {data_dir}")

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

    if step < 1:
        raise ValueError(f"step must be >= 1, got {step}")

    # Slice pairs: inclusive range, then apply step
    pairs = pairs[start_idx : end_idx + 1 : step]
    num_scans = len(pairs)

    logger.info(
        f"Processing scans {start_idx} to {end_idx} (step={step}) - "
        f"{num_scans} scans out of {total_scans} total"
    )

    # Save execution parameters
    params = {
        "timestamp": datetime.now().isoformat(),
        "data_dir": str(data_path),
        "output_dir": str(output_path),
        "voxel_size": voxel_size,
        "skip_raw": skip_raw,
        "skip_fused": skip_fused,
        "remove_outliers": remove_outliers_flag,
        "outlier_nb_neighbors": outlier_nb_neighbors,
        "outlier_std_ratio": outlier_std_ratio,
        "filter_distant": filter_distant_flag,
        "max_distance": max_distance,
        "distance_percentile": distance_percentile,
        "start_scan_requested": start_scan,
        "end_scan_requested": end_scan,
        "step": step,
        "start_scan_actual": start_idx,
        "end_scan_actual": end_idx,
        "total_scans_available": total_scans,
        "num_scans_processed": num_scans,
    }
    params_output = output_path / "parameters.json"
    save_parameters(params, params_output)

    # Process all scans
    logger.info("=" * 60)
    logger.info("Loading and transforming scans...")
    logger.info("=" * 60)
    transformed_scans = process_all_scans(pairs)

    # Save raw map
    if not skip_raw:
        logger.info("=" * 60)
        logger.info("Creating raw map...")
        logger.info("=" * 60)
        raw_map = create_raw_map(transformed_scans)

        # Filter distant points if requested
        if filter_distant_flag:
            raw_map = filter_distant_points(
                raw_map, max_distance=max_distance, percentile=distance_percentile
            )

        # Remove outliers if requested
        if remove_outliers_flag:
            raw_map = remove_outliers(
                raw_map, nb_neighbors=outlier_nb_neighbors, std_ratio=outlier_std_ratio
            )

        raw_output = output_path / "raw_map.ply"
        save_point_cloud_binary(raw_map, raw_output)

    # Save fused map
    if not skip_fused:
        logger.info("=" * 60)
        logger.info("Creating fused map...")
        logger.info("=" * 60)
        fused_map = create_fused_map(transformed_scans, voxel_size=voxel_size)

        # Filter distant points if requested
        if filter_distant_flag:
            fused_map = filter_distant_points(
                fused_map, max_distance=max_distance, percentile=distance_percentile
            )

        # Remove outliers if requested
        if remove_outliers_flag:
            fused_map = remove_outliers(
                fused_map,
                nb_neighbors=outlier_nb_neighbors,
                std_ratio=outlier_std_ratio,
            )

        fused_output = output_path / "fused_map.ply"
        save_point_cloud_binary(fused_map, fused_output)

    logger.info("=" * 60)
    logger.info("✓ Fusion complete!")
    logger.info("=" * 60)


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Fuse multiple point cloud scans using ground truth transformations",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Input directory containing .ply and .json scan pairs",
    )

    parser.add_argument(
        "--output",
        "-o",
        default="output/fused_scans",
        help="Output directory for fused maps",
    )

    parser.add_argument(
        "--voxel-size",
        type=float,
        default=10.0,
        help="Voxel size (mm) for downsampling in fused map",
    )

    parser.add_argument(
        "--skip-raw",
        action="store_true",
        help="Skip saving raw concatenated map",
    )

    parser.add_argument(
        "--skip-fused",
        action="store_true",
        help="Skip saving fused map",
    )

    parser.add_argument(
        "--remove-outliers",
        action="store_true",
        help="Remove outlier points from both raw and fused maps",
    )

    parser.add_argument(
        "--outlier-nb-neighbors",
        type=int,
        default=20,
        help="Number of neighbors to consider for outlier removal",
    )

    parser.add_argument(
        "--outlier-std-ratio",
        type=float,
        default=2.0,
        help="Standard deviation ratio threshold for outlier removal",
    )

    parser.add_argument(
        "--filter-distant",
        action="store_true",
        help="Filter out points that are too far from the point cloud centroid",
    )

    parser.add_argument(
        "--max-distance",
        type=float,
        default=None,
        help="Maximum distance (mm) from centroid for filtering. If not specified, uses percentile",
    )

    parser.add_argument(
        "--distance-percentile",
        type=float,
        default=99.0,
        help="Distance percentile threshold for filtering (only used if --max-distance not specified)",
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
        help="Process every nth scan within the selected range (default: 1 = all scans)",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging()

    try:
        fuse_scans(
            data_dir=args.input,
            output_dir=args.output,
            voxel_size=args.voxel_size,
            skip_raw=args.skip_raw,
            skip_fused=args.skip_fused,
            remove_outliers_flag=args.remove_outliers,
            outlier_nb_neighbors=args.outlier_nb_neighbors,
            outlier_std_ratio=args.outlier_std_ratio,
            filter_distant_flag=args.filter_distant,
            max_distance=args.max_distance,
            distance_percentile=args.distance_percentile,
            start_scan=args.start_scan,
            end_scan=args.end_scan,
            step=args.step,
        )
    except Exception as e:
        logger.error(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()
