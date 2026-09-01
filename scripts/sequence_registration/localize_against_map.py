#!/usr/bin/env python3
"""Localize scans against a global map using RANSAC-based global registration.

This script performs localization by registering each scan against a global map:
1. Load a global map (e.g., fused_map_optimized.ply)
2. For each scan in the dataset, estimate its pose using RANSAC-based global registration
3. Compare estimated poses with ground truth
4. Report statistics and errors for each scan
5. Save detailed results to JSON file

The registration uses FPFH features and RANSAC for robust feature matching,
which does not require an initial pose guess.
"""

import argparse
import json
import logging
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import open3d as o3d

from registration.utils.logging import setup_logging

# Add scripts directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from registration_common import (
    find_scan_pairs,
    load_point_cloud,
    load_poses_from_file,
    load_transformation_matrix,
    pairwise_registration,
    rotation_error_degrees,
    save_poses_to_file,
    translation_error,
)

logger = logging.getLogger(__name__)


def compute_fpfh_features(
    pcd: o3d.geometry.PointCloud,
    voxel_size: float,
) -> o3d.pipelines.registration.Feature:
    """Compute FPFH features for a point cloud.

    Args:
        pcd: Input point cloud (must have normals).
        voxel_size: Voxel size used for radius estimation.

    Returns:
        FPFH feature object.
    """
    radius_normal = voxel_size * 2
    radius_feature = voxel_size * 5

    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
    )

    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100),
    )
    return fpfh


def global_localize_scan_to_map(
    scan: o3d.geometry.PointCloud,
    global_map: o3d.geometry.PointCloud,
    voxel_size: float,
    max_correspondence_distance: float,
) -> tuple:
    """Localize a scan against a global map using RANSAC-based global registration.

    This function performs feature-based global registration without requiring
    an initial transformation guess. Assumes input point clouds are already
    downsampled.

    Args:
        scan: Input scan point cloud (should be downsampled).
        global_map: Global map point cloud (should be downsampled).
        voxel_size: Voxel size for feature computation radius.
        max_correspondence_distance: Maximum correspondence distance for RANSAC.

    Returns:
        Tuple of (estimated_pose, registration_result).
    """
    logger.debug("Computing FPFH features for scan...")
    scan_fpfh = compute_fpfh_features(scan, voxel_size)

    logger.debug("Computing FPFH features for map...")
    map_fpfh = compute_fpfh_features(global_map, voxel_size)

    logger.debug("Performing RANSAC-based global registration...")
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        scan,
        global_map,
        scan_fpfh,
        map_fpfh,
        mutual_filter=True,
        max_correspondence_distance=max_correspondence_distance,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(
            False
        ),
        ransac_n=3,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                max_correspondence_distance
            ),
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(
            max_iteration=100000, confidence=0.999
        ),
    )

    return result.transformation, result


def localize_scan_to_map(
    scan: o3d.geometry.PointCloud,
    global_map: o3d.geometry.PointCloud,
    max_correspondence_distance: float,
    init_transformation: np.ndarray = np.eye(4),
    max_iteration: int = 50,
) -> tuple:
    """Localize a scan against a global map using ICP.

    Args:
        scan: Input scan point cloud.
        global_map: Global map point cloud.
        max_correspondence_distance: Maximum correspondence distance for ICP.
        init_transformation: Initial pose guess.
        max_iteration: Maximum ICP iterations.

    Returns:
        Tuple of (estimated_pose, registration_result).
    """
    result = o3d.pipelines.registration.registration_icp(
        scan,
        global_map,
        max_correspondence_distance,
        init_transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iteration),
    )

    return result.transformation, result


def save_parameters(params: dict, output_path: Path):
    """Save execution parameters to a JSON file.

    Args:
        params: Dictionary of parameters used for the execution.
        output_path: Output file path.
    """
    with open(output_path, "w") as f:
        json.dump(params, f, indent=2)

    logger.info(f"Saved execution parameters to {output_path.name}")


def compute_stats(values: list[float]) -> dict:
    """Compute summary statistics for a list of numeric values.

    Args:
        values: List of numeric values to summarise.

    Returns:
        Dictionary with keys: mean, median, std, min, max.
    """
    arr = np.array(values)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def _needs_separate_refinement_clouds(
    refinement_voxel_size: float | None,
    ransac_voxel_size: float,
) -> bool:
    """Return True if refinement requires loading clouds at a different resolution.

    Args:
        refinement_voxel_size: Voxel size for refinement, or None to reuse RANSAC clouds.
        ransac_voxel_size: Voxel size used for RANSAC.

    Returns:
        True if a separate load at refinement resolution is needed.
    """
    if refinement_voxel_size is None:
        return False
    return refinement_voxel_size != ransac_voxel_size


def _load_scan_for_refinement(
    ply_path: Path,
    refinement_voxel_size: float | None,
    scan_ransac: o3d.geometry.PointCloud,
    ransac_voxel_size: float,
) -> o3d.geometry.PointCloud:
    """Load a scan at the refinement resolution, reusing the RANSAC cloud when possible.

    If refinement_voxel_size matches ransac_voxel_size (or is None), returns scan_ransac
    directly without re-loading from disk.

    Args:
        ply_path: Path to the scan PLY file.
        refinement_voxel_size: Voxel size for refinement (0 = original, None = reuse RANSAC).
        scan_ransac: Pre-loaded RANSAC scan to reuse when resolutions match.
        ransac_voxel_size: Voxel size used for RANSAC downsampling.

    Returns:
        Point cloud at the refinement resolution.
    """
    if not _needs_separate_refinement_clouds(refinement_voxel_size, ransac_voxel_size):
        return scan_ransac
    voxel_for_loading = (
        refinement_voxel_size if refinement_voxel_size is not None else 0.0
    )
    res_label = f"{voxel_for_loading}" if voxel_for_loading > 0 else "original"
    logger.debug(f"Loading scan at refinement resolution ({res_label})...")
    return load_point_cloud(ply_path, voxel_size=voxel_for_loading)


def process_single_scan(
    scan_idx: int,
    ply_path: Path,
    json_path: Path,
    global_map: o3d.geometry.PointCloud,
    voxel_size: float,
    max_correspondence_distance: float,
    ground_truth_poses: list | None = None,
    relative_idx: int = 0,
    end_idx: int = 0,
    refine_poses: bool = False,
    icp_refinement_distance: float = 50.0,
    use_gicp: bool = False,
    refinement_voxel_size: float | None = None,
    global_map_refinement: o3d.geometry.PointCloud | None = None,
    estimated_pose: np.ndarray | None = None,
) -> dict:
    """Process a single scan: load, localize, compute errors, and log results.

    Args:
        scan_idx: Absolute scan index in the dataset.
        ply_path: Path to the scan PLY file.
        json_path: Path to the scan JSON file.
        global_map: Global map point cloud at RANSAC resolution.
        voxel_size: Voxel size for downsampling and feature computation (RANSAC).
        max_correspondence_distance: Maximum correspondence distance for RANSAC.
        ground_truth_poses: Optional list of ground truth poses.
        relative_idx: Index relative to the start of current processing batch.
        end_idx: Last scan index for logging purposes.
        refine_poses: If True, refine the pose using ICP or GICP.
        icp_refinement_distance: Maximum correspondence distance for ICP refinement.
        use_gicp: If True, use GICP for refinement; otherwise use ICP.
        refinement_voxel_size: Voxel size for refinement step (None = reuse RANSAC clouds,
            0 = original resolution, positive = custom downsample).
        global_map_refinement: Pre-loaded map at refinement resolution. If None and
            refinement uses a different resolution, falls back to global_map.
        estimated_pose: Pre-estimated pose to use instead of running RANSAC. When provided,
            global registration is skipped and this pose is used as the initial estimate
            for refinement. Requires refine_poses=True (guaranteed by the caller).

    Returns:
        Dictionary containing rotation error, translation error, fitness, RMSE,
        and result entry for JSON output.
    """
    logger.info(f"\nScan {scan_idx}/{end_idx}: {ply_path.stem}")
    logger.info("-" * 80)

    t_total_start = time.perf_counter()

    # Load scan at RANSAC resolution
    logger.info(f"Loading scan: {ply_path.stem}")
    scan_ransac = load_point_cloud(ply_path, voxel_size=voxel_size)
    logger.info(f"    Scan (RANSAC): {len(scan_ransac.points)} points")

    # Load ground truth pose
    if ground_truth_poses is not None and relative_idx < len(ground_truth_poses):
        H_gt = ground_truth_poses[relative_idx]
    else:
        H_gt = load_transformation_matrix(json_path)

    # Perform global localization using RANSAC, or use a pre-estimated pose
    time_ransac_s: float | None = None
    if estimated_pose is not None:
        logger.info("Using pre-estimated pose (skipping RANSAC global registration)...")
        H_estimated = estimated_pose
        reg_result = None
    else:
        logger.info("Localizing scan against map with RANSAC...")
        t_ransac_start = time.perf_counter()
        H_estimated, reg_result = global_localize_scan_to_map(
            scan_ransac,
            global_map,
            voxel_size=voxel_size,
            max_correspondence_distance=max_correspondence_distance,
        )
        time_ransac_s = time.perf_counter() - t_ransac_start
        logger.debug(f"    RANSAC took {time_ransac_s:.3f} s")

    # Optionally refine the pose using ICP or GICP
    time_refinement_s: float | None = None
    if refine_poses:
        refinement_name = "GICP" if use_gicp else "ICP"
        scan_refine = _load_scan_for_refinement(
            ply_path, refinement_voxel_size, scan_ransac, voxel_size
        )
        map_refine = (
            global_map_refinement if global_map_refinement is not None else global_map
        )
        if _needs_separate_refinement_clouds(refinement_voxel_size, voxel_size):
            logger.info(
                f"Refining pose with {refinement_name} at refinement resolution "
                f"(scan: {len(scan_refine.points)} pts, map: {len(map_refine.points)} pts)..."
            )
        else:
            logger.info(
                f"Refining pose with {refinement_name} (reusing RANSAC clouds)..."
            )
        t_refine_start = time.perf_counter()
        H_estimated, icp_result = pairwise_registration(
            source=scan_refine,
            target=map_refine,
            max_correspondence_distance=icp_refinement_distance,
            init_transformation=H_estimated,
            max_iteration=30,
            verbose=True,
            use_generalized_icp=use_gicp,
        )
        time_refinement_s = time.perf_counter() - t_refine_start
        logger.debug(f"    Refinement took {time_refinement_s:.3f} s")
        # Update registration result to include refinement statistics
        reg_result = icp_result

    # Extract rotation and translation
    R_gt = H_gt[:3, :3]
    t_gt = H_gt[:3, 3]
    R_est = H_estimated[:3, :3]
    t_est = H_estimated[:3, 3]

    # Compute errors
    rot_error = rotation_error_degrees(R_gt, R_est)
    trans_error = translation_error(t_gt, t_est)

    # Log results
    if reg_result is not None:
        logger.info("Localization Statistics:")
        logger.info(f"    Fitness:      {reg_result.fitness:.4f}")
        logger.info(f"    Inlier RMSE:  {reg_result.inlier_rmse:.4f}")

    logger.info("Error vs Ground Truth:")
    logger.info(f"    Rotation error:    {rot_error:.4f}°")
    logger.info(f"    Translation error: {trans_error:.4f}")

    time_total_s = time.perf_counter() - t_total_start
    logger.info("Timing:")
    if time_ransac_s is not None:
        logger.info(f"    RANSAC:      {time_ransac_s:.3f} s")
    if time_refinement_s is not None:
        logger.info(f"    Refinement:  {time_refinement_s:.3f} s")
    logger.info(f"    Total:       {time_total_s:.3f} s")

    # Create result entry for JSON output
    localization_stats = (
        {
            "fitness": float(reg_result.fitness),
            "inlier_rmse": float(reg_result.inlier_rmse),
        }
        if reg_result is not None
        else {}
    )
    timing = {
        "total_s": time_total_s,
        "ransac_s": time_ransac_s,
        "refinement_s": time_refinement_s,
    }
    result_entry = {
        "scan_index": scan_idx,
        "scan_name": ply_path.stem,
        "localization": localization_stats,
        "errors": {
            "rotation_degrees": float(rot_error),
            "translation": float(trans_error),
        },
        "timing": timing,
        "ground_truth_pose": H_gt.tolist(),
        "estimated_pose": H_estimated.tolist(),
    }

    return {
        "rotation_error": rot_error,
        "translation_error": trans_error,
        "fitness": reg_result.fitness if reg_result is not None else float("nan"),
        "rmse": reg_result.inlier_rmse if reg_result is not None else float("nan"),
        "time_total_s": time_total_s,
        "time_ransac_s": time_ransac_s,
        "time_refinement_s": time_refinement_s,
        "result_entry": result_entry,
    }


def _build_method_string(
    refine_poses: bool,
    use_gicp: bool,
    voxel_size: float,
    refinement_voxel_size: float | None,
) -> str:
    """Build a human-readable description of the registration method.

    Args:
        refine_poses: Whether ICP/GICP refinement is enabled.
        use_gicp: Whether GICP (vs ICP) is used for refinement.
        voxel_size: RANSAC voxel size.
        refinement_voxel_size: Refinement voxel size (None = reuse RANSAC, 0 = original).

    Returns:
        A descriptive string summarising the pipeline.
    """
    if not refine_poses:
        return "RANSAC-based global registration"
    refinement_name = "GICP" if use_gicp else "ICP"
    if refinement_voxel_size is None or refinement_voxel_size == voxel_size:
        res_str = f"voxel={voxel_size} (same as RANSAC)"
    elif refinement_voxel_size == 0:
        res_str = "original resolution"
    else:
        res_str = f"voxel={refinement_voxel_size}"
    return f"RANSAC + {refinement_name} refinement ({res_str})"


def localize_scans(
    data_dir: str,
    map_file: str,
    voxel_size: float = 50.0,
    max_correspondence_distance: float = 150.0,
    output_file: str | None = None,
    start_scan: int | None = None,
    end_scan: int | None = None,
    step: int = 1,
    poses_file: str | None = None,
    refine_poses: bool = False,
    icp_refinement_distance: float = 50.0,
    use_gicp: bool = False,
    refinement_voxel_size: float | None = None,
    estimated_poses_file: str | None = None,
):
    """Localize all scans against a global map using RANSAC-based global registration.

    Args:
        data_dir: Directory containing scan pairs.
        map_file: Path to global map PLY file.
        voxel_size: Voxel size for downsampling and feature computation (RANSAC).
        max_correspondence_distance: Maximum correspondence distance for RANSAC.
        output_file: Optional file to save results (JSON format).
        start_scan: Index of first scan (0-based, inclusive).
        end_scan: Index of last scan (0-based, inclusive).
        step: Process every Nth scan within the range (1 = all scans).
        poses_file: Optional single file containing all ground truth poses.
        refine_poses: If True, refine the pose using ICP or GICP.
        icp_refinement_distance: Maximum correspondence distance for ICP refinement.
        use_gicp: If True, use GICP for refinement; otherwise use ICP.
        refinement_voxel_size: Voxel size for the refinement step (None = reuse RANSAC
            clouds, 0 = original resolution, positive = custom downsample).
        estimated_poses_file: Optional path to a JSON file of pre-estimated poses (same
            format as estimated_poses.json). When provided, RANSAC global registration is
            skipped and the loaded poses are used as initial estimates for refinement.
            Has no effect unless refine_poses=True.
    """
    data_path = Path(data_dir)
    map_path = Path(map_file)

    # Validate inputs
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    if not map_path.exists():
        raise FileNotFoundError(f"Map file not found: {map_file}")

    # Find scan pairs
    pairs = find_scan_pairs(data_path)
    if not pairs:
        raise ValueError("No scan pairs found in directory")

    # Validate and apply scan range
    total_scans = len(pairs)
    start_idx = start_scan if start_scan is not None else 0
    end_idx = end_scan if end_scan is not None else total_scans - 1

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
        raise ValueError(f"step ({step}) must be >= 1")

    pairs = pairs[start_idx : end_idx + 1 : step]
    num_scans = len(pairs)

    logger.info(
        f"Processing scans {start_idx} to {end_idx} (step={step}) - "
        f"{num_scans} scans out of {total_scans} total"
    )

    # Load poses from file if provided
    ground_truth_poses = None
    if poses_file:
        ground_truth_poses = load_poses_from_file(poses_file, num_scans)
        if ground_truth_poses is None:
            logger.warning(
                "Failed to load poses file, falling back to individual JSON files"
            )

    # Load pre-estimated poses if provided; skip processing when no refinement is requested
    prior_estimated_poses = None
    if estimated_poses_file:
        if not refine_poses:
            logger.warning(
                "--estimated-poses was provided but --refine-poses is not set. "
                "Nothing to do: exiting without processing."
            )
            return
        prior_estimated_poses = load_poses_from_file(estimated_poses_file, num_scans)
        if prior_estimated_poses is None:
            raise ValueError(
                f"Failed to load pre-estimated poses from: {estimated_poses_file}"
            )
        logger.info(
            f"Loaded {len(prior_estimated_poses)} pre-estimated poses from "
            f"{estimated_poses_file} (RANSAC will be skipped)"
        )

    # Determine method string based on refinement settings
    method_str = _build_method_string(
        refine_poses, use_gicp, voxel_size, refinement_voxel_size
    )
    if prior_estimated_poses is not None:
        method_str = f"{method_str} [from pre-estimated poses]"

    # Save execution parameters if output file is specified
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        params = {
            "timestamp": datetime.now(UTC).astimezone().isoformat(),
            "data_dir": str(data_path),
            "map_file": str(map_path),
            "voxel_size": voxel_size,
            "max_correspondence_distance": max_correspondence_distance,
            "refine_poses": refine_poses,
            "icp_refinement_distance": icp_refinement_distance
            if refine_poses
            else None,
            "use_gicp": use_gicp if refine_poses else None,
            "refinement_voxel_size": refinement_voxel_size if refine_poses else None,
            "estimated_poses_file": estimated_poses_file,
            "start_scan_requested": start_scan,
            "end_scan_requested": end_scan,
            "step": step,
            "start_scan_actual": start_idx,
            "end_scan_actual": end_idx,
            "total_scans_available": total_scans,
            "num_scans_processed": num_scans,
            "poses_file": poses_file,
            "method": method_str,
        }
        params_output = output_path.parent / "parameters.json"
        save_parameters(params, params_output)

    # Load global map at RANSAC resolution
    logger.info(f"Loading global map from: {map_file}")
    global_map = load_point_cloud(map_path, voxel_size=voxel_size)
    logger.info(f"Global map (RANSAC): {len(global_map.points)} points")

    # Load map at refinement resolution if a separate resolution is requested
    global_map_refinement = None
    if refine_poses and _needs_separate_refinement_clouds(
        refinement_voxel_size, voxel_size
    ):
        ref_voxel = refinement_voxel_size if refinement_voxel_size is not None else 0.0
        global_map_refinement = load_point_cloud(map_path, voxel_size=ref_voxel)
        logger.info(
            f"Global map (refinement): {len(global_map_refinement.points)} points"
        )

    logger.info(f"Voxel size: {voxel_size}")
    logger.info(f"Max correspondence distance: {max_correspondence_distance}")
    logger.info(f"Method: {method_str}")
    if refine_poses:
        logger.info(f"ICP refinement distance: {icp_refinement_distance}")
    logger.info("=" * 80)

    # Statistics accumulators
    rotation_errors = []
    translation_errors = []
    fitness_scores = []
    rmse_values = []
    times_total_s = []
    times_ransac_s = []
    times_refinement_s = []
    results = []

    # Process each scan
    for i, (ply_path, json_path) in enumerate(pairs):
        scan_idx = start_idx + i

        result = process_single_scan(
            scan_idx=scan_idx,
            ply_path=ply_path,
            json_path=json_path,
            global_map=global_map,
            voxel_size=voxel_size,
            max_correspondence_distance=max_correspondence_distance,
            ground_truth_poses=ground_truth_poses,
            relative_idx=i,
            end_idx=end_idx,
            refine_poses=refine_poses,
            icp_refinement_distance=icp_refinement_distance,
            use_gicp=use_gicp,
            refinement_voxel_size=refinement_voxel_size,
            global_map_refinement=global_map_refinement,
            estimated_pose=prior_estimated_poses[i]
            if prior_estimated_poses is not None
            else None,
        )

        # Store statistics
        rotation_errors.append(result["rotation_error"])
        translation_errors.append(result["translation_error"])
        fitness_scores.append(result["fitness"])
        rmse_values.append(result["rmse"])
        times_total_s.append(result["time_total_s"])
        if result["time_ransac_s"] is not None:
            times_ransac_s.append(result["time_ransac_s"])
        if result["time_refinement_s"] is not None:
            times_refinement_s.append(result["time_refinement_s"])
        results.append(result["result_entry"])

    # Compute summary statistics
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY STATISTICS")
    logger.info("=" * 80)

    def _log_stats(label: str, stats: dict, unit: str = "") -> None:
        suffix = f" {unit}" if unit else ""
        logger.info(f"{label}:")
        logger.info(f"  Mean:   {stats['mean']:.4f}{suffix}")
        logger.info(f"  Median: {stats['median']:.4f}{suffix}")
        logger.info(f"  Std:    {stats['std']:.4f}{suffix}")
        logger.info(f"  Min:    {stats['min']:.4f}{suffix}")
        logger.info(f"  Max:    {stats['max']:.4f}{suffix}")

    _log_stats("Rotation Error", compute_stats(rotation_errors), "°")
    _log_stats("Translation Error", compute_stats(translation_errors))
    _log_stats("Localization Fitness", compute_stats(fitness_scores))
    _log_stats("Localization RMSE", compute_stats(rmse_values))
    _log_stats("Total time per scan", compute_stats(times_total_s), "s")
    if times_ransac_s:
        _log_stats("RANSAC time per scan", compute_stats(times_ransac_s), "s")
    if times_refinement_s:
        _log_stats("Refinement time per scan", compute_stats(times_refinement_s), "s")

    # Save results to JSON
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        summary = {
            "parameters": {
                "data_dir": str(data_path),
                "map_file": str(map_path),
                "voxel_size": voxel_size,
                "max_correspondence_distance": max_correspondence_distance,
                "start_scan": start_idx,
                "end_scan": end_idx,
                "num_scans": num_scans,
                "poses_file": poses_file,
                "refine_poses": refine_poses,
                "icp_refinement_distance": icp_refinement_distance
                if refine_poses
                else None,
                "use_gicp": use_gicp if refine_poses else None,
                "refinement_voxel_size": refinement_voxel_size
                if refine_poses
                else None,
                "estimated_poses_file": estimated_poses_file,
                "method": method_str,
            },
            "statistics": {
                "rotation_error_degrees": compute_stats(rotation_errors),
                "translation_error": compute_stats(translation_errors),
                "fitness": compute_stats(fitness_scores),
                "inlier_rmse": compute_stats(rmse_values),
                "timing": {
                    "total_s": compute_stats(times_total_s),
                    "ransac_s": compute_stats(times_ransac_s)
                    if times_ransac_s
                    else None,
                    "refinement_s": compute_stats(times_refinement_s)
                    if times_refinement_s
                    else None,
                },
            },
            "num_scans": num_scans,
            "rotation_errors": rotation_errors,
            "translation_errors": translation_errors,
            "results": results,
        }

        with open(output_path, "w") as f:
            json.dump(summary, f, indent=2)

        logger.info(f"\nResults saved to: {output_file}")

        # Save estimated poses to separate file
        estimated_poses = [np.array(result["estimated_pose"]) for result in results]
        poses_output_path = output_path.parent / "estimated_poses.json"
        save_poses_to_file(estimated_poses, poses_output_path)

    logger.info("\n" + "=" * 80)
    logger.info("✓ Localization complete!")
    logger.info("=" * 80)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Localize scans against a global map and compare with ground truth",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Input directory containing .ply and .json scan pairs",
    )

    parser.add_argument(
        "--map",
        "-m",
        required=True,
        help="Path to global map PLY file (e.g., fused_map_optimized.ply)",
    )

    parser.add_argument(
        "--voxel-size",
        type=float,
        default=50.0,
        help="Voxel size for downsampling scans and map (0 = no downsampling)",
    )

    parser.add_argument(
        "--max-correspondence-distance",
        type=float,
        default=150.0,
        help="Maximum correspondence distance for RANSAC localization",
    )

    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output JSON file to save detailed results",
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
        help="Process every Nth scan within the selected range (1 = all scans)",
    )

    parser.add_argument(
        "--poses",
        "-p",
        default=None,
        help="Optional path to JSON file containing all ground truth poses (like optimized_poses.json). If not provided, uses individual .json files.",
    )

    parser.add_argument(
        "--refine-poses",
        action="store_true",
        help="Refine RANSAC pose estimates using ICP or GICP",
    )

    parser.add_argument(
        "--icp-refinement-distance",
        type=float,
        default=50.0,
        help="Maximum correspondence distance for ICP/GICP refinement",
    )

    parser.add_argument(
        "--use-gicp",
        action="store_true",
        help="Use Generalized ICP for refinement (otherwise use point-to-plane ICP)",
    )

    parser.add_argument(
        "--refinement-voxel-size",
        type=float,
        default=None,
        help=(
            "Voxel size for downsampling during the ICP/GICP refinement step. "
            "If not set, the RANSAC voxel size is reused (no extra loading). "
            "Set to 0 to use the original undownsampled point clouds."
        ),
    )

    parser.add_argument(
        "--estimated-poses",
        default=None,
        help=(
            "Path to a JSON file of pre-estimated poses (same format as estimated_poses.json "
            "produced by this script). When provided, RANSAC global registration is skipped "
            "and these poses are used as the initial estimate for refinement. "
            "Has no effect unless --refine-poses is also passed."
        ),
    )

    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set logging level",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(level=getattr(logging, args.log_level))

    try:
        localize_scans(
            data_dir=args.input,
            map_file=args.map,
            voxel_size=args.voxel_size,
            max_correspondence_distance=args.max_correspondence_distance,
            output_file=args.output,
            start_scan=args.start_scan,
            end_scan=args.end_scan,
            step=args.step,
            poses_file=args.poses,
            refine_poses=args.refine_poses,
            icp_refinement_distance=args.icp_refinement_distance,
            use_gicp=args.use_gicp,
            refinement_voxel_size=args.refinement_voxel_size,
            estimated_poses_file=args.estimated_poses,
        )
    except Exception as e:
        logger.error(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()
