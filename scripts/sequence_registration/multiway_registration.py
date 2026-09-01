#! /usr/bin/env python3
"""Multiway registration of multiple point cloud scans using pose graph optimization.

This script reads a folder containing paired .ply and .json files with ground truth
transformations, uses Open3D's multiway registration to optimize the poses, and
creates a fused map from the optimized point clouds.
"""

import argparse
import logging

# Add scripts directory to path for imports
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import open3d as o3d

from registration.utils.logging import setup_logging

scripts_dir = Path(__file__).parent
if str(scripts_dir) not in sys.path:
    sys.path.insert(0, str(scripts_dir))

from registration_common import (
    filter_distant_points,
    find_scan_pairs,
    load_and_transform_scan,
    load_point_cloud,
    load_transformation_matrix,
    pairwise_registration,
    remove_outliers,
    save_parameters,
    save_point_cloud_binary,
    save_poses_to_file,
)

logger = logging.getLogger(__name__)

# Registration quality thresholds
LOOP_CLOSURE_FITNESS_THRESHOLD = 0.3  # Minimum fitness to accept loop closure
WARNING_FITNESS_THRESHOLD = 0.9  # Fitness below this triggers warnings
WARNING_RMSE_THRESHOLD_MM = 100.0  # RMSE above this (in mm) triggers warnings

# Global optimization parameters
GLOBAL_OPTIMIZATION_MAX_CORRESPONDENCE = (
    1.0  # Max correspondence distance for global optimization
)
GLOBAL_OPTIMIZATION_EDGE_PRUNE_THRESHOLD = 0.25  # Threshold for pruning edges
GLOBAL_OPTIMIZATION_REFERENCE_NODE = 0  # Reference node for optimization (first scan)


def load_point_clouds_with_poses(
    pairs: list[tuple[Path, Path]],
    voxel_size: float = 0.0,
    load_ground_truth: bool = True,
) -> tuple[list[o3d.geometry.PointCloud], list[np.ndarray]]:
    """Load all point clouds and optionally their initial poses from JSON files.

    Args:
        pairs: List of (ply_path, json_path) tuples.
        voxel_size: If > 0, downsample point clouds with this voxel size.
        load_ground_truth: If True, load ground truth poses from JSON files.

    Returns:
        Tuple of (list of point clouds, list of 4x4 transformation matrices).
        If load_ground_truth is False, poses list contains only identity matrices.
    """
    point_clouds = []
    poses = []

    for i, (ply_path, json_path) in enumerate(pairs):
        logger.info(f"Loading scan {i}/{len(pairs)}: {ply_path.name}")

        # Load point cloud
        pcd = load_point_cloud(ply_path, voxel_size=voxel_size, estimate_normals=True)
        point_clouds.append(pcd)

        # Load ground truth pose or use identity
        if load_ground_truth:
            pose = load_transformation_matrix(json_path)
        else:
            pose = np.eye(4)
        poses.append(pose)

        logger.debug(f"  Loaded {len(pcd.points)} points")

    if load_ground_truth:
        logger.info(f"Loaded {len(point_clouds)} point clouds with ground truth poses")
    else:
        logger.info(f"Loaded {len(point_clouds)} point clouds (ground truth disabled)")
    return point_clouds, poses


def build_initial_poses_from_registration(
    point_clouds: list[o3d.geometry.PointCloud],
    max_correspondence_distance: float,
    use_generalized_icp: bool = False,
) -> list[np.ndarray]:
    """Build initial poses by registering consecutive scans.

    First scan is placed at identity. Each subsequent scan is registered
    to the previous scan, and poses are accumulated in world frame.

    Args:
        point_clouds: List of point clouds to register.
        max_correspondence_distance: Maximum correspondence distance for ICP.
        use_generalized_icp: If True, use GICP; otherwise use classic ICP.

    Returns:
        List of 4x4 transformation matrices (absolute poses in world frame).
    """
    logger.info("Building initial poses from consecutive scan registration...")

    poses = []

    # First scan at identity (world origin)
    poses.append(np.eye(4))
    logger.info("  Scan 0: placed at world origin")

    # Register each scan to previous and accumulate poses
    for i in range(1, len(point_clouds)):
        logger.info(f"  Registering scan {i} to scan {i - 1}...")

        # Register current scan to previous scan
        H_rel, result = pairwise_registration(
            point_clouds[i],
            point_clouds[i - 1],
            max_correspondence_distance,
            init_transformation=np.eye(4),
            verbose=False,
            use_generalized_icp=use_generalized_icp,
        )

        # Accumulate: H_i = H_{i-1} @ H_rel
        # This transforms scan i to world frame
        H_world = poses[i - 1] @ H_rel
        poses.append(H_world)

        logger.info(
            f"    Fitness: {result.fitness:.4f}, RMSE: {result.inlier_rmse:.4f}"
        )

        # Warn if registration quality is poor
        if result.fitness < WARNING_FITNESS_THRESHOLD:
            logger.warning(
                f"    WARNING: Poor fitness ({result.fitness:.4f}) for consecutive "
                f"registration {i - 1} -> {i}. This may indicate registration failure."
            )
        if result.inlier_rmse > WARNING_RMSE_THRESHOLD_MM:
            logger.warning(
                f"    WARNING: High RMSE ({result.inlier_rmse:.4f}) for consecutive "
                f"registration {i - 1} -> {i}. This may indicate poor alignment."
            )

    logger.info(f"Initial poses built for {len(poses)} scans")
    return poses


def build_pose_graph(
    point_clouds: list[o3d.geometry.PointCloud],
    initial_poses: list[np.ndarray],
    max_correspondence_distance: float,
    loop_closure_distance_threshold: float | None = None,
    use_generalized_icp: bool = False,
) -> o3d.pipelines.registration.PoseGraph:
    """Build a pose graph for multiway registration.

    Args:
        point_clouds: List of point clouds to register.
        initial_poses: List of initial pose estimates (4x4 matrices).
        max_correspondence_distance: Maximum correspondence distance for ICP.
        loop_closure_distance_threshold: Distance threshold for loop closure detection.
            If None, no loop closures are detected.
        use_generalized_icp: If True, use GICP; otherwise use classic ICP.

    Returns:
        Pose graph ready for optimization.
    """
    logger.info("Building pose graph...")
    pose_graph = o3d.pipelines.registration.PoseGraph()

    # Add nodes (one per scan)
    for i, pose in enumerate(initial_poses):
        node = o3d.pipelines.registration.PoseGraphNode(pose)
        pose_graph.nodes.append(node)
        logger.debug(f"  Added node {i}")

    # Add odometry edges (consecutive scans)
    logger.info("Adding odometry edges (consecutive scans)...")
    for i in range(len(point_clouds) - 1):
        source_id = i
        target_id = i + 1

        logger.info(f"  Registering scan {source_id} to scan {target_id}")

        # Transform source to target's frame using initial poses
        source_in_target = (
            np.linalg.inv(initial_poses[target_id]) @ initial_poses[source_id]
        )

        # Refine with ICP
        transformation, result = pairwise_registration(
            point_clouds[source_id],
            point_clouds[target_id],
            max_correspondence_distance,
            init_transformation=source_in_target,
            use_generalized_icp=use_generalized_icp,
        )

        # Add edge to pose graph
        information = (
            o3d.pipelines.registration.get_information_matrix_from_point_clouds(
                point_clouds[source_id],
                point_clouds[target_id],
                max_correspondence_distance,
                transformation,
            )
        )

        edge = o3d.pipelines.registration.PoseGraphEdge(
            source_id,
            target_id,
            transformation,
            information,
            uncertain=False,
        )
        pose_graph.edges.append(edge)
        logger.debug(f"    Added odometry edge {source_id} -> {target_id}")

        # Warn if odometry edge has poor registration quality
        if result.fitness < WARNING_FITNESS_THRESHOLD:
            logger.warning(
                f"    WARNING: Poor fitness ({result.fitness:.4f}) for odometry edge "
                f"{source_id} -> {target_id}. This may affect optimization quality."
            )
        if result.inlier_rmse > WARNING_RMSE_THRESHOLD_MM:
            logger.warning(
                f"    WARNING: High RMSE ({result.inlier_rmse:.4f}) for odometry edge "
                f"{source_id} -> {target_id}. This may indicate poor alignment."
            )

    # Add loop closure edges if requested
    if loop_closure_distance_threshold is not None:
        logger.info("Detecting loop closures...")
        num_loop_closures = 0

        for i, source_pcd in enumerate(point_clouds):
            for j in range(i + 2, len(point_clouds)):  # Skip adjacent scans
                # Check if scans are close in world coordinates
                pos_i = initial_poses[i][:3, 3]
                pos_j = initial_poses[j][:3, 3]
                distance = np.linalg.norm(pos_i - pos_j)

                if distance < loop_closure_distance_threshold:
                    logger.info(
                        f"  Potential loop closure: {i} <-> {j} (dist: {distance:.2f})"
                    )

                    # Transform i to j's frame
                    source_in_target = (
                        np.linalg.inv(initial_poses[j]) @ initial_poses[i]
                    )

                    # Try registration
                    transformation, result = pairwise_registration(
                        source_pcd,
                        point_clouds[j],
                        max_correspondence_distance,
                        init_transformation=source_in_target,
                        use_generalized_icp=use_generalized_icp,
                    )

                    # Only add if registration is good
                    if result.fitness > LOOP_CLOSURE_FITNESS_THRESHOLD:
                        information = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
                            source_pcd,
                            point_clouds[j],
                            max_correspondence_distance,
                            transformation,
                        )

                        edge = o3d.pipelines.registration.PoseGraphEdge(
                            i,
                            j,
                            transformation,
                            information,
                            uncertain=True,  # Loop closures are less certain
                        )
                        pose_graph.edges.append(edge)
                        num_loop_closures += 1
                        logger.info(f"    Added loop closure edge {i} <-> {j}")
                    else:
                        logger.warning(
                            f"    WARNING: Rejected loop closure candidate {i} <-> {j} "
                            f"due to poor fitness ({result.fitness:.4f} <= {LOOP_CLOSURE_FITNESS_THRESHOLD})"
                        )

        logger.info(f"Added {num_loop_closures} loop closure edges")

    logger.info(
        f"Pose graph built: {len(pose_graph.nodes)} nodes, {len(pose_graph.edges)} edges"
    )
    return pose_graph


def optimize_pose_graph(
    pose_graph: o3d.pipelines.registration.PoseGraph,
) -> o3d.pipelines.registration.PoseGraph:
    """Optimize a pose graph using global optimization.

    Args:
        pose_graph: Pose graph to optimize.

    Returns:
        Optimized pose graph.
    """
    logger.info("Optimizing pose graph...")

    option = o3d.pipelines.registration.GlobalOptimizationOption(
        max_correspondence_distance=GLOBAL_OPTIMIZATION_MAX_CORRESPONDENCE,
        edge_prune_threshold=GLOBAL_OPTIMIZATION_EDGE_PRUNE_THRESHOLD,
        reference_node=GLOBAL_OPTIMIZATION_REFERENCE_NODE,  # Use first scan as reference
    )

    o3d.pipelines.registration.global_optimization(
        pose_graph,
        o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
        o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
        option,
    )

    logger.info("Pose graph optimization complete")
    return pose_graph


def load_full_resolution_scans(
    pairs: list[tuple[Path, Path]],
    poses: list[np.ndarray],
) -> list[o3d.geometry.PointCloud]:
    """Reload all scans at full resolution and transform with the given poses.

    This is called after pose graph optimisation to build the fused map at full
    point density, independently of the voxel size used during registration.

    Args:
        pairs: List of (ply_path, json_path) tuples identifying the scans.
        poses: List of 4x4 transformation matrices, one per scan.

    Returns:
        List of transformed full-resolution point clouds.
    """
    logger.info(f"Reloading {len(pairs)} scans at full resolution for map fusion...")
    transformed_scans = []
    for i, ((ply_path, _), pose) in enumerate(zip(pairs, poses)):
        logger.info(f"  Loading scan {i}/{len(pairs)}: {ply_path.name}")
        pcd = load_and_transform_scan(ply_path, pose)
        transformed_scans.append(pcd)
        logger.debug(f"  Scan {i}: {len(pcd.points)} points")
    logger.info(f"Loaded {len(transformed_scans)} full-resolution scans")
    return transformed_scans


def create_fused_map(
    transformed_scans: list[o3d.geometry.PointCloud],
    voxel_size: float = 10.0,
    remove_outliers_flag: bool = False,
    outlier_nb_neighbors: int = 20,
    outlier_std_ratio: float = 2.0,
    filter_distant_flag: bool = False,
    max_distance: float | None = None,
    distance_percentile: float = 99.0,
) -> o3d.geometry.PointCloud:
    """Create a fused map from a list of pre-transformed point clouds.

    Args:
        transformed_scans: List of point clouds already in world frame.
        voxel_size: Voxel size for the final downsampling step. Set to 0
            to skip downsampling.
        remove_outliers_flag: If True, remove outliers from fused map.
        outlier_nb_neighbors: Number of neighbors for statistical outlier removal.
        outlier_std_ratio: Standard deviation ratio threshold for outlier removal.
        filter_distant_flag: If True, filter out points too far from centroid.
        max_distance: Maximum distance from centroid. If None, uses percentile.
        distance_percentile: Distance percentile threshold (default: 99.0).

    Returns:
        Fused and downsampled point cloud.
    """
    logger.info("Creating fused map...")

    # Concatenate all pre-transformed point clouds
    combined_pcd = o3d.geometry.PointCloud()
    for i, pcd in enumerate(transformed_scans):
        combined_pcd += pcd
        logger.debug(f"  Added scan {i} with {len(pcd.points)} points")

    logger.info(f"Combined map has {len(combined_pcd.points)} points")

    # Downsample to merge overlapping points from adjacent scans
    if voxel_size > 0:
        logger.info(f"Downsampling with voxel size {voxel_size}...")
        fused_pcd = combined_pcd.voxel_down_sample(voxel_size=voxel_size)
        logger.info(
            f"Fused map has {len(fused_pcd.points)} points "
            f"(reduction: {len(combined_pcd.points) - len(fused_pcd.points)} points)"
        )
    else:
        fused_pcd = combined_pcd

    # Filter distant points if requested
    if filter_distant_flag:
        fused_pcd = filter_distant_points(
            fused_pcd, max_distance=max_distance, percentile=distance_percentile
        )

    # Remove outliers if requested
    if remove_outliers_flag:
        fused_pcd = remove_outliers(
            fused_pcd, nb_neighbors=outlier_nb_neighbors, std_ratio=outlier_std_ratio
        )

    return fused_pcd


def multiway_registration(
    data_dir: str,
    output_dir: str,
    voxel_size_input: float = 50.0,
    voxel_size_fusion: float = 10.0,
    max_correspondence_distance: float = 150.0,
    loop_closure_distance: float | None = None,
    use_ground_truth: bool = True,
    start_scan: int | None = None,
    end_scan: int | None = None,
    step: int = 1,
    remove_outliers_flag: bool = False,
    outlier_nb_neighbors: int = 20,
    outlier_std_ratio: float = 2.0,
    filter_distant_flag: bool = False,
    max_distance: float | None = None,
    distance_percentile: float = 99.0,
    use_generalized_icp: bool = False,
):
    """Main function for multiway registration and map fusion.

    Args:
        data_dir: Directory containing .ply and .json scan pairs.
        output_dir: Directory where outputs will be saved.
        voxel_size_input: Voxel size for downsampling input scans (0 = no downsampling).
        voxel_size_fusion: Voxel size for final fused map downsampling.
        max_correspondence_distance: Maximum correspondence distance for ICP.
        loop_closure_distance: Distance threshold for loop closure detection (None = disabled).
        use_ground_truth: If True, use ground truth poses from JSON files as initial estimates.
            If False, perform preliminary consecutive registration to build initial poses.
        start_scan: Index of first scan to process (0-based, inclusive). None = start from 0.
        end_scan: Index of last scan to process (0-based, inclusive). None = process until end.
        step: Process every nth scan within the selected range (default: 1 = all scans).
        remove_outliers_flag: If True, remove outliers from fused map.
        outlier_nb_neighbors: Number of neighbors for statistical outlier removal.
        outlier_std_ratio: Standard deviation ratio threshold for outlier removal.
        filter_distant_flag: If True, filter out points too far from centroid.
        max_distance: Maximum distance from centroid. If None, uses percentile.
        distance_percentile: Distance percentile threshold (default: 99.0).
        use_generalized_icp: If True, use GICP; otherwise use classic ICP.
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

    # Validate range
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
    logger.info(
        f"Processing scans {start_idx} to {end_idx} (step={step}) - "
        f"{len(pairs)} scans out of {total_scans} total"
    )

    # Save execution parameters
    params = {
        "timestamp": datetime.now().isoformat(),
        "data_dir": str(data_path),
        "output_dir": str(output_path),
        "voxel_size_input": voxel_size_input,
        "voxel_size_fusion": voxel_size_fusion,
        "max_correspondence_distance": max_correspondence_distance,
        "loop_closure_distance": loop_closure_distance,
        "use_ground_truth": use_ground_truth,
        "start_scan_requested": start_scan,
        "end_scan_requested": end_scan,
        "step": step,
        "start_scan_actual": start_idx,
        "end_scan_actual": end_idx,
        "total_scans_available": total_scans,
        "num_scans_processed": len(pairs),
        "remove_outliers": remove_outliers_flag,
        "outlier_nb_neighbors": outlier_nb_neighbors,
        "outlier_std_ratio": outlier_std_ratio,
        "filter_distant": filter_distant_flag,
        "max_distance": max_distance,
        "distance_percentile": distance_percentile,
        "use_generalized_icp": use_generalized_icp,
    }
    params_output = output_path / "parameters.json"
    save_parameters(params, params_output)

    # Load point clouds with initial poses
    logger.info("=" * 60)
    logger.info("Loading point clouds...")
    logger.info("=" * 60)
    point_clouds, initial_poses = load_point_clouds_with_poses(
        pairs, voxel_size=voxel_size_input, load_ground_truth=use_ground_truth
    )

    # If not using ground truth, build initial poses from consecutive registration
    if not use_ground_truth:
        logger.info("=" * 60)
        logger.info("Building initial poses...")
        logger.info("=" * 60)
        initial_poses = build_initial_poses_from_registration(
            point_clouds, max_correspondence_distance, use_generalized_icp
        )

    # Build pose graph
    logger.info("=" * 60)
    logger.info("Building pose graph...")
    logger.info("=" * 60)
    pose_graph = build_pose_graph(
        point_clouds,
        initial_poses,
        max_correspondence_distance,
        loop_closure_distance_threshold=loop_closure_distance,
        use_generalized_icp=use_generalized_icp,
    )

    # Optimize pose graph
    logger.info("=" * 60)
    logger.info("Optimizing pose graph...")
    logger.info("=" * 60)
    optimized_pose_graph = optimize_pose_graph(pose_graph)

    # Extract optimized poses
    optimized_poses = [node.pose for node in optimized_pose_graph.nodes]

    # Save optimized poses
    poses_output = output_path / "optimized_poses.json"
    save_poses_to_file(optimized_poses, poses_output)

    # Reload full-resolution scans and apply optimised poses for map fusion.
    # This is separate from the downsampled clouds used during registration so
    # that --voxel-size-fusion controls the final map resolution independently.
    logger.info("=" * 60)
    logger.info("Reloading full-resolution scans for map fusion...")
    logger.info("=" * 60)
    full_res_scans = load_full_resolution_scans(pairs, optimized_poses)

    # Create and save fused map
    logger.info("=" * 60)
    logger.info("Creating fused map...")
    logger.info("=" * 60)
    fused_map = create_fused_map(
        full_res_scans,
        voxel_size=voxel_size_fusion,
        remove_outliers_flag=remove_outliers_flag,
        outlier_nb_neighbors=outlier_nb_neighbors,
        outlier_std_ratio=outlier_std_ratio,
        filter_distant_flag=filter_distant_flag,
        max_distance=max_distance,
        distance_percentile=distance_percentile,
    )
    map_output = output_path / "fused_map_optimized.ply"
    save_point_cloud_binary(fused_map, map_output)

    logger.info("=" * 60)
    logger.info("✓ Multiway registration complete!")
    logger.info("=" * 60)


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Multiway registration of point cloud scans using pose graph optimization",
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
        default="output/multiway_registration",
        help="Output directory for fused map and optimized poses",
    )

    parser.add_argument(
        "--voxel-size-input",
        type=float,
        default=50.0,
        help="Voxel size for downsampling input scans (0 = no downsampling)",
    )

    parser.add_argument(
        "--voxel-size-fusion",
        type=float,
        default=10.0,
        help="Voxel size for final fused map downsampling",
    )

    parser.add_argument(
        "--max-correspondence-distance",
        type=float,
        default=150.0,
        help="Maximum correspondence distance for ICP registration",
    )

    parser.add_argument(
        "--loop-closure-distance",
        type=float,
        default=None,
        help="Distance threshold for loop closure detection (None = disabled)",
    )

    parser.add_argument(
        "--no-ground-truth",
        action="store_true",
        help="Do not use ground truth poses; build initial poses from consecutive scan registration",
    )

    parser.add_argument(
        "--start-scan",
        type=int,
        default=None,
        help="Index of first scan to process (0-based, inclusive). Default: 0",
    )

    parser.add_argument(
        "--end-scan",
        type=int,
        default=None,
        help="Index of last scan to process (0-based, inclusive). Default: last scan",
    )

    parser.add_argument(
        "--step",
        type=int,
        default=1,
        help="Process every nth scan within the selected range (default: 1 = all scans)",
    )

    parser.add_argument(
        "--remove-outliers",
        action="store_true",
        help="Remove outlier points from the fused map",
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
        help="Maximum distance from centroid for filtering. If not specified, uses percentile",
    )

    parser.add_argument(
        "--distance-percentile",
        type=float,
        default=99.0,
        help="Distance percentile threshold for filtering (only used if --max-distance not specified)",
    )

    parser.add_argument(
        "--use-gicp",
        action="store_true",
        help="Use Generalized ICP (GICP) instead of classic ICP for registration",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging()

    try:
        multiway_registration(
            data_dir=args.input,
            output_dir=args.output,
            voxel_size_input=args.voxel_size_input,
            voxel_size_fusion=args.voxel_size_fusion,
            max_correspondence_distance=args.max_correspondence_distance,
            loop_closure_distance=args.loop_closure_distance,
            use_ground_truth=not args.no_ground_truth,
            start_scan=args.start_scan,
            end_scan=args.end_scan,
            step=args.step,
            remove_outliers_flag=args.remove_outliers,
            outlier_nb_neighbors=args.outlier_nb_neighbors,
            outlier_std_ratio=args.outlier_std_ratio,
            filter_distant_flag=args.filter_distant,
            max_distance=args.max_distance,
            distance_percentile=args.distance_percentile,
            use_generalized_icp=args.use_gicp,
        )
    except Exception as e:
        logger.error(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()
