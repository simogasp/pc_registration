"""Metrics for point cloud registration evaluation."""

import copy
import logging

import numpy as np
import open3d as o3d


def compute_rmse_between_point_clouds(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
) -> tuple[float, np.ndarray]:
    """Compute RMSE between corresponding points in two point clouds.

    Calculates the Root Mean Square Error (RMSE) between a source and target
    point cloud. The point clouds must have the same number of points, as the
    distance is computed for corresponding points at the same indices.

    Args:
        source: Source point cloud.
        target: Target point cloud (must have same number of points as source).

    Returns:
        A tuple containing:
            - rmse: Root Mean Square Error (scalar).
            - distances: Per-point Euclidean distances as a (N,) array.

    Raises:
        ValueError: If the point clouds have different numbers of points.

    Note:
        This function assumes point-to-point correspondence (i.e., source.points[i]
        corresponds to target.points[i]). For registration evaluation, typically
        the source would be transformed before calling this function.
    """
    source_points = np.asarray(source.points)
    target_points = np.asarray(target.points)

    if len(source_points) != len(target_points):
        raise ValueError(
            f"Point clouds must have the same number of points. "
            f"Source: {len(source_points)}, Target: {len(target_points)}"
        )

    dists = np.linalg.norm(source_points - target_points, axis=1)
    rmse_val = np.sqrt(np.mean(dists**2))
    logging.debug(f"Computed RMSE = {rmse_val:.6f}")
    return rmse_val, dists


def compute_rmse_transformations(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    T_est: np.ndarray,
    T_gt: np.ndarray,
) -> float:
    """Compute RMSE after applying estimated and ground-truth transformations.

    Transforms the source point cloud using both the estimated and ground truth
    transformations, then computes the RMSE between the two transformed results.
    This provides a measure of registration accuracy that accounts for both
    rotation and translation errors.

    Args:
        source: Source point cloud to transform.
        target: Target point cloud (used for reference, not directly compared).
        T_est: Estimated 4x4 transformation matrix.
        T_gt: Ground truth 4x4 transformation matrix.

    Returns:
        The RMSE between the point cloud transformed with T_est and the one
        transformed with T_gt.

    Note:
        The target point cloud is not used in the computation but is kept as
        a parameter for consistency with typical registration evaluation workflows.
        The function creates copies of the source to avoid modifying the original.
    """
    # Apply estimated transformation to a copy of the source
    source_copy_est = copy.deepcopy(source)
    source_copy_est.transform(T_est)

    # Apply ground truth transformation to another copy
    source_copy_gt = copy.deepcopy(source)
    source_copy_gt.transform(T_gt)

    # Compute RMSE between the two transformed versions
    rmse_val, _ = compute_rmse_between_point_clouds(source_copy_est, source_copy_gt)
    return rmse_val
