#! /usr/bin/env python3

import argparse
import logging

import numpy as np
import open3d as o3d

from registration.utils.logging import setup_logging
from registration.utils.metrics import compute_rmse_transformations
from registration.utils.transforms import transformation_error
from registration.visualization.viewer import (
    draw_registration_result,
    print_point_cloud_info,
)

logger = logging.getLogger(__name__)


def preprocess_point_cloud(pcd, voxel_size: float) -> tuple:
    """Preprocess a point cloud by downsampling and computing features.

    This function performs three main steps:
    1. Downsamples the point cloud using voxel-based downsampling
    2. Estimates normals for each point using a hybrid KD-tree search
    3. Computes Fast Point Feature Histogram (FPFH) features for registration

    Args:
        pcd: Input point cloud to preprocess.
        voxel_size: The size of the voxel for downsampling. Smaller values result in
            denser point clouds but slower processing.

    Returns:
        A tuple containing:
            - pcd_down: The downsampled point cloud with estimated normals
            - pcd_fpfh: The computed FPFH features for the downsampled point cloud
    """
    logger.debug(f"Downsample with a voxel size {voxel_size:.3f}")
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    logger.debug(f"Estimate normal with search radius {radius_normal:.3f}")
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
    )

    radius_feature = voxel_size * 5
    logger.debug(f"Compute FPFH feature with search radius {radius_feature:.3f}")
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100),
    )
    return pcd_down, pcd_fpfh


def prepare_dataset(
    source_file: str, target_file: str, voxel_size: float, trans_init=np.identity(4)
) -> tuple:
    """Load and prepare point cloud datasets for registration.

    Loads source and target point clouds from files, applies an initial transformation
    to the source cloud, and preprocesses both clouds by downsampling and computing
    FPFH features for feature-based registration.

    Args:
        source_file: File path to the source point cloud.
        target_file: File path to the target point cloud.
        voxel_size: The size of the voxel for downsampling both point clouds.

    Returns:
        A tuple containing:
            - source: The original source point cloud with initial transformation applied
            - target: The original target point cloud
            - source_down: Downsampled source point cloud
            - target_down: Downsampled target point cloud
            - source_fpfh: FPFH features of the downsampled source
            - target_fpfh: FPFH features of the downsampled target
    """
    logger.info("Load two point clouds and disturb initial pose")
    source = o3d.io.read_point_cloud(source_file)
    print_point_cloud_info(source, f"Source: {source_file}")
    target = o3d.io.read_point_cloud(target_file)
    print_point_cloud_info(target, f"Target: {target_file}")
    # trans_init = np.asarray([[0.0, 0.0, 1.0, 0.0],
    #                          [1.0, 0.0, 0.0, 0.0],
    #                          [0.0, 1.0, 0.0, 0.0],
    #                          [0.0, 0.0, 0.0, 1.0]])

    source.transform(trans_init)
    draw_registration_result(source, target, np.identity(4), "Initial settings")

    logger.info("Preprocessing source point cloud")
    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    print_point_cloud_info(source_down, "Downsampled source")

    logger.info("Preprocessing target point cloud")
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    print_point_cloud_info(target_down, "Downsampled target")

    return source, target, source_down, target_down, source_fpfh, target_fpfh


def execute_global_registration(
    source_down, target_down, source_fpfh, target_fpfh, voxel_size
):
    """Execute RANSAC-based global registration between two point clouds.

    Uses RANSAC (Random Sample Consensus) algorithm with feature matching to find
    the initial alignment between source and target point clouds. This method is
    robust to outliers and doesn't require a good initial alignment estimate.

    Args:
        source_down: Downsampled source point cloud.
        target_down: Downsampled target point cloud.
        source_fpfh: FPFH features of the source point cloud.
        target_fpfh: FPFH features of the target point cloud.
        voxel_size: The voxel size used for downsampling, used to compute distance threshold.

    Returns:
        Registration result containing the transformation matrix, fitness score,
        and inlier RMSE from the RANSAC-based feature matching registration.
    """
    distance_threshold = voxel_size * 1.5
    logger.info("RANSAC registration on downsampled point clouds")
    logger.info(
        f"Since the downsampling voxel size is {voxel_size:.3f}, we use a liberal distance threshold {distance_threshold:.3f}"
    )
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down,
        target_down,
        source_fpfh,
        target_fpfh,
        True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3,
        [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold
            ),
        ],
        o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999),
    )
    return result


def refine_registration(source, target, voxel_size: float, initial_transformation):
    """Refine registration using point-to-plane ICP algorithm.

    Performs Iterative Closest Point (ICP) registration with point-to-plane metric
    to refine the initial alignment obtained from global registration. This method
    uses a stricter distance threshold and operates on the original (non-downsampled)
    point clouds for higher accuracy.

    Args:
        source: Original source point cloud.
        target: Original target point cloud.
        voxel_size: The voxel size, used to compute a strict distance threshold.
        initial_transformation: Initial transformation matrix from global registration.

    Returns:
        Registration result containing the refined transformation matrix, fitness score,
        and inlier RMSE from the point-to-plane ICP registration.
    """
    distance_threshold = voxel_size * 0.4
    logger.info("Point-to-plane ICP registration is applied on original point clouds")
    logger.info(
        f"to refine the alignment. This time we use a strict distance threshold {distance_threshold:.3f}"
    )
    if not target.has_normals():
        logger.info("Target point cloud does not have normals, estimating them...")
        target.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        )

    result = o3d.pipelines.registration.registration_icp(
        source,
        target,
        distance_threshold,
        initial_transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
    )
    return result


def main(args: argparse.Namespace):
    """Main function to execute the complete point cloud registration pipeline.

    Orchestrates the full registration workflow:
    1. Loads and prepares source and target point clouds
    2. Executes RANSAC-based global registration for initial alignment
    3. Refines the alignment using point-to-plane ICP
    4. Visualizes the results at each stage

    Args:
        args: Namespace object containing command-line arguments:
            - source: Path to source point cloud file
            - target: Path to target point cloud file
            - voxel_size: Voxel size for downsampling
            - max_iter_icp: Maximum iterations for ICP (currently unused)
    """
    voxel_size = args.voxel_size

    trans_init = np.asarray(
        [
            [0.862, 0.011, -0.507, 0.05],
            [-0.139, 0.967, -0.215, 0.07],
            [0.487, 0.255, 0.835, -0.0004],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    source, target, source_down, target_down, source_fpfh, target_fpfh = (
        prepare_dataset(args.source, args.target, voxel_size, trans_init)
    )

    result_ransac = execute_global_registration(
        source_down, target_down, source_fpfh, target_fpfh, voxel_size
    )
    logger.info(f"RANSAC result: {result_ransac}")
    draw_registration_result(
        source_down,
        target_down,
        result_ransac.transformation,
        "RANSAC global registration on downsampled point clouds",
    )
    draw_registration_result(
        source,
        target,
        result_ransac.transformation,
        "RANSAC global registration on original point clouds",
    )

    result_icp = refine_registration(
        source, target, voxel_size, result_ransac.transformation
    )
    logger.info(f"ICP refinement result: {result_icp}")
    logger.info(f"Estimated matrix:\n{result_icp.transformation}")
    logger.info(
        f"Result fitness: {result_icp.fitness}, inlier RMSE: {result_icp.inlier_rmse}"
    )
    draw_registration_result(
        source, target, result_icp.transformation, "ICP refinement"
    )

    # difference between initial and final transformation
    rot_err, trans_err = transformation_error(result_icp.transformation, trans_init)
    logger.info(f"Rotation error (radians): {rot_err}, Translation error: {trans_err}")
    # compute the rms error between initial and final translation
    registration_rmse = compute_rmse_transformations(
        np.linalg.inv(result_icp.transformation), np.eye(4), source
    )
    logger.info(f"Registration RMSE: {registration_rmse}")


if __name__ == "__main__":
    # tutorial from here https://www.open3d.org/docs/0.10.0/tutorial/Advanced/global_registration.html#:~:text=We%20down%20sample%20the%20point,with%20similar%20local%20geometric%20structures

    # add input file argument
    argparse = argparse.ArgumentParser(description="Global registration")
    argparse.add_argument("--source", type=str, help="source file path", required=True)
    argparse.add_argument("--target", type=str, help="taraget file path", required=True)
    argparse.add_argument(
        "--voxel-size", type=float, help="voxels size for downsampling", default=0.05
    )
    argparse.add_argument(
        "--max_iter_icp", type=int, help="Input file path", default=2000
    )
    argparse.add_argument(
        "-v",
        "--verbose",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set logging level (default: WARNING)",
    )

    input_args = argparse.parse_args()

    # Set logging level based on user selection
    setup_logging(getattr(logging, input_args.verbose))

    main(input_args)
