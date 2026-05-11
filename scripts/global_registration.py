#! /usr/bin/env python3
"""Run global registration between two point clouds."""

import argparse
import logging
import time

import numpy as np
import open3d as o3d

from registration.utils.logging import setup_logging
from registration.utils.metrics import compute_rmse_transformations
from registration.utils.transforms import (
    transformation_error,
    rototranslation_from_rotation_translation,
    perturb_direction,
    generate_random_rotation_matrix,
    rotation_aligning_two_directions,
)
from registration.visualization.viewer import (
    draw_registration_result,
    print_point_cloud_info,
)
from registration.utils.point_cloud import (
    rough_scale_point_cloud,
    rough_scale_point_cloud_from_file,
    # align_centers_from_files,
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


def is_solution_upside_down(transformation: np.ndarray, idx_gravity_axis: int) -> bool:
    """Check if the given transformation results in an upside-down alignment.

    This function examines the rotation component of the provided transformation
    matrix to determine if the direction corresponding to the specified gravity axis
    is inverted (i.e., points in the opposite direction). This can be useful for
    validating registration results against expected orientations.

    Args:
        transformation: A 4x4 transformation matrix to evaluate.
        idx_gravity_axis: The index of the gravity axis (0 for x, 1 for y, 2 for z).

    Returns:
        True if the solution is upside down, False otherwise.

    Raises:
        ValueError: If idx_gravity_axis is not 0, 1, or 2.
    """
    if idx_gravity_axis < 0 or idx_gravity_axis > 2:
        raise ValueError("idx_gravity_axis must be 0 (x), 1 (y), or 2 (z)")

    gravity = np.eye(3)[:, idx_gravity_axis]
    direction = transformation[:3, :3] @ gravity
    return np.dot(direction, gravity) < 0


def prepare_dataset(
    source_file: str,
    target_file: str,
    voxel_size: float,
    trans_init: np.ndarray = np.identity(4),
    correction: np.ndarray = np.identity(4),
) -> tuple:
    """Load and prepare point cloud datasets for registration.

    Loads source and target point clouds from files, applies an initial transformation
    to the source cloud, and preprocesses both clouds by downsampling and computing
    FPFH features for feature-based registration.

    Args:
        source_file: File path to the source point cloud.
        target_file: File path to the target point cloud.
        voxel_size: The size of the voxel for downsampling both point clouds.
        trans_init: Initial transformation matrix to apply to the source cloud (default: identity matrix).
        correction: Correction transformation matrix to apply to both clouds, typically to align to the visual reference frame (default: identity matrix).

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
    source: o3d.geometry.PointCloud = o3d.io.read_point_cloud(source_file)
    source.transform(correction)
    print_point_cloud_info(source, f"Source: {source_file}")
    target: o3d.geometry.PointCloud = o3d.io.read_point_cloud(target_file)
    target.transform(correction)
    print_point_cloud_info(target, f"Target: {target_file}")
    # trans_init = np.asarray([[0.0, 0.0, 1.0, 0.0],
    #                          [1.0, 0.0, 0.0, 0.0],
    #                          [0.0, 1.0, 0.0, 0.0],
    #                          [0.0, 0.0, 0.0, 1.0]])

    source.transform(trans_init)
    frame_size = rough_scale_point_cloud(source)
    draw_registration_result(
        source, target, np.identity(4), "Initial settings", size=frame_size
    )

    logger.info("Preprocessing source point cloud")
    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    print_point_cloud_info(source_down, "Downsampled source")

    logger.info("Preprocessing target point cloud")
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    print_point_cloud_info(target_down, "Downsampled target")

    return source, target, source_down, target_down, source_fpfh, target_fpfh


def execute_global_registration(
    source_down: o3d.geometry.PointCloud,
    target_down: o3d.geometry.PointCloud,
    source_fpfh: o3d.pipelines.registration.Feature,
    target_fpfh: o3d.pipelines.registration.Feature,
    voxel_size: float,
    max_iter_icp: int = 2000,
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
        max_iter_icp: Maximum number of ICP iterations for RANSAC.

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
        source=source_down,
        target=target_down,
        source_feature=source_fpfh,
        target_feature=target_fpfh,
        mutual_filter=True,
        max_correspondence_distance=distance_threshold,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(
            False
        ),
        ransac_n=3,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold
            ),
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(
            max_iter_icp, 0.999
        ),
    )
    return result


def _refine_registration_icp(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    distance_threshold: float,
    initial_transformation: np.ndarray,
) -> o3d.pipelines.registration.RegistrationResult:
    """Refine registration using point-to-plane ICP.

    Estimates target normals if not already present.

    Args:
        source: Source point cloud.
        target: Target point cloud.
        distance_threshold: Maximum correspondence distance.
        initial_transformation: Initial transformation from global registration.

    Returns:
        Registration result from point-to-plane ICP.
    """
    if not target.has_normals():
        logger.info("Target point cloud does not have normals, estimating them...")
        radius_normal = distance_threshold * 2
        target.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
        )

    return o3d.pipelines.registration.registration_icp(
        source,
        target,
        distance_threshold,
        initial_transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
    )


def _refine_registration_gicp(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    distance_threshold: float,
    initial_transformation: np.ndarray,
) -> o3d.pipelines.registration.RegistrationResult:
    """Refine registration using Generalized ICP (GICP).

    Args:
        source: Source point cloud.
        target: Target point cloud.
        distance_threshold: Maximum correspondence distance.
        initial_transformation: Initial transformation from global registration.

    Returns:
        Registration result from GICP.
    """
    return o3d.pipelines.registration.registration_generalized_icp(
        source,
        target,
        distance_threshold,
        initial_transformation,
        o3d.pipelines.registration.TransformationEstimationForGeneralizedICP(),
    )


def refine_registration(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    voxel_size: float,
    initial_transformation: np.ndarray,
    use_gicp: bool = False,
) -> o3d.pipelines.registration.RegistrationResult:
    """Refine registration using ICP or GICP.

    Dispatches to point-to-plane ICP or Generalized ICP (GICP) based on
    the use_gicp flag. Both methods use a strict distance threshold derived
    from the voxel size and operate on the provided (typically full-resolution)
    point clouds for higher accuracy.

    Args:
        source: Source point cloud.
        target: Target point cloud.
        voxel_size: Voxel size used to compute the correspondence distance
            threshold (threshold = voxel_size * 0.4).
        initial_transformation: Initial transformation matrix from global registration.
        use_gicp: If True, use Generalized ICP; otherwise use point-to-plane ICP.

    Returns:
        Registration result containing the refined transformation matrix, fitness
        score, and inlier RMSE.
    """
    distance_threshold = voxel_size * 0.4
    algorithm_name = "Generalized ICP (GICP)" if use_gicp else "Point-to-plane ICP"
    logger.info(f"{algorithm_name} refinement on original point clouds")
    logger.info(f"Using strict distance threshold {distance_threshold:.3f}")

    if use_gicp:
        return _refine_registration_gicp(
            source, target, distance_threshold, initial_transformation
        )
    return _refine_registration_icp(
        source, target, distance_threshold, initial_transformation
    )


def gravity_transformation(
    gravity_direction: np.ndarray, gravity_axis: int = 1
) -> np.ndarray:
    """Compute a transformation matrix to align a given gravity direction.

    This function computes a rotation matrix that aligns the specified gravity
    direction with the desired gravity axis (default is y-axis). This can be useful if the
    gravity vector is given by an IMU sensor in a reference system similar (same up direction)
    to the one of the point cloud.
    It returns a 4x4 transformation matrix that can be applied to point clouds so that the
    point cloud is aligned with the gravity direction.

    Args:
        gravity_direction: A 3D vector representing the measured gravity direction in the point cloud's reference frame.
        gravity_axis: The axis index (0 for x, 1 for y, 2 for z) to align the gravity direction to.

    Returns:
        A 4x4 transformation matrix with null translation that aligns the gravity direction with the specified axis.
    """
    if gravity_axis < 0 or gravity_axis > 2:
        raise ValueError("gravity_axis must be 0 (x), 1 (y), or 2 (z)")

    # @TODO this is to add some noise to the gravity direction
    dst_gravity_direction = perturb_direction(
        np.eye(3)[:, gravity_axis], sigma=np.deg2rad(1)
    )
    dst_gravity_direction = np.eye(3)[:, gravity_axis]

    logger.info(f"dst_gravity_direction: {dst_gravity_direction}")
    gravity_aligned_rotation = rotation_aligning_two_directions(
        gravity_direction, dst_gravity_direction
    )
    gravity_transform = rototranslation_from_rotation_translation(
        gravity_aligned_rotation, np.zeros(3)
    )
    return gravity_transform


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
            - refinement_voxel_size: Voxel size for downsampling during the ICP/GICP refinement step
    """
    voxel_size = args.voxel_size
    refinement_voxel_size = (
        args.refinement_voxel_size
        if args.refinement_voxel_size is not None
        else voxel_size
    )
    frame_size = rough_scale_point_cloud_from_file(args.source)
    min_fitness = args.min_fitness

    # this is just for visualization purposes, the axis of the reference frame of the scan may be different than the one of the window
    # correction = rototranslation_from_rotation_translation(
    #     rot_mat_z(np.deg2rad(90)), np.zeros(3)
    # )
    correction = np.eye(4)

    trans_init = np.asarray(
        # [
        #     [0.862, 0.011, -0.507, 0.05],
        #     [-0.139, 0.967, -0.215, 0.07],
        #     [0.487, 0.255, 0.835, -0.0004],
        #     [0.0, 0.0, 0.0, 1.0],
        # ]
        # [
        #     [1.0, 0.0, 0.0, 2000.05],
        #     [0.0, 1.0, 0.0, 510.07],
        #     [0.0, 0.0, 1.0, -0.0004],
        #     [0.0, 0.0, 0.0, 1.0],
        # ]
        [
            [0.862, 0.011, -0.507, 3.10005 * frame_size],
            [-0.139, 0.967, -0.215, 3.51007 * frame_size],
            [0.487, 0.255, 0.835, -0.4 * frame_size],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    trans_init[:3, :3] = generate_random_rotation_matrix()
    trans_init = np.eye(4)

    # supposing that we know an estimation of the gravity vector (e.g. along the y-axis/up vector)
    # we can try to use it to align the point clouds so that y-axis is aligned
    # here we use the y vector of the initial transformation and perturb it a bit to simulate the
    # direction of the gravity

    idx_gravity_axis = 1

    # gravity_transform = gravity_transformation(
    #     trans_init[:3, idx_gravity_axis], gravity_axis=idx_gravity_axis
    # )
    # trans_init = gravity_transform @ trans_init

    # trans_init = (
    #     align_centers_from_files(args.source, args.target, trans_init, correction)
    #     @ trans_init
    # )

    # logger.debug(f"axis aligned:\n{trans_init @ np.eye(4)[:, idx_gravity_axis]}")

    source, target, source_down, target_down, source_fpfh, target_fpfh = (
        prepare_dataset(args.source, args.target, voxel_size, trans_init, correction)
    )

    result_ransac: o3d.pipelines.registration.RegistrationResult = None
    start = time.time()
    global_attempts = 0
    while not result_ransac or result_ransac.fitness < min_fitness:
        result_ransac = execute_global_registration(
            source_down,
            target_down,
            source_fpfh,
            target_fpfh,
            voxel_size,
            args.max_iter_icp,
        )

        global_attempts += 1    
        logger.info(f"RANSAC attempt {global_attempts} result: {result_ransac}")

        # check if the solution is correct wrt the gravity direction, we want to discard solutions that are upside down
        # @TODO maybe should pass transformation @ init_trans
        upside_down = is_solution_upside_down(
            result_ransac.transformation, idx_gravity_axis
        )
        if result_ransac.fitness >= min_fitness and upside_down:
            logger.warning(
                f"RANSAC attempt {global_attempts} result is upside down with a fitness of {result_ransac.fitness}, discarding."
            )
            result_ransac = None
            continue

    logger.info(
        f"Global found a solution in {global_attempts} attempts, taking {(time.time() - start): .3f} sec."
    )

    draw_registration_result(
        source_down,
        target_down,
        result_ransac.transformation,
        "RANSAC global registration on downsampled point clouds",
        size=frame_size,
    )
    draw_registration_result(
        source,
        target,
        result_ransac.transformation,
        "RANSAC global registration on original point clouds",
        size=frame_size,
    )

    result_icp = refine_registration(
        source,
        target,
        refinement_voxel_size,
        result_ransac.transformation,
        use_gicp=args.use_gicp,
    )
    refinement_label = "GICP" if args.use_gicp else "ICP"
    logger.info(f"{refinement_label} refinement result: {result_icp}")
    logger.info(f"Estimated matrix:\n{result_icp.transformation}")
    logger.info(
        f"Result fitness: {result_icp.fitness}, inlier RMSE: {result_icp.inlier_rmse}"
    )
    draw_registration_result(
        source,
        target,
        result_icp.transformation,
        f"{refinement_label} refinement",
        size=frame_size,
    )
    logger.debug(f"init mat:\n{trans_init}")
    logger.debug(
        f"product of the transformations:\n{result_icp.transformation @ (trans_init)}"
    )
    # NB this only make sense if you are aligning the same model
    # difference between initial and final transformation
    rot_err, trans_err = transformation_error(
        result_icp.transformation, np.linalg.inv(trans_init)
    )
    logger.info(
        f"Rotation error (radians): {rot_err:.4f} (degrees: {np.degrees(rot_err):.4f}), Translation error: {trans_err:.4f}"
    )
    # compute the rms error between initial and final translation (assuming that the points are corresponding)
    registration_rmse = compute_rmse_transformations(
        result_icp.transformation, np.linalg.inv(trans_init), source
    )
    logger.info(f"Registration RMSE: {registration_rmse}")


if __name__ == "__main__":
    # tutorial from here https://www.open3d.org/docs/0.10.0/tutorial/Advanced/global_registration.html#:~:text=We%20down%20sample%20the%20point,with%20similar%20local%20geometric%20structures

    # add input file argument
    parser = argparse.ArgumentParser(description="Global registration")
    parser.add_argument("--source", type=str, help="source file path", required=True)
    parser.add_argument("--target", type=str, help="target file path", required=True)
    parser.add_argument(
        "--min-fitness",
        type=float,
        help="minimum fitness for RANSAC",
        default=0.5,
        required=False,
    )
    parser.add_argument(
        "--voxel-size", type=float, help="voxels size for downsampling", default=0.05
    )
    parser.add_argument(
        "--max_iter_icp", type=int, help="Input file path", default=2000
    )
    parser.add_argument(
        "-v",
        "--verbose",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set logging level (default: WARNING)",
    )

    parser.add_argument(
        "--use-gicp",
        action="store_true",
        default=False,
        help="Use Generalized ICP (GICP) instead of point-to-plane ICP for refinement",
    )

    parser.add_argument(
        "--refinement-voxel-size",
        type=float,
        default=None,
        help=(
            "Voxel size for downsampling during the ICP/GICP refinement step. "
            "If not set, the RANSAC voxel size is reused."
            "Set to 0 to use the original undownsampled point clouds."
        ),
    )

    input_args = parser.parse_args()

    # Set logging level based on user selection
    setup_logging(getattr(logging, input_args.verbose))

    main(input_args)
