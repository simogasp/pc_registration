"""Common utilities for point cloud registration scripts.

This module provides shared functionality for loading scan pairs,
transformation matrices, and performing pairwise registration.
"""

import json
import logging
from pathlib import Path

import numpy as np
import open3d as o3d

logger = logging.getLogger(__name__)

# Normal estimation parameters
NORMAL_ESTIMATION_RADIUS_MULTIPLIER = 2  # Multiplier for voxel size to determine radius
# Fraction of the oriented bounding box diagonal used as the normal estimation radius
# when voxel_size=0 (downsampling disabled). This is scale-independent and avoids
# hardcoding a unit-specific fallback value.
NORMAL_ESTIMATION_RADIUS_FRACTION = 0.01
NORMAL_ESTIMATION_MAX_NEIGHBORS = 30  # Maximum nearest neighbors for normal estimation


def compute_normal_estimation_radius(
    pcd: o3d.geometry.PointCloud, voxel_size: float
) -> float:
    """Compute a scale-invariant normal estimation search radius.

    When downsampling is active (voxel_size > 0) the radius is derived from
    the voxel size via NORMAL_ESTIMATION_RADIUS_MULTIPLIER, which is the
    standard Open3D convention.

    When no downsampling is used (voxel_size == 0) the radius is computed as
    NORMAL_ESTIMATION_RADIUS_FRACTION of the oriented bounding box (OBB)
    diagonal. This makes the radius proportional to the actual extent of the
    point cloud and independent of the unit system.

    Args:
        pcd: Point cloud for which to compute the radius.
        voxel_size: Voxel size used for downsampling (0 if no downsampling).

    Returns:
        Normal estimation search radius.
    """
    if voxel_size > 0:
        return voxel_size * NORMAL_ESTIMATION_RADIUS_MULTIPLIER

    obb = pcd.get_oriented_bounding_box()
    diagonal = float(np.linalg.norm(obb.extent))
    return diagonal * NORMAL_ESTIMATION_RADIUS_FRACTION


def find_scan_pairs(data_dir: Path) -> list[tuple[Path, Path]]:
    """Find all matching pairs of .ply and .json files in a directory.

    Args:
        data_dir: Directory containing the scan files.

    Returns:
        List of tuples (ply_path, json_path) for matching pairs, sorted by filename.

    Raises:
        ValueError: If a .ply file has no corresponding .json file.
    """
    ply_files = sorted(data_dir.glob("*.ply"))
    json_files = sorted(data_dir.glob("*.json"))

    # Create dictionaries keyed by stem (filename without extension)
    ply_dict = {p.stem: p for p in ply_files}
    json_dict = {j.stem: j for j in json_files}

    # Find matching pairs
    pairs = []
    for stem in sorted(ply_dict.keys(), key=lambda x: int(x) if x.isdigit() else x):
        if stem not in json_dict:
            # try using adding _pose suffix to match JSON files that have this convention
            test_stem = f"{stem}_pose"
            if test_stem in json_dict:
                pairs.append((ply_dict[stem], json_dict[test_stem]))
                logger.warning(
                    f"Matched {ply_dict[stem].name} to {json_dict[test_stem].name} "
                    f"using fallback stem '{test_stem}' (original stem '{stem}' not found in JSON files)"
                )
                continue
            else:
                raise ValueError(f"Missing JSON file for {ply_dict[stem]}")
        pairs.append((ply_dict[stem], json_dict[stem]))

    logger.info(f"Found {len(pairs)} matching scan pairs")
    return pairs


def load_transformation_matrix(json_path: Path) -> np.ndarray:
    """Load a 4x4 transformation matrix from a JSON file.

    Args:
        json_path: Path to JSON file containing the "H" transformation matrix.

    Returns:
        4x4 numpy array representing the transformation matrix.

    Raises:
        ValueError: If the JSON file doesn't contain a valid "H" field.
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    if "H" not in data:
        raise ValueError(f"JSON file {json_path} does not contain 'H' field")

    H = np.array(data["H"])

    if H.shape != (4, 4):
        raise ValueError(
            f"Transformation matrix in {json_path} has shape {H.shape}, expected (4, 4)"
        )

    return H


def matrix_to_json_dict(matrix: np.ndarray) -> dict:
    """Serialise a 4x4 matrix to the JSON dict format used by load_transformation_matrix.

    Args:
        matrix: 4x4 numpy array.

    Returns:
        Dict with key "H" mapping to a list-of-rows representation.
    """
    return {"H": matrix.tolist()}


def write_pose_json(matrix: np.ndarray, output_path: Path) -> None:
    """Write a single 4x4 pose matrix to a JSON file.

    The output format is compatible with load_transformation_matrix.

    Args:
        matrix: 4x4 numpy array representing the pose.
        output_path: Destination JSON file path.
    """
    with open(output_path, "w") as f:
        json.dump(matrix_to_json_dict(matrix), f, indent=4)


def load_point_cloud(
    ply_path: Path,
    voxel_size: float = 0.0,
    estimate_normals: bool = True,
) -> o3d.geometry.PointCloud:
    """Load a point cloud from a PLY file.

    Args:
        ply_path: Path to PLY file.
        voxel_size: If > 0, downsample point cloud with this voxel size.
        estimate_normals: If True, estimate normals when not present.

    Returns:
        Point cloud (possibly downsampled, with normals if requested).
    """
    pcd = o3d.io.read_point_cloud(str(ply_path))
    if not pcd.has_points():
        raise ValueError(f"Point cloud {ply_path} is empty")

    # Downsample if requested
    if voxel_size > 0:
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)

    # Estimate normals if not present
    if estimate_normals and not pcd.has_normals():
        radius = compute_normal_estimation_radius(pcd, voxel_size)
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=radius, max_nn=NORMAL_ESTIMATION_MAX_NEIGHBORS
            )
        )

    return pcd


def pairwise_registration(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    max_correspondence_distance: float,
    init_transformation: np.ndarray = np.eye(4),
    max_iteration: int = 30,
    verbose: bool = True,
    use_generalized_icp: bool = False,
) -> tuple[np.ndarray, o3d.pipelines.registration.RegistrationResult]:
    """Perform pairwise ICP or GICP registration between two point clouds.

    Args:
        source: Source point cloud.
        target: Target point cloud.
        max_correspondence_distance: Maximum correspondence distance for ICP.
        init_transformation: Initial transformation guess.
        max_iteration: Maximum number of ICP iterations.
        verbose: If True, log registration details.
        use_generalized_icp: If True, use GICP; otherwise use classic ICP.

    Returns:
        Tuple of (transformation matrix, registration result).
    """
    algorithm_name = "Generalized ICP (GICP)" if use_generalized_icp else "ICP"

    if verbose:
        logger.debug(f"  Pairwise {algorithm_name} registration...")

    # Select registration method
    if use_generalized_icp:
        result = o3d.pipelines.registration.registration_generalized_icp(
            source,
            target,
            max_correspondence_distance,
            init_transformation,
            o3d.pipelines.registration.TransformationEstimationForGeneralizedICP(),
            o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=max_iteration
            ),
        )
    else:
        # Use point-to-plane ICP for better convergence
        result = o3d.pipelines.registration.registration_icp(
            source,
            target,
            max_correspondence_distance,
            init_transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=max_iteration
            ),
        )

    if verbose:
        logger.debug(
            f"    Fitness: {result.fitness:.4f}, RMSE: {result.inlier_rmse:.4f}"
        )

    return result.transformation, result


def remove_outliers(
    pcd: o3d.geometry.PointCloud, nb_neighbors: int = 20, std_ratio: float = 2.0
) -> o3d.geometry.PointCloud:
    """Remove outlier points using statistical outlier removal.

    This method removes points that are far from their neighbors compared to the
    average distance for all points. It computes the average distance from each
    point to its k-nearest neighbors and removes points whose distance is more than
    std_ratio * standard_deviation away from the mean.

    Args:
        pcd: Input point cloud.
        nb_neighbors: Number of neighbors to consider for computing average distance.
        std_ratio: Standard deviation ratio threshold. Points with average distance
            larger than (mean + std_ratio * std_dev) are considered outliers.

    Returns:
        Point cloud with outliers removed.
    """
    points_before = len(pcd.points)
    logger.info(
        f"Removing outliers (nb_neighbors={nb_neighbors}, std_ratio={std_ratio})..."
    )

    cleaned_pcd, _ = pcd.remove_statistical_outlier(
        nb_neighbors=nb_neighbors, std_ratio=std_ratio
    )

    points_removed = points_before - len(cleaned_pcd.points)
    removal_percent = (points_removed / points_before) * 100 if points_before > 0 else 0

    logger.info(
        f"Removed {points_removed} outlier points ({removal_percent:.2f}%), "
        f"{len(cleaned_pcd.points)} points remaining"
    )

    return cleaned_pcd


def filter_distant_points(
    pcd: o3d.geometry.PointCloud,
    max_distance: float | None = None,
    percentile: float = 99.0,
    use_centroid: bool = True,
) -> o3d.geometry.PointCloud:
    """Filter out points that are too far from the point cloud centroid or origin.

    This function removes outlier points that are at extreme distances, which often
    occur near LiDAR range limits and can create unnecessarily large bounding boxes.
    Two filtering modes are available:
    1. Absolute distance threshold (if max_distance is specified)
    2. Percentile-based filtering (removes points beyond the Nth percentile distance)

    Args:
        pcd: Input point cloud.
        max_distance: Maximum distance threshold. If None, uses percentile-based filtering.
        percentile: Distance percentile to use as threshold (default: 99.0).
            Only used if max_distance is None. Points beyond this percentile are removed.
        use_centroid: If True, compute distances from centroid; if False, from origin.

    Returns:
        Point cloud with distant points removed.
    """
    points = np.asarray(pcd.points)
    points_before = len(points)

    # Compute reference point (centroid or origin)
    if use_centroid:
        # Use median instead of mean for robustness against outliers
        reference = np.median(points, axis=0)
        ref_name = "centroid (median)"
    else:
        reference = np.zeros(3)
        ref_name = "origin"

    # Compute distances from reference point
    distances = np.linalg.norm(points - reference, axis=1)

    # Determine threshold
    if max_distance is not None:
        threshold = max_distance
        logger.info(f"Filtering points beyond {threshold:.2f} from {ref_name}...")
    else:
        threshold = np.percentile(distances, percentile)
        logger.info(
            f"Filtering points beyond {percentile}th percentile "
            f"({threshold:.2f} from {ref_name})..."
        )

    # Filter points
    mask = distances <= threshold
    filtered_points = points[mask]

    # Create new point cloud with filtered points
    filtered_pcd = o3d.geometry.PointCloud()
    filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points)

    # Copy colors if present
    if pcd.has_colors():
        colors = np.asarray(pcd.colors)
        filtered_pcd.colors = o3d.utility.Vector3dVector(colors[mask])

    # Copy normals if present
    if pcd.has_normals():
        normals = np.asarray(pcd.normals)
        filtered_pcd.normals = o3d.utility.Vector3dVector(normals[mask])

    points_removed = points_before - len(filtered_points)
    removal_percent = (points_removed / points_before) * 100 if points_before > 0 else 0

    logger.info(
        f"Removed {points_removed} distant points ({removal_percent:.2f}%), "
        f"{len(filtered_points)} points remaining"
    )
    logger.info(
        f"Distance statistics - Min: {distances.min():.2f}, "
        f"Mean: {distances.mean():.2f}, Max: {distances.max():.2f}, "
        f"Threshold: {threshold:.2f}"
    )

    return filtered_pcd


def rotation_error_degrees(R1: np.ndarray, R2: np.ndarray) -> float:
    """Compute rotation error between two rotation matrices in degrees.

    Args:
        R1: First 3x3 rotation matrix.
        R2: Second 3x3 rotation matrix.

    Returns:
        Rotation error in degrees (angular distance).
    """
    R_rel = R2.T @ R1
    trace = np.trace(R_rel)
    cos_theta = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
    theta_rad = np.arccos(cos_theta)
    theta_deg = np.degrees(theta_rad)
    return float(theta_deg)


def translation_error(t1: np.ndarray, t2: np.ndarray) -> float:
    """Compute translation error (Euclidean distance).

    Args:
        t1: First 3D translation vector.
        t2: Second 3D translation vector.

    Returns:
        Euclidean distance between translations.
    """
    return float(np.linalg.norm(t1 - t2))


def load_poses_from_file(poses_file: str, num_scans: int) -> list[np.ndarray] | None:
    """Load all poses from a JSON file.

    Args:
        poses_file: Path to JSON file containing poses.
        num_scans: Expected number of scans for validation.

    Returns:
        List of pose matrices, or None on error.
    """
    poses_path = Path(poses_file)
    if not poses_path.exists():
        logger.error(f"Poses file not found: {poses_file}")
        return None

    try:
        with open(poses_path, "r") as f:
            data = json.load(f)

        if "poses" not in data:
            logger.error("Invalid poses file format: missing 'poses' key")
            return None

        poses = [np.array(pose) for pose in data["poses"]]
        logger.info(f"Loaded {len(poses)} poses from {poses_file}")

        if len(poses) != num_scans:
            logger.warning(
                f"Number of poses ({len(poses)}) does not match "
                f"number of scans ({num_scans})"
            )

        return poses

    except json.JSONDecodeError as e:
        logger.error(f"Error loading poses file: {e}")
        return None


def save_poses_to_file(poses: list[np.ndarray], output_path: Path):
    """Save poses to a JSON file.

    Args:
        poses: List of 4x4 transformation matrices.
        output_path: Output file path.
    """
    poses_data = {"poses": [pose.tolist() for pose in poses]}

    with open(output_path, "w") as f:
        json.dump(poses_data, f, indent=2)

    logger.info(f"Saved {len(poses)} poses to {output_path.name}")


POSE_MATRIX_SIZE = 4
POSE_FLOATS_PER_LINE = POSE_MATRIX_SIZE * POSE_MATRIX_SIZE


def parse_pose_line(line: str, line_number: int) -> np.ndarray:
    """Parse one line of a flat-text poses file into a 4x4 matrix.

    Args:
        line: Space-separated string of 16 floats.
        line_number: 1-based line index used in error messages.

    Returns:
        4x4 numpy array in row-major order.

    Raises:
        ValueError: If the line does not contain exactly 16 floats.
    """
    values = line.split()
    if len(values) != POSE_FLOATS_PER_LINE:
        raise ValueError(
            f"Line {line_number}: expected {POSE_FLOATS_PER_LINE} values, "
            f"got {len(values)}"
        )
    return np.array([float(v) for v in values], dtype=float).reshape(
        POSE_MATRIX_SIZE, POSE_MATRIX_SIZE
    )


def read_poses_from_txt(poses_file: Path) -> list[np.ndarray]:
    """Read all poses from a flat-text poses file.

    Blank lines and lines starting with '#' are skipped. Each remaining line
    must contain exactly 16 space-separated floats representing a 4x4
    transformation matrix in row-major order.

    Args:
        poses_file: Path to the text file.

    Returns:
        List of 4x4 numpy arrays, one per non-empty line.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If any line has an unexpected number of values.
    """
    if not poses_file.exists():
        raise FileNotFoundError(f"Poses file not found: {poses_file}")

    poses = []
    line_number = 0
    with open(poses_file, "r") as f:
        for raw_line in f:
            line_number += 1
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            poses.append(parse_pose_line(line, line_number))

    logger.info(f"Read {len(poses)} poses from {poses_file}")
    return poses


def scale_translation(matrix: np.ndarray, scale: float) -> np.ndarray:
    """Return a copy of a 4x4 pose matrix with the translation scaled.

    Only the first three elements of the last column (tx, ty, tz) are
    affected; the rotation block and the homogeneous row are left unchanged.

    Args:
        matrix: 4x4 numpy array representing the pose.
        scale: Multiplicative factor applied to the translation components.

    Returns:
        New 4x4 numpy array with scaled translation.
    """
    scaled = matrix.copy()
    scaled[:3, 3] *= scale
    return scaled


def load_and_transform_scan(
    ply_path: Path, transformation: np.ndarray
) -> o3d.geometry.PointCloud:
    """Load a point cloud at full resolution and apply a rigid transformation.

    Normals are not estimated, making this suitable for map fusion where ICP
    is not performed on the loaded cloud.

    Args:
        ply_path: Path to PLY file.
        transformation: 4x4 transformation matrix to apply.

    Returns:
        Transformed point cloud at full resolution.

    Raises:
        ValueError: If the point cloud is empty.
    """
    pcd = load_point_cloud(ply_path, voxel_size=0.0, estimate_normals=False)
    pcd.transform(transformation)
    logger.debug(f"Loaded and transformed {ply_path.name}: {len(pcd.points)} points")
    return pcd


def save_point_cloud_binary(pcd: o3d.geometry.PointCloud, output_path: Path) -> None:
    """Save a point cloud in binary PLY format.

    Args:
        pcd: Point cloud to save.
        output_path: Output file path.

    Raises:
        IOError: If Open3D reports a write failure.
    """
    success = o3d.io.write_point_cloud(
        str(output_path), pcd, write_ascii=False, compressed=False
    )

    if not success:
        raise OSError(f"Failed to save point cloud to {output_path}")

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info(f"Saved {output_path.name} ({file_size_mb:.2f} MB)")


def save_parameters(params: dict, output_path: Path) -> None:
    """Save execution parameters to a JSON file.

    Args:
        params: Dictionary of parameters to serialise.
        output_path: Output file path.
    """
    with open(output_path, "w") as f:
        json.dump(params, f, indent=2)

    logger.info(f"Saved execution parameters to {output_path.name}")
