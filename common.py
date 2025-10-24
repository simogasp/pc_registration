import open3d as o3d
import numpy as np
import copy
import logging
import sys


# Custom formatter with colored log levels
class ColoredFormatter(logging.Formatter):
    """Custom formatter that adds color to log level names only."""
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record):
        # Save the original levelname
        levelname = record.levelname
        
        # Add color to levelname only
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{levelname}{self.RESET}"
        
        # Format the message
        result = super().format(record)
        
        # Restore original levelname
        record.levelname = levelname
        
        return result


def setup_logging(level=logging.INFO):
    """Set up logging configuration with colored output.
    
    Args:
        level: The logging level (default: logging.INFO).
    
    Returns:
        The root logger instance.
    """
    # Create formatter with timestamp and level
    formatter = ColoredFormatter(
        fmt='[%(asctime)s][%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Set up console handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.handlers.clear()  # Remove any existing handlers
    root_logger.addHandler(handler)
    
    return root_logger


# Initialize logging when module is imported
setup_logging()


def draw_registration_result(source, target, transformation, window_name : str):
    """Visualize the registration result between source and target point clouds.

    Args:
        source: Source point cloud to be transformed.
        target: Target point cloud (reference).
        transformation: 4x4 transformation matrix to apply to source.
        window_name: Name of the visualization window.
    """
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp], window_name=window_name)
                                      # zoom=0.4459,
                                      # front=[0.9288, -0.2951, -0.2242],
                                      # lookat=[1.6784, 2.0612, 1.4451],
                                      # up=[-0.3402, -0.9189, -0.1996])

# print stats about a point cloud
def print_point_cloud_info(pcd: o3d.geometry.PointCloud, name: str):
    """Prints information about a point cloud including number of points and bounding box.

    Args:
        pcd: The point cloud to analyze.
        name: Name of the point cloud for identification in the output.
    """
    num_points = len(pcd.points)
    aabb = pcd.get_axis_aligned_bounding_box()
    obb = pcd.get_oriented_bounding_box()
    logging.info(f"Point Cloud '{name}':")
    logging.info(f"  Number of points: {num_points}")
    logging.info(f"  Axis-Aligned Bounding Box: min {aabb.min_bound}, max {aabb.max_bound}")
    logging.info(f"  Oriented Bounding Box: center {obb.center}, extent {obb.extent}")

def axis_angle_from_rotation(R: np.ndarray) -> tuple[np.ndarray, float]:
    """Convert a rotation matrix to axis-angle representation.
    
    Extracts the rotation axis and angle from a 3x3 rotation matrix using
    the Rodrigues formula. Handles special cases including identity rotation
    (angle ≈ 0) and 180° rotation.
    
    Args:
        R: A 3x3 rotation matrix (proper orthogonal matrix with det(R) = +1).
    
    Returns:
        A tuple containing:
            - axis: A (3,) unit vector representing the rotation axis. 
                   Undefined/arbitrary if rotation is approximately identity.
            - angle: Rotation angle in radians, in the range [0, π].
    
    Note:
        For very small rotations (angle ≈ 0), the axis is set to [1, 0, 0]
        by convention. For 180° rotations, the axis is extracted from the
        diagonal elements of the rotation matrix.
    """
    eps = 1e-12
    angle = np.arccos(np.clip((np.trace(R) - 1) / 2.0, -1.0, 1.0))

    if np.isclose(angle, 0.0, atol=1e-8):
        # No rotation → arbitrary axis
        return np.array([1.0, 0.0, 0.0]), 0.0

    if np.isclose(angle, np.pi, atol=1e-6):
        # 180° rotation → extract from diagonal elements
        axis = np.sqrt(np.maximum(np.diagonal(R) + 1.0, 0.0)) / np.sqrt(2.0)
        axis = axis / np.linalg.norm(axis + eps)
        return axis, angle

    axis = np.array([
        R[2, 1] - R[1, 2],
        R[0, 2] - R[2, 0],
        R[1, 0] - R[0, 1],
    ]) / (2.0 * np.sin(angle))
    axis = axis / np.linalg.norm(axis + eps)
    return axis, angle

def rotation_error_angle(R_est : np.ndarray, R_gt: np.ndarray) -> float:
    """Calculate the angular error between two rotation matrices.
    
    Computes the angle (in radians) of the relative rotation between an 
    estimated rotation matrix and a ground truth rotation matrix. This is
    equivalent to finding the rotation angle needed to transform R_est to R_gt.
    
    Args:
        R_est: Estimated 3x3 rotation matrix.
        R_gt: Ground truth 3x3 rotation matrix.
    
    Returns:
        The rotation error angle in radians, in the range [0, π].
    
    Note:
        The error is computed as arccos((trace(R_est @ R_gt^T) - 1) / 2),
        which gives the geodesic distance on SO(3).
    """
    R_err = R_est @ R_gt.T
    trace = np.clip((np.trace(R_err) - 1) / 2.0, -1.0, 1.0)
    angle = np.arccos(trace)   # radians
    return angle

def translation_error(R_est : np.ndarray, t_est : np.ndarray, R_gt : np.ndarray, t_gt : np.ndarray) -> tuple[float, np.ndarray]:
    """Calculate the translation error between two transformations.
    
    Computes the translation error accounting for the rotation difference.
    The error is calculated as t_est - R_err @ t_gt, where R_err is the
    relative rotation between estimated and ground truth rotations.
    
    Args:
        R_est: Estimated 3x3 rotation matrix.
        t_est: Estimated 3D translation vector.
        R_gt: Ground truth 3x3 rotation matrix.
        t_gt: Ground truth 3D translation vector.
    
    Returns:
        A tuple containing:
            - norm: The Euclidean norm (magnitude) of the translation error.
            - vector: The 3D translation error vector.
    
    Note:
        This function correctly accounts for the rotation difference when
        computing translation error, ensuring the error is measured in the
        same reference frame.
    """
    R_err = R_est @ R_gt.T
    t_err = t_est - R_err @ t_gt
    return np.linalg.norm(t_err), t_err  # (norm, vector)

def transformation_error(T_est: np.ndarray, T_gt: np.ndarray) -> tuple[float, float]:
    """Calculate both rotation and translation errors between two transformations.
    
    Decomposes two 4x4 transformation matrices into rotation and translation
    components, then computes the angular error between rotations and the
    translation error magnitude.
    
    Args:
        T_est: Estimated 4x4 transformation matrix (homogeneous coordinates).
        T_gt: Ground truth 4x4 transformation matrix (homogeneous coordinates).
    
    Returns:
        A tuple containing:
            - rot_err: Rotation error angle in radians, in the range [0, π].
            - trans_err: Translation error magnitude (Euclidean norm).
    
    Raises:
        ValueError: If either T_est or T_gt is not a 4x4 matrix.
    
    Note:
        The transformation matrices should be in the standard form:
        T = [[R, t],
             [0, 1]]
        where R is a 3x3 rotation matrix and t is a 3D translation vector.
    """
    # check the matrices are 4x4
    if T_est.shape != (4, 4) or T_gt.shape != (4, 4):
        raise ValueError("Both T_est and T_gt must be 4x4 matrices.")
    
    R_est = T_est[:3, :3]
    t_est = T_est[:3, 3]
    R_gt = T_gt[:3, :3]
    t_gt = T_gt[:3, 3]
    rot_err = rotation_error_angle(R_est, R_gt)
    trans_err, trans_vec = translation_error(R_est, t_est, R_gt, t_gt)
    return rot_err, trans_err

def compute_rmse_between_point_clouds(pcd1: o3d.geometry.PointCloud, pcd2: o3d.geometry.PointCloud) -> float:
    """Compute the RMSE between two point clouds.
    
    Args:
        pcd1: First point cloud.
        pcd2: Second point cloud.
    
    Returns:
        The root mean square error (RMSE) between corresponding points in the two point clouds.
    
    Raises:
        ValueError: If the two point clouds do not have the same number of points.
    """
    if len(pcd1.points) != len(pcd2.points):
        raise ValueError("Point clouds must have the same number of points to compute RMSE.")
    
    points1 = np.asarray(pcd1.points)
    points2 = np.asarray(pcd2.points)
    diffs = points1 - points2
    squared_diffs = np.sum(diffs ** 2, axis=1)
    mse = np.mean(squared_diffs)
    rmse = np.sqrt(mse)
    return rmse

def compute_rmse_transformations(T_est: np.ndarray, T_gt: np.ndarray, pcd: o3d.geometry.PointCloud) -> float:
    """Compute the RMSE between two transformations applied to a point cloud.
    
    Args:
        T_est: Estimated transformation (4x4 matrix).
        T_gt: Ground truth transformation (4x4 matrix).
        pcd: Point cloud to which the transformations will be applied.
    
    Returns:
        The root mean square error (RMSE) between the point clouds obtained
        by applying T_est and T_gt to the input point cloud.
    """
    pcd_est = copy.deepcopy(pcd)
    pcd_gt = copy.deepcopy(pcd)
    pcd_est.transform(T_est)
    pcd_gt.transform(T_gt)
    rmse = compute_rmse_between_point_clouds(pcd_est, pcd_gt)
    return rmse