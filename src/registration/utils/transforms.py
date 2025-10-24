"""Transformation and rotation utilities."""

from typing import Tuple
import numpy as np
import numpy.typing as npt


def axis_angle_from_rotation(R: np.ndarray) -> Tuple[np.ndarray, float]:
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

    axis = np.array(
        [
            R[2, 1] - R[1, 2],
            R[0, 2] - R[2, 0],
            R[1, 0] - R[0, 1],
        ]
    ) / (2.0 * np.sin(angle))
    axis = axis / np.linalg.norm(axis + eps)
    return axis, angle


def rotation_error_angle(R_est: np.ndarray, R_gt: np.ndarray) -> float:
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
    angle = np.arccos(trace)  # radians
    return angle


def translation_error(
    R_est: np.ndarray, t_est: np.ndarray, R_gt: np.ndarray, t_gt: np.ndarray
) -> Tuple[float, npt.NDArray[np.floating]]:
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
    norm = float(np.linalg.norm(t_err))
    return norm, t_err  # (norm, vector)


def transformation_error(T_est: np.ndarray, T_gt: np.ndarray) -> Tuple[float, float]:
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
