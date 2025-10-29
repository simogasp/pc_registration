"""Transformation and rotation utilities."""

from typing import Tuple
import numpy as np
import numpy.typing as npt


def axis_angle_from_rotation(rot_mat: np.ndarray) -> Tuple[np.ndarray, float]:
    """Convert a rotation matrix to axis-angle representation.

    Extracts the rotation axis and angle from a 3x3 rotation matrix using
    the Rodrigues formula. Handles special cases including identity rotation
    (angle ≈ 0) and 180° rotation.

    Args:
        rot_mat: A 3x3 rotation matrix (proper orthogonal matrix with det(R) = +1).

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
    angle = np.arccos(np.clip((np.trace(rot_mat) - 1) / 2.0, -1.0, 1.0))

    if np.isclose(angle, 0.0, atol=1e-8):
        # No rotation → arbitrary axis
        return np.array([1.0, 0.0, 0.0]), 0.0

    if np.isclose(angle, np.pi, atol=1e-6):
        # 180° rotation → extract from diagonal elements
        axis = np.sqrt(np.maximum(np.diagonal(rot_mat) + 1.0, 0.0)) / np.sqrt(2.0)
        axis = axis / np.linalg.norm(axis + eps)
        return axis, angle

    axis = np.array(
        [
            rot_mat[2, 1] - rot_mat[1, 2],
            rot_mat[0, 2] - rot_mat[2, 0],
            rot_mat[1, 0] - rot_mat[0, 1],
        ]
    ) / (2.0 * np.sin(angle))
    axis = axis / np.linalg.norm(axis + eps)
    return axis, angle


def rotation_error_angle(rot_est: np.ndarray, rot_gt: np.ndarray) -> float:
    """Calculate the angular error between two rotation matrices.

    Computes the angle (in radians) of the relative rotation between an
    estimated rotation matrix and a ground truth rotation matrix. This is
    equivalent to finding the rotation angle needed to transform R_est to R_gt.

    Args:
        rot_est: Estimated 3x3 rotation matrix.
        rot_gt: Ground truth 3x3 rotation matrix.

    Returns:
        The rotation error angle in radians, in the range [0, π].

    Note:
        The error is computed as arccos((trace(R_est @ R_gt^T) - 1) / 2),
        which gives the geodesic distance on SO(3).
    """
    rot_err = rot_est @ rot_gt.T
    trace = np.clip((np.trace(rot_err) - 1) / 2.0, -1.0, 1.0)
    angle = np.arccos(trace)  # radians
    return angle


def translation_error(
    rot_est: np.ndarray, t_est: np.ndarray, rot_gt: np.ndarray, t_gt: np.ndarray
) -> Tuple[float, npt.NDArray[np.floating]]:
    """Calculate the translation error between two transformations.

    Computes the translation error accounting for the rotation difference.
    The error is calculated as t_est - R_err @ t_gt, where R_err is the
    relative rotation between estimated and ground truth rotations.

    Args:
        rot_est: Estimated 3x3 rotation matrix.
        t_est: Estimated 3D translation vector.
        rot_gt: Ground truth 3x3 rotation matrix.
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
    rot_err = rot_est @ rot_gt.T
    t_err = t_est - rot_err @ t_gt
    norm = float(np.linalg.norm(t_err))
    return norm, t_err  # (norm, vector)


def transformation_error(t_est: np.ndarray, t_gt: np.ndarray) -> Tuple[float, float]:
    """Calculate both rotation and translation errors between two transformations.

    Decomposes two 4x4 transformation matrices into rotation and translation
    components, then computes the angular error between rotations and the
    translation error magnitude.

    Args:
        t_est: Estimated 4x4 transformation matrix (homogeneous coordinates).
        t_gt: Ground truth 4x4 transformation matrix (homogeneous coordinates).

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
    if t_est.shape != (4, 4) or t_gt.shape != (4, 4):
        raise ValueError("Both T_est and T_gt must be 4x4 matrices.")

    rot_est = t_est[:3, :3]
    tra_est = t_est[:3, 3]
    rot_gt = t_gt[:3, :3]
    tra_gt = t_gt[:3, 3]
    rot_err = rotation_error_angle(rot_est, rot_gt)
    trans_err, trans_vec = translation_error(rot_est, tra_est, rot_gt, tra_gt)
    return rot_err, trans_err


def generate_random_rotation_matrix() -> np.ndarray:
    """Generate a random 3x3 rotation matrix.

    Uses QR method to generate a random rotation matrix uniformly sampled from SO(3).

    Returns:
        A random 3x3 rotation matrix.
    """
    random_state = np.random.default_rng()
    A = random_state.normal(size=(3, 3))
    Q, R = np.linalg.qr(A)

    # Ensure a proper rotation (det(Q) = +1)
    if np.linalg.det(Q) < 0:
        Q[:, 2] *= -1

    return Q


def is_rotation_matrix(mat: np.ndarray) -> bool:
    """Check if a matrix is a valid rotation matrix.

    A valid rotation matrix is orthogonal (R.T @ R = I) and has a determinant of +1.

    Args:
        mat: A square NxN matrix to check.

    Returns:
        True if the matrix is a valid rotation matrix, False otherwise.
    """

    # Check orthogonality
    if not np.allclose(mat.T @ mat, np.eye(3)):
        return False
    # Check determinant
    if not np.isclose(np.linalg.det(mat), 1):
        return False
    return True
