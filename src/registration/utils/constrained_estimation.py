"""Constrained rigid transformation estimation for point cloud registration.

This module provides functions for estimating rigid transformations where the
rotation is constrained to a rotation around a specified axis. This is useful
when external information (e.g., the gravity direction from an IMU) constrains
the admissible rotations, reducing the rotation search space from 3-DOF to 1-DOF.

The main entry point is estimate_constrained_rototranslation, which mirrors the
interface of Open3D's TransformationEstimation and is designed to be embedded in
a custom RANSAC loop.
"""

from typing import Tuple

import numpy as np
import open3d as o3d

from registration.utils.transforms import (
    rotation_matrix_from_axis_angle,
    rototranslation_from_rotation_translation,
)


def _normalize_axis(axis: np.ndarray) -> np.ndarray:
    """Normalize a 3D vector to a unit vector.

    Args:
        axis: A 3D vector representing the rotation axis.

    Returns:
        The normalized unit vector.

    Raises:
        ValueError: If the axis is not a 3D vector or is the zero vector.
    """
    if axis.shape != (3,):
        raise ValueError(f"Rotation axis must be a 3D vector, got shape {axis.shape}.")
    norm = float(np.linalg.norm(axis))
    if norm < 1e-12:
        raise ValueError("Rotation axis must be a non-zero vector.")
    return axis / norm


def _extract_correspondence_points(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    correspondences: o3d.utility.Vector2iVector,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract matched point arrays from source and target using a correspondence set.

    Args:
        source: Source point cloud.
        target: Target point cloud.
        correspondences: Pairs of (source_index, target_index) correspondences.

    Returns:
        Tuple of (source_points, target_points), each an (N, 3) array.

    Raises:
        ValueError: If fewer than 2 correspondences are provided.
    """
    corres = np.asarray(correspondences)
    if len(corres) < 2:
        raise ValueError(
            f"At least 2 correspondences are required, got {len(corres)}. "
            "A single correspondence does not constrain the rotation angle "
            "after centroid subtraction."
        )
    source_pts = np.asarray(source.points)[corres[:, 0]]
    target_pts = np.asarray(target.points)[corres[:, 1]]
    return source_pts, target_pts


def _project_perpendicular_to_axis(
    points: np.ndarray, unit_axis: np.ndarray
) -> np.ndarray:
    """Project an (N, 3) array of points onto the plane perpendicular to a unit axis.

    Each point is decomposed into its axial component (along unit_axis) and its
    perpendicular component. This function returns only the perpendicular part.

    Args:
        points: An (N, 3) array of 3D points.
        unit_axis: A unit vector defining the axis to project out.

    Returns:
        An (N, 3) array where each row has zero component along unit_axis.
    """
    axial_components = (points @ unit_axis)[:, np.newaxis] * unit_axis
    return points - axial_components


def _estimate_rotation_angle_around_axis(
    source_centered_perp: np.ndarray,
    target_centered_perp: np.ndarray,
    unit_axis: np.ndarray,
) -> float:
    """Estimate the rotation angle minimizing squared distances between projected pairs.

    The closed-form solution minimizes sum_i || q'_perp_i - R(theta) p'_perp_i ||^2
    over theta, where primes denote centroid-subtracted points and perp denotes
    projection onto the plane perpendicular to the axis.

    The solution is theta = atan2(B, A) where:

        A = sum_i  p'_perp_i . q'_perp_i
        B = unit_axis . sum_i (p'_perp_i x q'_perp_i)

    Args:
        source_centered_perp: (N, 3) source points, centroid-subtracted and
            projected perpendicular to the rotation axis.
        target_centered_perp: (N, 3) target points, centroid-subtracted and
            projected perpendicular to the rotation axis.
        unit_axis: Unit vector defining the rotation axis.

    Returns:
        Estimated rotation angle in radians, in the range (-pi, pi]. Returns 0.0
        when A and B both vanish (degenerate case where all points lie on the axis).
    """
    A = float(np.sum(source_centered_perp * target_centered_perp))
    cross_sum = np.cross(source_centered_perp, target_centered_perp).sum(axis=0)
    B = float(unit_axis @ cross_sum)

    if np.isclose(A, 0.0) and np.isclose(B, 0.0):
        return 0.0

    return float(np.arctan2(B, A))


def estimate_constrained_rototranslation(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    correspondences: o3d.utility.Vector2iVector,
    rotation_axis: np.ndarray,
) -> np.ndarray:
    """Estimate a rigid transformation with the rotation constrained to a fixed axis.

    The rotation component is restricted to a rotation around the specified axis;
    the translation is unconstrained. The interface mirrors Open3D's
    TransformationEstimation.compute_transformation and is designed for use in a
    custom RANSAC framework where an axis constraint (e.g., gravity direction)
    reduces the rotation search space from 3-DOF to 1-DOF.

    The estimation proceeds as follows:
    1. Extract source and target points at the specified correspondence indices.
    2. Subtract centroids from each set.
    3. Project the centered points onto the plane perpendicular to the rotation axis.
    4. Estimate the rotation angle theta via closed-form atan2 minimization.
    5. Build the 3D rotation matrix R(axis, theta) using Rodrigues' formula.
    6. Recover the translation t = mean_target - R @ mean_source.

    Args:
        source: Source point cloud.
        target: Target point cloud.
        correspondences: Pairs of (source_index, target_index) indices. At least 2
            correspondences are required. Correspondences where both points lie on
            the rotation axis do not constrain the angle.
        rotation_axis: A 3D vector defining the rotation axis. Need not be a unit
            vector; it is normalized internally.

    Returns:
        A 4x4 homogeneous transformation matrix T such that approximately
        target[j] = T @ source[i] for each correspondence (i, j).

    Raises:
        ValueError: If rotation_axis is the zero vector or not 3D, or if fewer
            than 2 correspondences are provided.
    """
    unit_axis = _normalize_axis(rotation_axis)
    source_pts, target_pts = _extract_correspondence_points(
        source, target, correspondences
    )

    source_mean = source_pts.mean(axis=0)
    target_mean = target_pts.mean(axis=0)
    source_centered = source_pts - source_mean
    target_centered = target_pts - target_mean

    source_perp = _project_perpendicular_to_axis(source_centered, unit_axis)
    target_perp = _project_perpendicular_to_axis(target_centered, unit_axis)

    theta = _estimate_rotation_angle_around_axis(source_perp, target_perp, unit_axis)
    R = rotation_matrix_from_axis_angle(unit_axis, theta)
    t = target_mean - R @ source_mean

    return rototranslation_from_rotation_translation(R, t)


class AxisConstrainedTransformationEstimation:
    """Estimate rigid transformations with the rotation constrained to a fixed axis.

    This class mirrors the interface of Open3D's TransformationEstimation and is
    intended for use inside a custom RANSAC loop where the rotation is known to be
    a rotation around a specified axis (e.g., the vertical/gravity direction).

    Args:
        rotation_axis: A 3D vector defining the rotation axis. Need not be a unit
            vector; it is normalized at construction time.

    Raises:
        ValueError: If rotation_axis is the zero vector or not 3D.
    """

    def __init__(self, rotation_axis: np.ndarray) -> None:
        """Initialize the estimator and normalize the rotation axis.

        Args:
            rotation_axis: A 3D vector defining the rotation axis. Need not be a
                unit vector; it is normalized at construction time.

        Raises:
            ValueError: If rotation_axis is the zero vector or not 3D.
        """
        self._unit_axis = _normalize_axis(rotation_axis)

    @property
    def rotation_axis(self) -> np.ndarray:
        """The unit rotation axis used for estimation."""
        return self._unit_axis

    def compute_transformation(
        self,
        source: o3d.geometry.PointCloud,
        target: o3d.geometry.PointCloud,
        correspondences: o3d.utility.Vector2iVector,
    ) -> np.ndarray:
        """Compute the constrained rototranslation from the given correspondences.

        Args:
            source: Source point cloud.
            target: Target point cloud.
            correspondences: Pairs of (source_index, target_index) indices.
                At least 2 correspondences are required.

        Returns:
            A 4x4 homogeneous transformation matrix.

        Raises:
            ValueError: If fewer than 2 correspondences are provided.
        """
        return estimate_constrained_rototranslation(
            source, target, correspondences, self._unit_axis
        )
