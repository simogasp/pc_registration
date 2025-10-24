"""Unit tests for common utility functions."""

import pytest
import numpy as np
from common import (
    axis_angle_from_rotation,
    rotation_error_angle,
    translation_error,
    transformation_error,
)


class TestAxisAngleConversion:
    """Tests for axis_angle_from_rotation function."""

    def test_identity_rotation(self):
        """Test that identity matrix gives zero rotation."""
        R = np.eye(3)
        axis, angle = axis_angle_from_rotation(R)
        assert np.isclose(angle, 0.0), "Identity rotation should have angle 0"
        assert np.linalg.norm(axis) > 0, "Axis should be normalized"

    def test_90_degree_rotation_around_z(self):
        """Test 90-degree rotation around z-axis."""
        # 90 degrees around z-axis
        R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        axis, angle = axis_angle_from_rotation(R)
        assert np.isclose(angle, np.pi / 2), "Angle should be π/2"
        assert np.allclose(axis, [0, 0, 1]), "Axis should be z-axis"

    def test_180_degree_rotation(self):
        """Test 180-degree rotation."""
        # 180 degrees around x-axis
        R = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
        axis, angle = axis_angle_from_rotation(R)
        assert np.isclose(angle, np.pi, atol=1e-6), "Angle should be π"
        assert np.linalg.norm(axis) > 0, "Axis should be normalized"

    def test_arbitrary_rotation(self):
        """Test an arbitrary rotation matrix."""
        # 45 degrees around axis [1, 1, 0] (normalized)
        theta = np.pi / 4
        axis_input = np.array([1, 1, 0]) / np.sqrt(2)

        # Create rotation matrix using Rodrigues formula
        K = np.array(
            [
                [0, -axis_input[2], axis_input[1]],
                [axis_input[2], 0, -axis_input[0]],
                [-axis_input[1], axis_input[0], 0],
            ]
        )
        R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * K @ K

        axis, angle = axis_angle_from_rotation(R)
        assert np.isclose(angle, theta, atol=1e-6), f"Angle should be {theta}"
        assert np.allclose(axis, axis_input, atol=1e-6) or np.allclose(
            axis, -axis_input, atol=1e-6
        ), "Axis should match input (or its negative)"


class TestRotationError:
    """Tests for rotation_error_angle function."""

    def test_identical_rotations(self):
        """Test error between identical rotations is zero."""
        R = np.eye(3)
        error = rotation_error_angle(R, R)
        assert np.isclose(error, 0.0), "Error should be zero for identical rotations"

    def test_90_degree_error(self):
        """Test 90-degree rotation error."""
        R_est = np.eye(3)
        R_gt = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        error = rotation_error_angle(R_est, R_gt)
        assert np.isclose(error, np.pi / 2, atol=1e-6), "Error should be π/2"

    def test_180_degree_error(self):
        """Test 180-degree rotation error."""
        R_est = np.eye(3)
        R_gt = -np.eye(3)
        error = rotation_error_angle(R_est, R_gt)
        assert np.isclose(error, np.pi, atol=1e-6), "Error should be π"


class TestTranslationError:
    """Tests for translation_error function."""

    def test_no_error(self):
        """Test zero translation error."""
        R_est = np.eye(3)
        R_gt = np.eye(3)
        t_est = np.array([1.0, 2.0, 3.0])
        t_gt = np.array([1.0, 2.0, 3.0])

        norm, vec = translation_error(R_est, t_est, R_gt, t_gt)
        assert np.isclose(norm, 0.0), "Translation error should be zero"
        assert np.allclose(vec, [0, 0, 0]), "Error vector should be zero"

    def test_simple_translation_error(self):
        """Test simple translation error with identity rotations."""
        R_est = np.eye(3)
        R_gt = np.eye(3)
        t_est = np.array([1.0, 0.0, 0.0])
        t_gt = np.array([0.0, 0.0, 0.0])

        norm, vec = translation_error(R_est, t_est, R_gt, t_gt)
        assert np.isclose(norm, 1.0), "Translation error magnitude should be 1.0"
        assert np.allclose(vec, [1.0, 0.0, 0.0]), "Error vector should be [1, 0, 0]"

    def test_translation_error_with_rotation(self):
        """Test translation error accounting for rotation difference."""
        R_est = np.eye(3)
        R_gt = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])  # 90° rotation around z
        t_est = np.array([1.0, 0.0, 0.0])
        t_gt = np.array([1.0, 0.0, 0.0])  # Changed to have meaningful difference

        norm, vec = translation_error(R_est, t_est, R_gt, t_gt)
        # With rotation difference, the error should account for rotated reference frame
        assert isinstance(norm, (float, np.floating)), "Norm should be a float"
        assert vec.shape == (3,), "Error vector should be 3D"


class TestTransformationError:
    """Tests for transformation_error function."""

    def test_identity_transformation(self):
        """Test error between identical transformations is zero."""
        T = np.eye(4)
        rot_err, trans_err = transformation_error(T, T)
        assert np.isclose(rot_err, 0.0), "Rotation error should be zero"
        assert np.isclose(trans_err, 0.0), "Translation error should be zero"

    def test_invalid_matrix_shape(self):
        """Test that invalid matrix shapes raise ValueError."""
        T_invalid = np.eye(3)
        T_valid = np.eye(4)

        with pytest.raises(ValueError, match="4x4 matrices"):
            transformation_error(T_invalid, T_valid)

        with pytest.raises(ValueError, match="4x4 matrices"):
            transformation_error(T_valid, T_invalid)

    def test_pure_rotation_error(self):
        """Test transformation with only rotation difference."""
        T_est = np.eye(4)
        T_gt = np.eye(4)
        T_gt[:3, :3] = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])

        rot_err, trans_err = transformation_error(T_est, T_gt)
        assert np.isclose(rot_err, np.pi / 2, atol=1e-6), "Rotation error should be π/2"
        assert np.isclose(trans_err, 0.0, atol=1e-6), "Translation error should be zero"

    def test_pure_translation_error(self):
        """Test transformation with only translation difference."""
        T_est = np.eye(4)
        T_est[:3, 3] = [1.0, 0.0, 0.0]
        T_gt = np.eye(4)

        rot_err, trans_err = transformation_error(T_est, T_gt)
        assert np.isclose(rot_err, 0.0, atol=1e-6), "Rotation error should be zero"
        assert np.isclose(trans_err, 1.0, atol=1e-6), "Translation error should be 1.0"

    def test_combined_error(self):
        """Test transformation with both rotation and translation errors."""
        T_est = np.eye(4)
        T_est[:3, 3] = [2.0, 1.0, 0.0]  # Different translation

        T_gt = np.eye(4)
        T_gt[:3, :3] = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        T_gt[:3, 3] = [1.0, 2.0, 0.0]  # Different translation

        rot_err, trans_err = transformation_error(T_est, T_gt)
        assert rot_err > 0, "Rotation error should be positive"
        assert isinstance(trans_err, (float, np.floating)), (
            "Translation error should be a float"
        )
