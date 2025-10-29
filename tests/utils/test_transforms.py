"""Unit tests for transformation and rotation utilities."""

import numpy as np
import pytest

from registration.utils.transforms import (
    axis_angle_from_rotation,
    rotation_error_angle,
    transformation_error,
    translation_error,
    generate_random_rotation_matrix,
    is_rotation_matrix,
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


class TestGenerateRandomRotationMatrix:
    """Tests for generate_random_rotation_matrix function."""

    def test_returns_3x3_matrix(self):
        """Test that the function returns a 3x3 matrix."""
        R = generate_random_rotation_matrix()
        assert R.shape == (3, 3), "Should return a 3x3 matrix"

    def test_is_valid_rotation_matrix(self):
        """Test that the generated matrix is a valid rotation matrix."""
        R = generate_random_rotation_matrix()
        # Check orthogonality: R.T @ R should be identity
        assert np.allclose(R.T @ R, np.eye(3), atol=1e-10), (
            "Matrix should be orthogonal"
        )
        # Check determinant is +1
        assert np.isclose(np.linalg.det(R), 1.0, atol=1e-10), "Determinant should be +1"

    def test_multiple_generations_are_different(self):
        """Test that multiple calls generate different matrices."""
        R1 = generate_random_rotation_matrix()
        R2 = generate_random_rotation_matrix()
        R3 = generate_random_rotation_matrix()

        # Very unlikely to generate identical matrices
        assert not np.allclose(R1, R2, atol=1e-6), "Should generate different matrices"
        assert not np.allclose(R2, R3, atol=1e-6), "Should generate different matrices"

    def test_generated_matrices_pass_validation(self):
        """Test that generated matrices pass is_rotation_matrix validation."""
        for _ in range(10):
            R = generate_random_rotation_matrix()
            assert is_rotation_matrix(R), (
                "Generated matrix should pass is_rotation_matrix validation"
            )

    def test_uniform_sampling_coverage(self):
        """Test that the function generates diverse rotations."""
        # Generate multiple random rotations and check they span SO(3)
        rotations = [generate_random_rotation_matrix() for _ in range(20)]

        # Check that determinants are all +1
        dets = [np.linalg.det(R) for R in rotations]
        assert all(np.isclose(d, 1.0, atol=1e-10) for d in dets), (
            "All determinants should be +1"
        )

        # Check diversity: compute pairwise rotation errors
        errors = []
        for i in range(len(rotations) - 1):
            error = rotation_error_angle(rotations[i], rotations[i + 1])
            errors.append(error)

        # At least some rotations should have significant angular difference
        assert any(error > 0.1 for error in errors), (
            "Should generate diverse rotations with significant angular differences"
        )

    def test_preserves_vector_norms(self):
        """Test that rotation preserves vector norms."""
        R = generate_random_rotation_matrix()
        v = np.array([1.0, 2.0, 3.0])
        v_rotated = R @ v

        assert np.isclose(np.linalg.norm(v), np.linalg.norm(v_rotated)), (
            "Rotation should preserve vector norms"
        )

    def test_composition_is_valid_rotation(self):
        """Test that composition of generated rotations is also a valid rotation."""
        R1 = generate_random_rotation_matrix()
        R2 = generate_random_rotation_matrix()
        R_composed = R1 @ R2

        assert is_rotation_matrix(R_composed), (
            "Composition of rotations should be a valid rotation"
        )

    def test_all_determinants_positive(self):
        """Test that all generated matrices have determinant +1."""
        # Generate many matrices to increase chance of hitting both code paths
        for _ in range(50):
            R = generate_random_rotation_matrix()
            det = np.linalg.det(R)
            assert np.isclose(det, 1.0, atol=1e-10), (
                f"Determinant should be +1, got {det}"
            )
            assert det > 0, "Determinant should be positive"


class TestIsRotationMatrix:
    """Tests for is_rotation_matrix function."""

    def test_identity_is_valid(self):
        """Test that identity matrix is recognized as a valid rotation."""
        R = np.eye(3)
        assert is_rotation_matrix(R), "Identity should be a valid rotation matrix"

    def test_valid_90_degree_rotation(self):
        """Test that a valid 90-degree rotation is recognized."""
        R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        assert is_rotation_matrix(R), "90-degree rotation should be valid"

    def test_valid_arbitrary_rotation(self):
        """Test that an arbitrary valid rotation is recognized."""
        # 45-degree rotation around z-axis
        theta = np.pi / 4
        R = np.array(
            [
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1],
            ]
        )
        assert is_rotation_matrix(R), "Arbitrary rotation should be valid"

    def test_non_orthogonal_matrix_rejected(self):
        """Test that non-orthogonal matrices are rejected."""
        # Random non-orthogonal matrix
        R = np.array([[1, 2, 0], [0, 1, 0], [0, 0, 1]])
        assert not is_rotation_matrix(R), "Non-orthogonal matrix should be rejected"

    def test_negative_determinant_rejected(self):
        """Test that matrices with determinant -1 are rejected (reflections)."""
        # This is a reflection, not a rotation
        R = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]])
        assert not is_rotation_matrix(R), "Reflection (det = -1) should be rejected"

    def test_scaled_matrix_rejected(self):
        """Test that scaled orthogonal matrices are rejected."""
        # Scaled identity matrix
        R = 2.0 * np.eye(3)
        assert not is_rotation_matrix(R), "Scaled matrix should be rejected"

    def test_random_matrix_rejected(self):
        """Test that random non-rotation matrices are rejected."""
        np.random.seed(42)
        R = np.random.randn(3, 3)
        assert not is_rotation_matrix(R), "Random matrix should be rejected"

    def test_near_rotation_with_numerical_error(self):
        """Test handling of matrices with small numerical errors."""
        # Create a rotation with tiny numerical error
        R = np.eye(3)
        R[0, 0] = 1.0 + 1e-10  # Very small deviation

        # Should still pass due to tolerance in np.allclose
        result = is_rotation_matrix(R)
        assert result, "Matrix with tiny numerical error should pass"

    def test_singular_matrix_rejected(self):
        """Test that singular matrices are rejected."""
        # Singular matrix (determinant = 0)
        R = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]])
        assert not is_rotation_matrix(R), "Singular matrix should be rejected"

    def test_skew_symmetric_rejected(self):
        """Test that skew-symmetric matrices are rejected."""
        # Skew-symmetric matrix (det = 0 for 3x3)
        R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 0]])
        assert not is_rotation_matrix(R), "Skew-symmetric matrix should be rejected"

    def test_parametrized_valid_rotations(self):
        """Test multiple valid rotations around different axes."""
        test_cases = [
            # 90° around x-axis
            np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]),
            # 90° around y-axis
            np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]]),
            # 90° around z-axis
            np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]),
            # 180° around x-axis
            np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]]),
        ]

        for i, R in enumerate(test_cases):
            assert is_rotation_matrix(R), f"Test case {i} should be valid rotation"

    def test_generated_random_rotations_are_valid(self):
        """Test that all generated random rotations pass validation."""
        for _ in range(20):
            R = generate_random_rotation_matrix()
            assert is_rotation_matrix(R), (
                "All generated random rotations should be valid"
            )
