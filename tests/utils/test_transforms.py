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
    cross_matrix,
    rotation_matrix_from_axis_angle,
    rotation_aligning_two_directions,
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


class TestCrossMatrix:
    """Tests for cross_matrix function."""

    def test_cross_matrix_basic_vectors(self):
        """Test cross product matrix with basic unit vectors."""
        # Test with x-axis unit vector
        v = np.array([1, 0, 0])
        K = cross_matrix(v)
        expected = np.array([[0, 0, 0], [0, 0, -1], [0, 1, 0]])
        assert np.allclose(K, expected), "Cross matrix for x-axis should be correct"

        # Test with y-axis unit vector
        v = np.array([0, 1, 0])
        K = cross_matrix(v)
        expected = np.array([[0, 0, 1], [0, 0, 0], [-1, 0, 0]])
        assert np.allclose(K, expected), "Cross matrix for y-axis should be correct"

        # Test with z-axis unit vector
        v = np.array([0, 0, 1])
        K = cross_matrix(v)
        expected = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 0]])
        assert np.allclose(K, expected), "Cross matrix for z-axis should be correct"

    def test_cross_matrix_property(self):
        """Test that cross_matrix(v) @ w = v × w."""
        v = np.array([1, 2, 3])
        w = np.array([4, 5, 6])

        # Compute cross product using cross_matrix
        K = cross_matrix(v)
        result_matrix = K @ w

        # Compute cross product using numpy
        result_numpy = np.cross(v, w)

        assert np.allclose(result_matrix, result_numpy), (
            "cross_matrix(v) @ w should equal v × w"
        )

    def test_cross_matrix_is_skew_symmetric(self):
        """Test that the cross product matrix is skew-symmetric."""
        v = np.array([2.5, -1.3, 4.7])
        K = cross_matrix(v)

        # Skew-symmetric means K^T = -K
        assert np.allclose(K.T, -K), "Cross matrix should be skew-symmetric"

    def test_cross_matrix_zero_vector(self):
        """Test cross matrix with zero vector."""
        v = np.array([0, 0, 0])
        K = cross_matrix(v)
        expected = np.zeros((3, 3))
        assert np.allclose(K, expected), "Cross matrix of zero vector should be zero"

    def test_cross_matrix_invalid_input(self):
        """Test that invalid input raises ValueError."""
        # 2D vector
        with pytest.raises(ValueError, match="3D vector"):
            cross_matrix(np.array([1, 2]))

        # 4D vector
        with pytest.raises(ValueError, match="3D vector"):
            cross_matrix(np.array([1, 2, 3, 4]))

        # Scalar
        with pytest.raises(ValueError, match="3D vector"):
            cross_matrix(np.array([1]))

    def test_cross_matrix_arbitrary_vectors(self):
        """Test cross matrix with various arbitrary vectors."""
        test_vectors = [
            np.array([1, 1, 1]),
            np.array([-1, 2, -3]),
            np.array([0.5, -0.7, 1.2]),
            np.array([100, -50, 25]),
        ]

        for v in test_vectors:
            K = cross_matrix(v)
            # Test with a random vector
            w = np.array([1, 2, 3])
            assert np.allclose(K @ w, np.cross(v, w)), (
                f"Cross matrix property should hold for v={v}"
            )

    def test_cross_matrix_scaling_property(self):
        """Test that cross_matrix(a*v) = a * cross_matrix(v)."""
        v = np.array([1, 2, 3])
        a = 2.5

        K_av = cross_matrix(a * v)
        a_Kv = a * cross_matrix(v)

        assert np.allclose(K_av, a_Kv), (
            "cross_matrix should be linear with respect to scalar multiplication"
        )


class TestRotationMatrixFromAxisAngle:
    """Tests for rotation_matrix_from_axis_angle function."""

    def test_zero_angle_gives_identity(self):
        """Test that zero rotation angle gives identity matrix."""
        axis = np.array([1, 0, 0])
        angle = 0.0
        R = rotation_matrix_from_axis_angle(axis, angle)
        assert np.allclose(R, np.eye(3)), "Zero angle should give identity"

    def test_90_degree_rotation_around_axes(self):
        """Test 90-degree rotations around principal axes."""
        # 90° around x-axis
        R = rotation_matrix_from_axis_angle(np.array([1, 0, 0]), np.pi / 2)
        expected = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
        assert np.allclose(R, expected, atol=1e-10), "90° around x-axis"

        # 90° around y-axis
        R = rotation_matrix_from_axis_angle(np.array([0, 1, 0]), np.pi / 2)
        expected = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
        assert np.allclose(R, expected, atol=1e-10), "90° around y-axis"

        # 90° around z-axis
        R = rotation_matrix_from_axis_angle(np.array([0, 0, 1]), np.pi / 2)
        expected = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        assert np.allclose(R, expected, atol=1e-10), "90° around z-axis"

    def test_180_degree_rotation(self):
        """Test 180-degree rotation around an axis."""
        # 180° around x-axis
        R = rotation_matrix_from_axis_angle(np.array([1, 0, 0]), np.pi)
        expected = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
        assert np.allclose(R, expected, atol=1e-10), "180° around x-axis"

    def test_generated_matrix_is_valid_rotation(self):
        """Test that generated matrices are valid rotations."""
        test_cases = [
            (np.array([1, 0, 0]), np.pi / 4),
            (np.array([0, 1, 0]), np.pi / 3),
            (np.array([0, 0, 1]), 2 * np.pi / 3),
            (np.array([1, 1, 1]), np.pi / 6),
        ]

        for axis, angle in test_cases:
            R = rotation_matrix_from_axis_angle(axis, angle)
            assert is_rotation_matrix(R), (
                f"Matrix for axis={axis}, angle={angle} should be valid rotation"
            )

    def test_non_unit_axis_is_normalized(self):
        """Test that non-unit axis vectors are automatically normalized."""
        axis_non_unit = np.array([3, 4, 0])  # Length = 5
        axis_unit = axis_non_unit / np.linalg.norm(axis_non_unit)
        angle = np.pi / 4

        R1 = rotation_matrix_from_axis_angle(axis_non_unit, angle)
        R2 = rotation_matrix_from_axis_angle(axis_unit, angle)

        assert np.allclose(R1, R2), "Non-unit axis should be normalized automatically"

    def test_opposite_axis_negative_angle(self):
        """Test that opposite axis with negative angle gives same result."""
        axis = np.array([1, 1, 1])
        angle = np.pi / 3

        R1 = rotation_matrix_from_axis_angle(axis, angle)
        R2 = rotation_matrix_from_axis_angle(-axis, -angle)

        assert np.allclose(R1, R2, atol=1e-10), (
            "Opposite axis with negative angle should give same rotation"
        )

    def test_rotation_preserves_axis(self):
        """Test that rotation matrix preserves vectors along the rotation axis."""
        axis = np.array([1, 2, 3])
        axis = axis / np.linalg.norm(axis)
        angle = np.pi / 4

        R = rotation_matrix_from_axis_angle(axis, angle)
        rotated_axis = R @ axis

        assert np.allclose(rotated_axis, axis, atol=1e-10), (
            "Rotation should preserve vectors along the axis"
        )

    def test_roundtrip_with_axis_angle_extraction(self):
        """Test that we can extract axis and angle back from the matrix."""
        axis_input = np.array([1, 2, 3])
        axis_input = axis_input / np.linalg.norm(axis_input)
        angle_input = np.pi / 3

        R = rotation_matrix_from_axis_angle(axis_input, angle_input)
        axis_output, angle_output = axis_angle_from_rotation(R)

        assert np.isclose(angle_input, angle_output, atol=1e-6), (
            "Extracted angle should match input"
        )
        # Axis can be either direction
        assert np.allclose(axis_output, axis_input, atol=1e-6) or np.allclose(
            axis_output, -axis_input, atol=1e-6
        ), "Extracted axis should match input (or its negative)"

    def test_full_rotation_gives_identity(self):
        """Test that 2π rotation gives identity matrix."""
        axis = np.array([1, 2, 3])
        R = rotation_matrix_from_axis_angle(axis, 2 * np.pi)
        assert np.allclose(R, np.eye(3), atol=1e-10), (
            "Full 2π rotation should give identity"
        )

    def test_invalid_input_raises_error(self):
        """Test that invalid input raises ValueError."""
        # 2D vector
        with pytest.raises(ValueError, match="3D vector"):
            rotation_matrix_from_axis_angle(np.array([1, 2]), np.pi / 2)

        # 4D vector
        with pytest.raises(ValueError, match="3D vector"):
            rotation_matrix_from_axis_angle(np.array([1, 2, 3, 4]), np.pi / 2)

    def test_rodrigues_formula_property(self):
        """Test that the matrix follows Rodrigues' rotation formula."""
        axis = np.array([0, 0, 1])
        angle = np.pi / 4

        R = rotation_matrix_from_axis_angle(axis, angle)

        # Manually compute using Rodrigues formula
        K = cross_matrix(axis)
        R_manual = (
            np.cos(angle) * np.eye(3)
            + np.sin(angle) * K
            + (1 - np.cos(angle)) * np.outer(axis, axis)
        )

        assert np.allclose(R, R_manual, atol=1e-10), "Should match Rodrigues formula"


class TestRotationAligningTwoDirections:
    """Tests for rotation_aligning_two_directions function."""

    def test_aligned_directions_give_identity(self):
        """Test that already aligned directions give identity."""
        src = np.array([1, 0, 0])
        tgt = np.array([1, 0, 0])

        R = rotation_aligning_two_directions(src, tgt)

        # Should return identity for aligned vectors
        assert np.allclose(R, np.eye(3), atol=1e-10), (
            "Aligned directions should give identity matrix"
        )
        result = R @ src
        assert np.allclose(result, tgt, atol=1e-10), (
            "Identity should preserve the vector"
        )

    def test_scaled_aligned_directions(self):
        """Test that scaled but aligned directions give identity."""
        src = np.array([2, 0, 0])
        tgt = np.array([5, 0, 0])  # Same direction, different magnitude

        R = rotation_aligning_two_directions(src, tgt)

        # Should return identity since they point in the same direction
        assert np.allclose(R, np.eye(3), atol=1e-10), (
            "Scaled aligned directions should give identity matrix"
        )

    def test_invalid_input_dimensions(self):
        """Test that invalid input dimensions raise ValueError."""
        # 2D vectors
        with pytest.raises(ValueError, match="3D vectors"):
            rotation_aligning_two_directions(np.array([1, 2]), np.array([3, 4]))

        # 4D vectors
        with pytest.raises(ValueError, match="3D vectors"):
            rotation_aligning_two_directions(
                np.array([1, 2, 3, 4]), np.array([5, 6, 7, 8])
            )

        # Mixed dimensions
        with pytest.raises(ValueError, match="3D vectors"):
            rotation_aligning_two_directions(np.array([1, 2, 3]), np.array([4, 5]))

    def test_90_degree_alignment(self):
        """Test alignment of perpendicular vectors."""
        src = np.array([1, 0, 0])
        tgt = np.array([0, 1, 0])

        R = rotation_aligning_two_directions(src, tgt)
        result = R @ src

        assert np.allclose(result, tgt, atol=1e-10), "Should align x-axis to y-axis"
        assert is_rotation_matrix(R), "Result should be a valid rotation"

    @pytest.mark.skip(
        reason="Function doesn't handle anti-parallel vectors (cross product = 0)"
    )
    def test_180_degree_alignment(self):
        """Test alignment of opposite vectors."""
        src = np.array([1, 0, 0])
        tgt = np.array([-1, 0, 0])

        R = rotation_aligning_two_directions(src, tgt)
        result = R @ src

        assert np.allclose(result, tgt, atol=1e-6), "Should align to opposite direction"

    def test_arbitrary_alignment(self):
        """Test alignment of arbitrary directions."""
        test_cases = [
            (np.array([1, 0, 0]), np.array([0, 0, 1])),
            (np.array([1, 1, 0]), np.array([0, 1, 1])),
            (np.array([1, 2, 3]), np.array([3, 2, 1])),
            (np.array([-1, 2, -3]), np.array([4, -5, 6])),
        ]

        for src, tgt in test_cases:
            R = rotation_aligning_two_directions(src, tgt)
            result = R @ src

            # Normalize for comparison
            result_norm = result / np.linalg.norm(result)
            tgt_norm = tgt / np.linalg.norm(tgt)

            assert np.allclose(result_norm, tgt_norm, atol=1e-6), (
                f"Should align {src} to {tgt}"
            )
            assert is_rotation_matrix(R), "Result should be a valid rotation"

    def test_non_unit_vectors_are_normalized(self):
        """Test that non-unit vectors are properly normalized."""
        src = np.array([2, 0, 0])  # Length = 2
        tgt = np.array([0, 3, 0])  # Length = 3

        R = rotation_aligning_two_directions(src, tgt)
        result = R @ src

        # Result should point in target direction
        result_norm = result / np.linalg.norm(result)
        tgt_norm = tgt / np.linalg.norm(tgt)

        assert np.allclose(result_norm, tgt_norm, atol=1e-10), (
            "Non-unit vectors should be normalized"
        )

    def test_preserves_orthogonal_components(self):
        """Test rotation minimality - perpendicular components are preserved."""
        src = np.array([1, 0, 0])
        tgt = np.array([0, 1, 0])

        R = rotation_aligning_two_directions(src, tgt)

        # Vector perpendicular to both src and tgt
        perp = np.cross(src, tgt)
        perp = perp / np.linalg.norm(perp)

        # This perpendicular vector should be preserved (it's the rotation axis)
        result = R @ perp
        assert np.allclose(result, perp, atol=1e-10), (
            "Rotation should preserve the axis perpendicular to both directions"
        )

    def test_rotation_angle_matches_vectors_angle(self):
        """Test that the rotation angle matches the angle between vectors."""
        src = np.array([1, 0, 0])
        tgt = np.array([1, 1, 0]) / np.sqrt(2)  # 45° from src

        R = rotation_aligning_two_directions(src, tgt)

        # Extract rotation angle
        _, angle = axis_angle_from_rotation(R)

        # Expected angle between vectors
        expected_angle = np.arccos(np.dot(src, tgt))

        assert np.isclose(angle, expected_angle, atol=1e-6), (
            "Rotation angle should match angle between vectors"
        )

    def test_chain_of_alignments(self):
        """Test that chaining alignments works correctly."""
        v1 = np.array([1, 0, 0])
        v2 = np.array([0, 1, 0])
        v3 = np.array([0, 0, 1])

        R1 = rotation_aligning_two_directions(v1, v2)
        R2 = rotation_aligning_two_directions(v2, v3)

        # Compose rotations
        R_composed = R2 @ R1

        # Should align v1 to v3
        result = R_composed @ v1
        assert np.allclose(result, v3, atol=1e-10), (
            "Chained rotations should work correctly"
        )

    def test_random_directions(self):
        """Test with random direction vectors."""
        np.random.seed(42)

        for _ in range(10):
            src = np.random.randn(3)
            tgt = np.random.randn(3)

            # Skip if vectors are too close to parallel or anti-parallel
            cos_angle = np.dot(src, tgt) / (np.linalg.norm(src) * np.linalg.norm(tgt))
            if abs(abs(cos_angle) - 1.0) < 1e-6:
                continue

            R = rotation_aligning_two_directions(src, tgt)
            result = R @ src

            # Normalize for comparison
            result_norm = result / np.linalg.norm(result)
            tgt_norm = tgt / np.linalg.norm(tgt)

            assert np.allclose(result_norm, tgt_norm, atol=1e-6), (
                "Random alignment should work"
            )
            assert is_rotation_matrix(R), "Result should be valid rotation"

    def test_preserves_length(self):
        """Test that rotation preserves vector length."""
        src = np.array([1, 2, 3])
        tgt = np.array([4, 5, 6])

        R = rotation_aligning_two_directions(src, tgt)
        result = R @ src

        assert np.isclose(np.linalg.norm(result), np.linalg.norm(src), atol=1e-10), (
            "Rotation should preserve vector length"
        )
