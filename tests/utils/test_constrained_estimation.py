"""Unit tests for axis-constrained rigid transformation estimation."""

from typing import TypedDict

import numpy as np
import open3d as o3d
import pytest

from registration.utils.constrained_estimation import (
    AxisConstrainedTransformationEstimation,
    _estimate_rotation_angle_around_axis,
    _normalize_axis,
    _project_perpendicular_to_axis,
    estimate_constrained_rototranslation,
)
from registration.utils.transforms import (
    is_rotation_matrix,
    rotation_matrix_from_axis_angle,
    rototranslation_from_rotation_translation,
)

# ---------------------------------------------------------------------------
# Test-case TypedDicts
# ---------------------------------------------------------------------------


class _AngleCase(TypedDict):
    angle: float
    description: str


class _TransformCase(TypedDict):
    description: str
    axis: np.ndarray
    angle: float
    translation: np.ndarray


class _CorresCase(TypedDict):
    n_corres: int
    description: str


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

# Source points used across estimation tests. Seven non-collinear points that
# span all three dimensions and are not aligned with any cardinal axis, so they
# exercise the perpendicular projection and angle estimation steps fully.
SOURCE_POINTS = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 1.0],
        [1.0, 0.0, 1.0],
        [-1.0, 0.5, 0.3],
    ],
    dtype=np.float64,
)


def _make_point_cloud(points: np.ndarray) -> o3d.geometry.PointCloud:
    """Create an Open3D point cloud from an (N, 3) numpy array."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd


def _make_correspondences(n: int) -> o3d.utility.Vector2iVector:
    """Create identity correspondences (i, i) for i in range(n)."""
    return o3d.utility.Vector2iVector([[i, i] for i in range(n)])


def _apply_transform(points: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Apply a 4x4 homogeneous transformation to an (N, 3) point array."""
    homogeneous = np.hstack([points, np.ones((len(points), 1))])
    return (T @ homogeneous.T).T[:, :3]


def _build_ground_truth_transform(
    axis: np.ndarray, angle: float, translation: np.ndarray
) -> np.ndarray:
    """Build a 4x4 ground-truth transformation from axis, angle, and translation."""
    unit_axis = axis / np.linalg.norm(axis)
    R = rotation_matrix_from_axis_angle(unit_axis, angle)
    return rototranslation_from_rotation_translation(R, translation)


# ---------------------------------------------------------------------------
# TestNormalizeAxis
# ---------------------------------------------------------------------------


class TestNormalizeAxis:
    """Tests for the _normalize_axis helper."""

    def test_already_unit_vector_unchanged(self):
        """A unit vector should pass through unchanged."""
        axis = np.array([0.0, 0.0, 1.0])

        result = _normalize_axis(axis)

        assert np.allclose(result, axis)

    def test_non_unit_vector_normalized(self):
        """A non-unit vector should be scaled to unit length."""
        axis = np.array([3.0, 0.0, 0.0])

        result = _normalize_axis(axis)

        assert np.allclose(result, np.array([1.0, 0.0, 0.0]))
        assert np.isclose(np.linalg.norm(result), 1.0)

    def test_arbitrary_vector_normalized(self):
        """An arbitrary non-unit vector should produce a unit vector."""
        axis = np.array([1.0, 2.0, 3.0])

        result = _normalize_axis(axis)

        assert np.isclose(np.linalg.norm(result), 1.0)
        assert np.allclose(result, axis / np.linalg.norm(axis))

    def test_zero_vector_raises_value_error(self):
        """A zero vector should raise ValueError."""
        axis = np.zeros(3)

        with pytest.raises(ValueError, match="non-zero"):
            _normalize_axis(axis)

    def test_wrong_shape_raises_value_error(self):
        """A non-3D vector should raise ValueError."""
        test_cases = [
            np.array([1.0, 0.0]),
            np.array([1.0, 0.0, 0.0, 0.0]),
            np.array([[1.0, 0.0, 0.0]]),
        ]
        for axis in test_cases:
            with pytest.raises(ValueError, match="3D"):
                _normalize_axis(axis)


# ---------------------------------------------------------------------------
# TestProjectPerpendicularToAxis
# ---------------------------------------------------------------------------


class TestProjectPerpendicularToAxis:
    """Tests for the _project_perpendicular_to_axis helper."""

    def test_points_along_axis_project_to_zero(self):
        """Points that lie along the axis should project to zero."""
        unit_axis = np.array([0.0, 0.0, 1.0])
        points = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 3.0], [0.0, 0.0, -2.0]])

        result = _project_perpendicular_to_axis(points, unit_axis)

        assert np.allclose(result, np.zeros_like(points))

    def test_points_perpendicular_to_axis_unchanged(self):
        """Points already in the perpendicular plane should be returned unchanged."""
        unit_axis = np.array([0.0, 0.0, 1.0])
        points = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]])

        result = _project_perpendicular_to_axis(points, unit_axis)

        assert np.allclose(result, points)

    def test_mixed_points_correct_projection(self):
        """Mixed points should have their axial component removed."""
        unit_axis = np.array([0.0, 0.0, 1.0])
        # Point (1, 2, 5): axial part is (0, 0, 5), perp part should be (1, 2, 0)
        points = np.array([[1.0, 2.0, 5.0]])

        result = _project_perpendicular_to_axis(points, unit_axis)

        assert np.allclose(result, np.array([[1.0, 2.0, 0.0]]))

    def test_result_orthogonal_to_axis(self):
        """All projected points must be orthogonal to the rotation axis."""
        unit_axis = np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0)
        np.random.seed(0)
        points = np.random.randn(20, 3)

        result = _project_perpendicular_to_axis(points, unit_axis)

        dot_products = result @ unit_axis
        assert np.allclose(dot_products, 0.0, atol=1e-12)


# ---------------------------------------------------------------------------
# TestEstimateRotationAngleAroundAxis
# ---------------------------------------------------------------------------


class TestEstimateRotationAngleAroundAxis:
    """Tests for the _estimate_rotation_angle_around_axis helper."""

    def test_degenerate_all_points_on_axis(self):
        """When all projections are zero, should return 0.0."""
        unit_axis = np.array([0.0, 0.0, 1.0])
        zeros = np.zeros((4, 3))

        theta = _estimate_rotation_angle_around_axis(zeros, zeros, unit_axis)

        assert theta == 0.0

    def test_known_angle_around_z_axis(self):
        """Rotate source points by a known angle around z and recover it."""
        unit_axis = np.array([0.0, 0.0, 1.0])
        test_cases: list[_AngleCase] = [
            {"angle": np.pi / 4, "description": "45 degrees around z"},
            {"angle": np.pi / 2, "description": "90 degrees around z"},
            {"angle": -np.pi / 3, "description": "-60 degrees around z"},
        ]
        source = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.5, 0.0]])

        for case in test_cases:
            angle_gt = float(case["angle"])
            R = rotation_matrix_from_axis_angle(unit_axis, angle_gt)
            target = (R @ source.T).T

            theta = _estimate_rotation_angle_around_axis(source, target, unit_axis)

            assert np.isclose(theta, angle_gt, atol=1e-10), (
                f"Failed for {case['description']}: "
                f"expected {angle_gt:.4f}, got {theta:.4f}"
            )

    def test_known_angle_around_x_axis(self):
        """Rotate source points by a known angle around x and recover it."""
        unit_axis = np.array([1.0, 0.0, 0.0])
        angle_gt = np.pi / 3
        source = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.5]])
        R = rotation_matrix_from_axis_angle(unit_axis, angle_gt)
        target = (R @ source.T).T

        theta = _estimate_rotation_angle_around_axis(source, target, unit_axis)

        assert np.isclose(theta, angle_gt, atol=1e-10)

    def test_zero_angle_returns_zero(self):
        """When source and target are equal, the recovered angle should be zero."""
        unit_axis = np.array([0.0, 1.0, 0.0])
        source = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, -1.0]])

        theta = _estimate_rotation_angle_around_axis(source, source, unit_axis)

        assert np.isclose(theta, 0.0, atol=1e-10)


# ---------------------------------------------------------------------------
# TestEstimateConstrainedRototranslation
# ---------------------------------------------------------------------------


class TestEstimateConstrainedRototranslation:
    """Tests for estimate_constrained_rototranslation."""

    def test_output_is_4x4(self):
        """Output transformation must be a 4x4 matrix."""
        source_pcd = _make_point_cloud(SOURCE_POINTS)
        target_pcd = _make_point_cloud(SOURCE_POINTS)
        corres = _make_correspondences(len(SOURCE_POINTS))

        T = estimate_constrained_rototranslation(
            source_pcd, target_pcd, corres, np.array([0.0, 0.0, 1.0])
        )

        assert T.shape == (4, 4)

    def test_rotation_part_is_valid_rotation_matrix(self):
        """The rotation submatrix of the output must be in SO(3)."""
        source_pcd = _make_point_cloud(SOURCE_POINTS)
        T_gt = _build_ground_truth_transform(
            np.array([0.0, 0.0, 1.0]), np.pi / 4, np.array([1.0, 2.0, 3.0])
        )
        target_pcd = _make_point_cloud(_apply_transform(SOURCE_POINTS, T_gt))
        corres = _make_correspondences(len(SOURCE_POINTS))

        T = estimate_constrained_rototranslation(
            source_pcd, target_pcd, corres, np.array([0.0, 0.0, 1.0])
        )

        assert is_rotation_matrix(T[:3, :3]), (
            "Rotation part must be a valid SO(3) matrix"
        )

    def test_various_known_transformations(self):
        """Data-driven test: recover known axis-constrained transformations."""
        test_cases: list[_TransformCase] = [
            {
                "description": "identity: zero rotation, zero translation",
                "axis": np.array([0.0, 0.0, 1.0]),
                "angle": 0.0,
                "translation": np.zeros(3),
            },
            {
                "description": "90 degrees around z, no translation",
                "axis": np.array([0.0, 0.0, 1.0]),
                "angle": np.pi / 2,
                "translation": np.zeros(3),
            },
            {
                "description": "45 degrees around z with translation",
                "axis": np.array([0.0, 0.0, 1.0]),
                "angle": np.pi / 4,
                "translation": np.array([1.0, -2.0, 3.0]),
            },
            {
                "description": "90 degrees around x with translation",
                "axis": np.array([1.0, 0.0, 0.0]),
                "angle": np.pi / 2,
                "translation": np.array([0.5, 1.5, -0.5]),
            },
            {
                "description": "-60 degrees around y with translation",
                "axis": np.array([0.0, 1.0, 0.0]),
                "angle": -np.pi / 3,
                "translation": np.array([-1.0, 0.0, 2.5]),
            },
            {
                "description": "arbitrary axis with translation",
                "axis": np.array([1.0, 1.0, 1.0]),
                "angle": np.pi / 5,
                "translation": np.array([2.0, -1.0, 0.5]),
            },
            {
                "description": "non-unit axis: same result as normalized",
                "axis": np.array([0.0, 0.0, 5.0]),
                "angle": np.pi / 6,
                "translation": np.array([0.0, 1.0, -1.0]),
            },
            {
                "description": "pure translation: zero rotation angle",
                "axis": np.array([0.0, 0.0, 1.0]),
                "angle": 0.0,
                "translation": np.array([3.0, -2.0, 1.0]),
            },
        ]

        for case in test_cases:
            T_gt = _build_ground_truth_transform(
                case["axis"], case["angle"], case["translation"]
            )
            target_pts = _apply_transform(SOURCE_POINTS, T_gt)
            source_pcd = _make_point_cloud(SOURCE_POINTS)
            target_pcd = _make_point_cloud(target_pts)
            corres = _make_correspondences(len(SOURCE_POINTS))

            T_est = estimate_constrained_rototranslation(
                source_pcd, target_pcd, corres, case["axis"]
            )

            rot_err = np.linalg.norm(T_est[:3, :3] - T_gt[:3, :3])
            trans_err = np.linalg.norm(T_est[:3, 3] - T_gt[:3, 3])

            assert rot_err < 1e-8, (
                f"Rotation error too large for '{case['description']}': {rot_err}"
            )
            assert trans_err < 1e-8, (
                f"Translation error too large for '{case['description']}': {trans_err}"
            )

    def test_minimum_two_correspondences(self):
        """Two correspondences should be sufficient to estimate the transformation."""
        axis = np.array([0.0, 0.0, 1.0])
        angle = np.pi / 3
        translation = np.array([1.0, 0.5, -1.0])
        T_gt = _build_ground_truth_transform(axis, angle, translation)
        # Use only 2 points that are not collinear after centering
        source_pts = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
        target_pts = _apply_transform(source_pts, T_gt)
        source_pcd = _make_point_cloud(source_pts)
        target_pcd = _make_point_cloud(target_pts)
        corres = _make_correspondences(2)

        T_est = estimate_constrained_rototranslation(
            source_pcd, target_pcd, corres, axis
        )

        rot_err = np.linalg.norm(T_est[:3, :3] - T_gt[:3, :3])
        trans_err = np.linalg.norm(T_est[:3, 3] - T_gt[:3, 3])
        assert rot_err < 1e-8
        assert trans_err < 1e-8

    def test_fewer_than_two_correspondences_raises(self):
        """Fewer than 2 correspondences must raise ValueError."""
        source_pcd = _make_point_cloud(SOURCE_POINTS)
        target_pcd = _make_point_cloud(SOURCE_POINTS)
        axis = np.array([0.0, 0.0, 1.0])

        test_cases: list[_CorresCase] = [
            {"n_corres": 0, "description": "zero correspondences"},
            {"n_corres": 1, "description": "one correspondence"},
        ]
        for case in test_cases:
            corres = _make_correspondences(case["n_corres"])
            with pytest.raises(ValueError, match="2"):
                estimate_constrained_rototranslation(
                    source_pcd, target_pcd, corres, axis
                )

    def test_zero_axis_raises(self):
        """A zero rotation axis must raise ValueError."""
        source_pcd = _make_point_cloud(SOURCE_POINTS)
        target_pcd = _make_point_cloud(SOURCE_POINTS)
        corres = _make_correspondences(len(SOURCE_POINTS))

        with pytest.raises(ValueError, match="non-zero"):
            estimate_constrained_rototranslation(
                source_pcd, target_pcd, corres, np.zeros(3)
            )

    def test_non_3d_axis_raises(self):
        """A non-3D rotation axis must raise ValueError."""
        source_pcd = _make_point_cloud(SOURCE_POINTS)
        target_pcd = _make_point_cloud(SOURCE_POINTS)
        corres = _make_correspondences(len(SOURCE_POINTS))

        with pytest.raises(ValueError, match="3D"):
            estimate_constrained_rototranslation(
                source_pcd, target_pcd, corres, np.array([0.0, 0.0])
            )

    def test_degenerate_all_points_on_axis(self):
        """When all correspondence points lie on the rotation axis, return theta=0."""
        unit_axis = np.array([0.0, 0.0, 1.0])
        translation = np.array([1.0, 2.0, 0.0])
        # Points along the z axis only
        source_pts = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 2.0], [0.0, 0.0, -1.0]])
        target_pts = source_pts + translation
        source_pcd = _make_point_cloud(source_pts)
        target_pcd = _make_point_cloud(target_pts)
        corres = _make_correspondences(3)

        T = estimate_constrained_rototranslation(
            source_pcd, target_pcd, corres, unit_axis
        )

        # No rotation should be applied; only translation in x-y plane
        assert np.allclose(T[:3, :3], np.eye(3), atol=1e-10)


# ---------------------------------------------------------------------------
# TestAxisConstrainedTransformationEstimation
# ---------------------------------------------------------------------------


class TestAxisConstrainedTransformationEstimation:
    """Tests for the AxisConstrainedTransformationEstimation class interface."""

    def test_rotation_axis_property_returns_unit_vector(self):
        """The rotation_axis property must return a unit vector."""
        estimator = AxisConstrainedTransformationEstimation(np.array([0.0, 0.0, 3.0]))

        axis = estimator.rotation_axis

        assert np.isclose(np.linalg.norm(axis), 1.0)
        assert np.allclose(axis, np.array([0.0, 0.0, 1.0]))

    def test_compute_transformation_matches_function(self):
        """compute_transformation must return the same result as the standalone function."""
        axis = np.array([0.0, 0.0, 1.0])
        T_gt = _build_ground_truth_transform(axis, np.pi / 4, np.array([1.0, 2.0, 3.0]))
        target_pts = _apply_transform(SOURCE_POINTS, T_gt)
        source_pcd = _make_point_cloud(SOURCE_POINTS)
        target_pcd = _make_point_cloud(target_pts)
        corres = _make_correspondences(len(SOURCE_POINTS))

        estimator = AxisConstrainedTransformationEstimation(axis)
        T_class = estimator.compute_transformation(source_pcd, target_pcd, corres)
        T_func = estimate_constrained_rototranslation(
            source_pcd, target_pcd, corres, axis
        )

        assert np.allclose(T_class, T_func)

    def test_zero_axis_raises_at_construction(self):
        """Constructing with a zero axis must raise ValueError immediately."""
        with pytest.raises(ValueError, match="non-zero"):
            AxisConstrainedTransformationEstimation(np.zeros(3))

    def test_non_unit_axis_normalized_at_construction(self):
        """A non-unit axis passed at construction must be normalized."""
        estimator = AxisConstrainedTransformationEstimation(np.array([2.0, 0.0, 0.0]))

        assert np.allclose(estimator.rotation_axis, np.array([1.0, 0.0, 0.0]))

    def test_compute_transformation_various_axes(self):
        """Data-driven: compute_transformation recovers known transforms for various axes."""
        test_cases: list[_TransformCase] = [
            {
                "description": "z-axis 90 degrees with translation",
                "axis": np.array([0.0, 0.0, 1.0]),
                "angle": np.pi / 2,
                "translation": np.array([1.0, -1.0, 0.5]),
            },
            {
                "description": "arbitrary axis with translation",
                "axis": np.array([1.0, 1.0, 0.0]),
                "angle": np.pi / 3,
                "translation": np.array([0.0, 0.0, 2.0]),
            },
        ]

        for case in test_cases:
            T_gt = _build_ground_truth_transform(
                case["axis"], case["angle"], case["translation"]
            )
            target_pts = _apply_transform(SOURCE_POINTS, T_gt)
            source_pcd = _make_point_cloud(SOURCE_POINTS)
            target_pcd = _make_point_cloud(target_pts)
            corres = _make_correspondences(len(SOURCE_POINTS))

            estimator = AxisConstrainedTransformationEstimation(case["axis"])
            T_est = estimator.compute_transformation(source_pcd, target_pcd, corres)

            rot_err = np.linalg.norm(T_est[:3, :3] - T_gt[:3, :3])
            trans_err = np.linalg.norm(T_est[:3, 3] - T_gt[:3, 3])

            assert rot_err < 1e-8, (
                f"Rotation error too large for '{case['description']}': {rot_err}"
            )
            assert trans_err < 1e-8, (
                f"Translation error too large for '{case['description']}': {trans_err}"
            )
