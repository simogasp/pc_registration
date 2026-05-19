"""Test script to verify the plane annotation with perfect planar data."""

import numpy as np
import open3d as o3d
from pathlib import Path
import sys
from typing import Tuple

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from scripts.semantic.pc_annotation import compute_obb, compute_planar_obb

# Test parameters
NUM_POINTS = 100
EXTENT_RANGE = 10.0
MARGIN = 0.01
RANDOM_SEED = 42


def get_plane_basis(normal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute two orthogonal basis vectors in the plane perpendicular to the normal.

    Args:
        normal: Normal vector of the plane (will be normalized internally).

    Returns:
        Tuple of (u, v) - two orthonormal vectors in the plane.
    """
    # Normalize the normal vector
    normal = normal / np.linalg.norm(normal)

    # Find a vector not parallel to normal
    if np.abs(normal[2]) < 0.9:
        # Use z-axis if normal is not too close to z
        v1 = np.array([0, 0, 1])
    else:
        # Use x-axis otherwise
        v1 = np.array([0, 1, 0])

    # Compute first basis vector using cross product
    u = np.cross(normal, v1)
    u = u / np.linalg.norm(u)

    # Compute second basis vector
    v = np.cross(normal, u)
    v = v / np.linalg.norm(v)

    return u, v


def create_plane_points_generic(
    point_on_plane: np.ndarray,
    plane_normal: np.ndarray,
    extent_u: float = EXTENT_RANGE,
    extent_v: float = EXTENT_RANGE / 2,
    n_points: int = NUM_POINTS,
) -> np.ndarray:
    """Create points on an arbitrary plane defined by a point and normal.

    Args:
        point_on_plane: A point on the plane (3D).
        plane_normal: Normal vector to the plane (3D, will be normalized).
        extent_u: Range for the first in-plane direction (uses -extent_u to +extent_u).
        extent_v: Range for the second in-plane direction (uses -extent_v to +extent_v).
        n_points: Number of points to generate.

    Returns:
        Array of points lying on the plane.
    """
    np.random.seed(RANDOM_SEED)

    # Get orthonormal basis vectors in the plane
    u, v = get_plane_basis(plane_normal)

    # Generate random coefficients for the basis vectors
    coeffs_u = np.random.uniform(-extent_u, extent_u, n_points)
    coeffs_v = np.random.uniform(-extent_v, extent_v, n_points)

    # Create points: point_on_plane + a*u + b*v
    points = (
        point_on_plane[np.newaxis, :]
        + coeffs_u[:, np.newaxis] * u[np.newaxis, :]
        + coeffs_v[:, np.newaxis] * v[np.newaxis, :]
    )

    return points


def create_point_cloud(points: np.ndarray) -> o3d.geometry.PointCloud:
    """Create an Open3D point cloud from a numpy array.

    Args:
        points: Nx3 array of point coordinates.

    Returns:
        Open3D PointCloud object.
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd


def verify_planar_obb(
    obb: dict, expected_normal_extent: float, plane_name: str
) -> bool:
    """Verify that an OBB has the expected properties for a planar object.

    Args:
        obb: OBB dictionary with 'center', 'extents', and 'axes'.
        expected_normal_extent: Expected extent in the normal direction.
        plane_name: Name of the plane for error messages.

    Returns:
        True if all verifications pass.

    Raises:
        ValueError: If any verification fails.
    """
    extents = obb["extents"]

    # All extents should be positive
    if extents[0] <= 0:
        raise ValueError(f"{plane_name}: U-extent should be positive")
    if extents[1] <= 0:
        raise ValueError(f"{plane_name}: V-extent should be positive")
    if extents[2] <= 0:
        raise ValueError(f"{plane_name}: Normal extent should be positive")

    # Normal extent should match expected value (2 * margin)
    if not np.isclose(extents[2], expected_normal_extent, rtol=1e-10):
        raise ValueError(
            f"{plane_name}: Normal extent should be {expected_normal_extent}, "
            f"got {extents[2]}"
        )

    # Center should be a valid 3D point
    if len(obb["center"]) != 3:
        raise ValueError(f"{plane_name}: Center should be 3D")

    # Axes should be a 3x3 matrix
    axes = np.array(obb["axes"])
    if axes.shape != (3, 3):
        raise ValueError(f"{plane_name}: Axes should be 3x3 matrix")

    return True


def test_planar_obb_generic(
    point_on_plane: np.ndarray,
    plane_normal: np.ndarray,
    plane_name: str,
    extent_u: float = EXTENT_RANGE,
    extent_v: float = EXTENT_RANGE / 2,
) -> bool:
    """Generic test for planar OBB computation with arbitrary plane.

    Args:
        point_on_plane: A point on the plane.
        plane_normal: Normal vector to the plane.
        plane_name: Descriptive name for the plane (for logging).
        extent_u: Range for first in-plane direction.
        extent_v: Range for second in-plane direction.

    Returns:
        True if test passes, False otherwise.
    """
    print(f"\nTesting {plane_name} planar OBB computation...")

    # Create points on the specified plane
    points = create_plane_points_generic(
        point_on_plane, plane_normal, extent_u, extent_v
    )
    pcd = create_point_cloud(points)

    print(f"  Number of points: {len(points)}")
    print(f"  Plane normal: {plane_normal}")
    print(f"  Margin: {MARGIN}")

    try:
        # Compute OBB for planar data
        obb = compute_planar_obb(pcd, plane_normal, margin=MARGIN)

        print("  ✓ OBB computed successfully!")
        print(f"    Center: {np.array(obb['center'])}")
        print(f"    Extents: {obb['extents']}")
        print(f"    Axes:\n{np.array(obb['axes'])}")

        # Verify OBB properties
        expected_normal_extent = 2 * MARGIN
        verify_planar_obb(obb, expected_normal_extent, plane_name)

        center = np.array(obb["center"])
        axes = np.array(obb["axes"])
        extents = np.array(obb["extents"])
        corners = []
        for i in [-1, 1]:
            for j in [-1, 1]:
                for k in [-1, 1]:
                    corner = (
                        center
                        + i * extents[0] / 2 * axes[0]
                        + j * extents[1] / 2 * axes[1]
                        + k * extents[2] / 2 * axes[2]
                    )
                    corners.append(corner)

        corners = np.array(corners)
        print(f"    Corners:\n{corners}")

        print(f"  ✓ All assertions passed for {plane_name}!")
        return True

    except Exception as e:
        print(f"  ✗ Error for {plane_name}: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_planar_obb_z0() -> bool:
    """Test planar OBB computation for z=0 plane."""
    point = np.array([0, 0, 0])
    normal = np.array([0, 0, 1])
    return test_planar_obb_generic(point, normal, "z=0")


def test_planar_obb_x0() -> bool:
    """Test planar OBB computation for x=0 plane."""
    point = np.array([0, 0, 0])
    normal = np.array([1, 0, 0])
    return test_planar_obb_generic(point, normal, "x=0")


def test_planar_obb_y0() -> bool:
    """Test planar OBB computation for y=0 plane."""
    point = np.array([0, 0, 0])
    normal = np.array([0, 1, 0])
    return test_planar_obb_generic(point, normal, "y=0")


def test_planar_obb_arbitrary() -> bool:
    """Test planar OBB computation for an arbitrary tilted plane."""
    # Define a plane: passing through (1, 2, 3) with normal (1, 1, 1)
    point = np.array([1.0, 2.0, 3.0])
    normal = np.array([1.0, 1.0, 1.0])  # Will be normalized
    return test_planar_obb_generic(point, normal, "arbitrary (1,1,1) normal")


def test_3d_obb_fallback() -> bool:
    """Test that standard 3D OBB computation still works."""
    print("\n\nTesting standard 3D OBB computation...")

    # Create 3D cube with points distributed in all dimensions
    np.random.seed(RANDOM_SEED)
    points = np.random.uniform(-EXTENT_RANGE / 10, EXTENT_RANGE / 10, (NUM_POINTS, 3))
    pcd = create_point_cloud(points)

    try:
        obb = compute_obb(pcd)

        print("  ✓ 3D OBB computed successfully!")
        print(f"    Center: {obb['center']}")
        print(f"    Extents: {obb['extents']}")

        # Verify basic properties
        extents = obb["extents"]
        if len(extents) != 3:
            raise ValueError("OBB should have 3 extents")
        if not all(e > 0 for e in extents):
            raise ValueError("All extents should be positive")

        return True

    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("Running Planar OBB Tests")
    print("=" * 60)

    success = True
    success = test_planar_obb_z0() and success
    # success = test_planar_obb_x0() and success
    # success = test_planar_obb_y0() and success
    # success = test_planar_obb_arbitrary() and success
    # success = test_3d_obb_fallback() and success

    print("\n" + "=" * 60)
    if success:
        print("✓ All tests passed!")
    else:
        print("✗ Some tests failed")
    print("=" * 60)

    sys.exit(0 if success else 1)
