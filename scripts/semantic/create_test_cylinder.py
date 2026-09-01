"""Create a test PLY file with cylindrical data for testing."""

from pathlib import Path

import numpy as np
import open3d as o3d


def create_cylinder_point_cloud(
    center=(0, 0, 0), axis=(0, 0, 1), radius=1.0, height=5.0, n_points=1000, noise=0.01
):
    """Create a synthetic point cloud of a cylinder.

    Args:
        center: Center point of the cylinder.
        axis: Axis direction of the cylinder.
        radius: Radius of the cylinder.
        height: Height of the cylinder.
        n_points: Number of points to generate.
        noise: Gaussian noise standard deviation.

    Returns:
        Open3D point cloud of a cylinder.
    """
    # Normalize axis
    axis = np.array(axis)
    axis = axis / np.linalg.norm(axis)

    # Generate points on cylinder surface
    # Angular parameter
    theta = np.random.uniform(0, 2 * np.pi, n_points)
    # Height parameter
    h = np.random.uniform(-height / 2, height / 2, n_points)

    # Points in cylinder coordinate system (axis is z)
    x_cyl = radius * np.cos(theta)
    y_cyl = radius * np.sin(theta)
    z_cyl = h

    # Create rotation matrix to align z-axis with desired axis
    z = np.array([0, 0, 1])
    if np.allclose(axis, z):
        R = np.eye(3)
    elif np.allclose(axis, -z):
        R = np.diag([1, -1, -1])
    else:
        v = np.cross(z, axis)
        s = np.linalg.norm(v)
        c = np.dot(z, axis)
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R = np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s))

    # Transform points
    points_cyl = np.column_stack([x_cyl, y_cyl, z_cyl])
    points = (R @ points_cyl.T).T + np.array(center)

    # Add noise
    points += np.random.normal(0, noise, points.shape)

    # Create point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Estimate normals
    pcd.estimate_normals()

    return pcd


if __name__ == "__main__":
    # Create output directory
    output_dir = Path("test_data")
    output_dir.mkdir(exist_ok=True)

    # Create cylinder point cloud
    print("Creating synthetic cylinder point cloud...")
    pcd = create_cylinder_point_cloud(
        center=(10, 5, 0),
        axis=(0, 0, 1),
        radius=0.5,
        height=10.0,
        n_points=2000,
        noise=0.005,
    )

    # Save
    output_path = output_dir / "test_cylinder.ply"
    o3d.io.write_point_cloud(str(output_path), pcd)

    print(f"Saved cylinder point cloud to: {output_path}")
    print(f"Number of points: {len(pcd.points)}")
