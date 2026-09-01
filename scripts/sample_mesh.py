#!/usr/bin/env python3
"""Sample a point cloud from a mesh surface.

This script loads a triangle mesh in any format supported by Open3D and samples
a point cloud from its surface using one of two strategies:

- uniform:      Monte Carlo sampling proportional to triangle area.
- poisson-disk: Poisson-disk sampling for a more spatially uniform distribution
                (no two output points are closer than a minimum radius).

Surface normals of the sampled points are inherited from the mesh face normals.
"""

import argparse
import logging
import math
import sys
from enum import Enum
from pathlib import Path

import open3d as o3d

from registration.utils.logging import setup_logging

logger = logging.getLogger(__name__)


class SamplingMethod(str, Enum):
    """Available mesh-to-point-cloud sampling methods."""

    UNIFORM = "uniform"
    POISSON_DISK = "poisson-disk"


def load_mesh(mesh_path: Path) -> o3d.geometry.TriangleMesh:
    """Load a triangle mesh from disk.

    Args:
        mesh_path: Path to the mesh file (any format supported by Open3D).

    Returns:
        Loaded triangle mesh.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file could not be loaded or contains no triangles.
    """
    if not mesh_path.exists():
        raise FileNotFoundError(f"Mesh file not found: {mesh_path}")

    logger.info(f"Loading mesh from: {mesh_path}")
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))

    if not mesh.has_triangles():
        raise ValueError(
            f"File contains no triangles: {mesh_path}. "
            "Make sure the input is a mesh, not a point cloud."
        )

    num_vertices = len(mesh.vertices)
    num_triangles = len(mesh.triangles)
    logger.info(f"Mesh loaded: {num_vertices} vertices, {num_triangles} triangles")

    return mesh


def prepare_mesh_normals(mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
    """Ensure the mesh has consistent vertex normals for normal transfer.

    Args:
        mesh: Input triangle mesh.

    Returns:
        Mesh with computed vertex normals.
    """
    mesh.compute_vertex_normals()
    return mesh


def resolve_num_points(
    mesh: o3d.geometry.TriangleMesh,
    num_points: int | None,
    spacing: float | None,
    method: "SamplingMethod",
) -> int:
    """Compute the number of sample points from either ``--num-points`` or ``--spacing``.

    When ``spacing`` is provided the target count is derived from the mesh
    surface area so that the expected average distance between neighbouring
    points equals ``spacing``:

    - **uniform**: each point occupies a surface cell of area ``spacing^2``,
      so ``N = surface_area / spacing^2``.
    - **poisson-disk**: the minimum inter-point distance equals ``spacing``
      and Open3D derives it from ``N`` via
      ``r_min = sqrt(surface_area / (N * pi))``, so
      ``N = surface_area / (pi * spacing^2)``.

    Args:
        mesh: Triangle mesh (used to compute surface area when spacing is given).
        num_points: Explicit point count, or None if spacing is used.
        spacing: Desired average inter-point distance in mesh units, or None.
        method: Sampling method (affects the spacing-to-N conversion).

    Returns:
        Resolved number of points to request from the sampler.

    Raises:
        ValueError: If neither or both of ``num_points`` and ``spacing`` are given,
            or if the derived count is less than 1.
    """
    if num_points is not None and spacing is not None:
        raise ValueError("--num-points and --spacing are mutually exclusive.")

    if num_points is not None:
        return num_points

    if spacing is None:
        raise ValueError("Either --num-points or --spacing must be provided.")

    # Derive N from spacing and surface area.
    surface_area = mesh.get_surface_area()
    logger.debug(f"Mesh surface area: {surface_area:.4f} (mesh units^2)")

    if method == SamplingMethod.UNIFORM:
        n = surface_area / (spacing**2)
    else:  # poisson-disk
        n = surface_area / (math.pi * spacing**2)

    result = max(1, round(n))
    logger.info(
        f"Spacing {spacing} -> surface area {surface_area:.4f} -> "
        f"target N = {result} points"
    )
    return result


def sample_uniform(
    mesh: o3d.geometry.TriangleMesh,
    num_points: int,
) -> o3d.geometry.PointCloud:
    """Sample points uniformly at random from the mesh surface.

    Points are drawn with probability proportional to triangle area (Monte Carlo
    sampling). The resulting distribution is statistically uniform over the
    surface but may contain local clustering artefacts.

    The output contains **exactly** ``num_points`` points.

    Args:
        mesh: Triangle mesh to sample from. Must have vertex normals.
        num_points: Exact number of points in the output cloud.

    Returns:
        Sampled point cloud with normals.
    """
    logger.info(f"Uniform sampling: {num_points} points")
    pcd = mesh.sample_points_uniformly(number_of_points=num_points)
    return pcd


def sample_poisson_disk(
    mesh: o3d.geometry.TriangleMesh,
    num_points: int,
    init_factor: int,
    use_triangle_normal: bool,
) -> o3d.geometry.PointCloud:
    """Sample points using Poisson-disk sampling.

    First generates an oversampled uniform cloud of ``num_points * init_factor``
    points, then eliminates samples that are too close to each other using a
    farthest-point strategy.

    ``num_points`` controls the **minimum inter-point distance** via:

        r_min = sqrt(surface_area / (num_points * pi))

    The actual output count may be **lower** than ``num_points`` when the
    surface area is too small to place that many non-overlapping disks of
    radius ``r_min``. It is therefore a target, not a guarantee.

    Args:
        mesh: Triangle mesh to sample from. Must have vertex normals.
        num_points: Target number of output points. Determines the minimum
            inter-point distance; the actual count may be lower.
        init_factor: Oversampling factor for the initial uniform sample
            (higher values improve quality at the cost of memory; >= 5 recommended).
        use_triangle_normal: If True, assign face normals to sampled points
            instead of interpolated vertex normals.

    Returns:
        Sampled point cloud with normals.
    """
    logger.info(
        f"Poisson-disk sampling: {num_points} points, "
        f"init_factor={init_factor}, use_triangle_normal={use_triangle_normal}"
    )
    pcd = mesh.sample_points_poisson_disk(
        number_of_points=num_points,
        init_factor=init_factor,
        use_triangle_normal=use_triangle_normal,
    )
    return pcd


def sample_mesh(
    mesh: o3d.geometry.TriangleMesh,
    method: SamplingMethod,
    num_points: int,
    init_factor: int,
    use_triangle_normal: bool,
) -> o3d.geometry.PointCloud:
    """Dispatch mesh sampling to the selected method.

    Args:
        mesh: Triangle mesh to sample from.
        method: Sampling method to use.
        num_points: Number of points to sample.
        init_factor: Oversampling factor (Poisson-disk only).
        use_triangle_normal: Use triangle normals instead of vertex normals
            (Poisson-disk only).

    Returns:
        Sampled point cloud.
    """
    if method == SamplingMethod.UNIFORM:
        return sample_uniform(mesh, num_points)
    if method == SamplingMethod.POISSON_DISK:
        return sample_poisson_disk(mesh, num_points, init_factor, use_triangle_normal)
    raise ValueError(f"Unknown sampling method: {method}")


def log_point_cloud_stats(pcd: o3d.geometry.PointCloud) -> None:
    """Log basic statistics of the sampled point cloud.

    Args:
        pcd: Sampled point cloud.
    """
    num_points = len(pcd.points)
    has_normals = pcd.has_normals()
    has_colors = pcd.has_colors()
    logger.info(
        f"Sampled point cloud: {num_points} points, "
        f"normals={has_normals}, colors={has_colors}"
    )


def save_point_cloud(pcd: o3d.geometry.PointCloud, output_path: Path) -> None:
    """Save the point cloud to disk in binary PLY format.

    Args:
        output_path: Destination file path (must have a .ply extension).
        pcd: Point cloud to save.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    success = o3d.io.write_point_cloud(str(output_path), pcd, write_ascii=False)
    if not success:
        raise RuntimeError(f"Failed to write point cloud to: {output_path}")

    size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info(f"Saved: {output_path} ({size_mb:.2f} MB)")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description="Sample a point cloud from a mesh surface.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Path to the input mesh file (any format supported by Open3D: "
        ".ply, .obj, .stl, .off, .gltf, .glb, ...).",
    )

    parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="Path for the output point cloud file (.ply).",
    )

    parser.add_argument(
        "--num-points",
        "-n",
        type=int,
        default=None,
        help=(
            "Number of points to sample. For 'uniform': exact output count. "
            "For 'poisson-disk': target that sets the minimum inter-point distance "
            "r_min = sqrt(surface_area / (N * pi)); actual output may be lower. "
            "Mutually exclusive with --spacing. Defaults to 100000 when neither is given."
        ),
    )

    parser.add_argument(
        "--spacing",
        "-d",
        type=float,
        default=None,
        help=(
            "Desired average distance between sampled points, in mesh units "
            "(e.g. 0.5 for 0.5 mm when the mesh is in mm). "
            "The required point count is derived automatically from the mesh surface area. "
            "For 'uniform': N = surface_area / spacing^2. "
            "For 'poisson-disk': N = surface_area / (pi * spacing^2). "
            "Mutually exclusive with --num-points."
        ),
    )

    parser.add_argument(
        "--method",
        "-m",
        choices=[m.value for m in SamplingMethod],
        default=SamplingMethod.POISSON_DISK.value,
        help=(
            "Sampling method. "
            "'uniform': Monte Carlo sampling proportional to triangle area. "
            "'poisson-disk': spatially uniform sampling with minimum inter-point distance."
        ),
    )

    parser.add_argument(
        "--init-factor",
        type=int,
        default=5,
        help=(
            "Poisson-disk only. Oversampling factor for the initial uniform cloud "
            "before elimination. Higher values improve uniformity (>= 5 recommended)."
        ),
    )

    parser.add_argument(
        "--use-triangle-normal",
        action="store_true",
        default=False,
        help=(
            "Poisson-disk only. Assign flat face normals to sampled points instead "
            "of interpolated vertex normals."
        ),
    )

    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity level.",
    )

    return parser.parse_args()


def main() -> None:
    """Entry point."""
    args = parse_args()
    setup_logging(args.log_level)

    mesh_path = Path(args.input)
    output_path = Path(args.output)
    method = SamplingMethod(args.method)

    if args.num_points is not None and args.num_points < 1:
        logger.error(f"--num-points must be >= 1, got {args.num_points}")
        sys.exit(1)

    if args.spacing is not None and args.spacing <= 0.0:
        logger.error(f"--spacing must be > 0, got {args.spacing}")
        sys.exit(1)

    if args.init_factor < 1:
        logger.error(f"--init-factor must be >= 1, got {args.init_factor}")
        sys.exit(1)

    # Default when neither density flag is given.
    num_points_arg = args.num_points if (args.num_points or args.spacing) else 100_000

    try:
        mesh = load_mesh(mesh_path)
        mesh = prepare_mesh_normals(mesh)
        num_points = resolve_num_points(mesh, num_points_arg, args.spacing, method)
        pcd = sample_mesh(
            mesh,
            method=method,
            num_points=num_points,
            init_factor=args.init_factor,
            use_triangle_normal=args.use_triangle_normal,
        )
        log_point_cloud_stats(pcd)
        save_point_cloud(pcd, output_path)

    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        logger.error(str(exc))
        sys.exit(1)


if __name__ == "__main__":
    main()
