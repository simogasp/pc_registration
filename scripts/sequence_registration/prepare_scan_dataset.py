"""Convert a directory of XYZ scans and a flat-text poses file to binary PLY and JSON.

The script expects an input directory containing:
- One or more point cloud files in XYZ format (*.xyz), typically with numeric stems
  (0.xyz, 1.xyz, ...).
- A flat-text poses file (default: poses.txt) where each non-empty, non-comment
  line holds 16 space-separated floats representing a 4x4 transformation matrix
  in row-major order.

For each XYZ file found (sorted numerically by stem) the script:
1. Converts the point cloud to binary PLY format, preserving the stem name.
2. Writes the corresponding pose as a JSON file with the same stem, in the format
   consumed by registration_common.load_transformation_matrix:

       { "H": [[r00, r01, r02, tx],
                [r10, r11, r12, ty],
                [r20, r21, r22, tz],
                [0,   0,   0,   1 ]] }

Output files are written to the input directory unless --output-dir is specified.
"""

import argparse
import logging
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List

# Add scripts directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import open3d as o3d

from registration.utils.logging import setup_logging
from registration_common import read_poses_from_txt, scale_translation, write_pose_json

logger = logging.getLogger(__name__)

DEFAULT_POSES_FILENAME = "poses.txt"
DEFAULT_WORKERS = 4


def extract_scan_index(stem: str) -> str:
    """Extract the trailing integer from a file stem and return it as a string.

    Handles stems like 'scan_0', 'frame_042', or plain '7'.
    Falls back to the full stem when no trailing integer is found.

    Args:
        stem: Filename without extension (e.g. 'scan_0', '42').

    Returns:
        The trailing integer as a zero-stripped string (e.g. '0', '42'),
        or the original stem if no trailing integer is present.
    """
    match = re.search(r"(\d+)$", stem)
    return match.group(1) if match else stem


def _convert_worker(args: tuple) -> str:
    """Top-level worker function for parallel XYZ-to-PLY conversion.

    Defined at module level so it can be pickled by ProcessPoolExecutor.

    Args:
        args: Tuple of (xyz_path, output_path, scale) matching convert_xyz_to_ply.

    Returns:
        Name of the written output file.
    """
    xyz_path, output_path, scale = args
    convert_xyz_to_ply(xyz_path, output_path, scale=scale)
    return output_path.name


def find_xyz_files(data_dir: Path) -> List[Path]:
    """Find all XYZ point cloud files in a directory, sorted numerically by stem.

    Args:
        data_dir: Directory to search.

    Returns:
        List of paths to .xyz files, sorted with numeric stems first.

    Raises:
        FileNotFoundError: If no .xyz files are found in the directory.
    """
    xyz_files = sorted(
        data_dir.glob("*.xyz"),
        key=lambda p: (
            int(extract_scan_index(p.stem))
            if extract_scan_index(p.stem).isdigit()
            else extract_scan_index(p.stem)
        ),
    )
    if not xyz_files:
        raise FileNotFoundError(f"No .xyz files found in {data_dir}")
    logger.info(f"Found {len(xyz_files)} XYZ files in {data_dir}")
    return xyz_files


def scale_point_cloud(
    pcd: o3d.geometry.PointCloud, scale: float
) -> o3d.geometry.PointCloud:
    """Return a copy of a point cloud with all coordinates multiplied by scale.

    Args:
        pcd: Input point cloud.
        scale: Multiplicative factor applied to every XYZ coordinate.

    Returns:
        New point cloud with scaled coordinates (colors and normals preserved).
    """
    points = np.asarray(pcd.points) * scale
    scaled = o3d.geometry.PointCloud()
    scaled.points = o3d.utility.Vector3dVector(points)
    if pcd.has_colors():
        scaled.colors = pcd.colors
    if pcd.has_normals():
        scaled.normals = pcd.normals
    return scaled


def convert_xyz_to_ply(xyz_path: Path, output_path: Path, scale: float = 1.0) -> None:
    """Convert a single XYZ point cloud file to binary PLY format.

    Args:
        xyz_path: Path to the source .xyz file.
        output_path: Destination .ply file path.
        scale: Multiplicative factor applied to all XYZ coordinates before
            writing (default: 1.0, no scaling).

    Raises:
        ValueError: If the loaded point cloud contains no points.
    """
    pcd = o3d.io.read_point_cloud(str(xyz_path))
    if not pcd.has_points():
        raise ValueError(f"Point cloud {xyz_path} is empty")
    if scale != 1.0:
        pcd = scale_point_cloud(pcd, scale)
    o3d.io.write_point_cloud(str(output_path), pcd, write_ascii=False)
    logger.debug(f"Converted {xyz_path.name} -> {output_path.name}")


def convert_all_xyz_files(
    xyz_files: List[Path], output_dir: Path, scale: float = 1.0, workers: int = 1
) -> None:
    """Convert a list of XYZ files to binary PLY files in the output directory.

    Output files share the stem of the corresponding input file (e.g. 0.xyz -> 0.ply).
    When workers > 1, conversions run in parallel using a process pool.

    Args:
        xyz_files: Ordered list of .xyz file paths.
        output_dir: Directory where PLY files will be written.
        scale: Multiplicative factor applied to all XYZ coordinates of each
            point cloud before writing (default: 1.0, no scaling).
        workers: Number of worker processes for parallel conversion (default: 1).
    """
    tasks = [
        (xyz_path, output_dir / f"{extract_scan_index(xyz_path.stem)}.ply", scale)
        for xyz_path in xyz_files
    ]
    if workers > 1:
        logger.info(f"Converting {len(xyz_files)} scans using {workers} workers...")
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_convert_worker, task): task for task in tasks}
            for future in as_completed(futures):
                name = future.result()  # re-raises any exception from the worker
                logger.debug(f"Converted -> {name}")
    else:
        logger.info(f"Converting {len(xyz_files)} scans sequentially...")
        for task in tasks:
            _convert_worker(task)
    logger.info(f"Wrote {len(xyz_files)} PLY files to {output_dir}")


def write_pose_jsons(
    poses: List[np.ndarray],
    xyz_files: List[Path],
    output_dir: Path,
    translation_scale: float = 1.0,
) -> None:
    """Write one JSON pose file per XYZ scan.

    Poses and XYZ files are matched by position in their respective lists.
    If the counts differ, only the overlapping range is processed and a
    warning is emitted.

    Args:
        poses: List of 4x4 pose matrices read from the poses file.
        xyz_files: Ordered list of .xyz file paths (determines output stem names).
        output_dir: Directory where JSON files will be written.
        translation_scale: Scale factor applied to the translation components of
            each pose before writing (default: 1.0, no scaling).
    """
    n_pairs = min(len(poses), len(xyz_files))
    if len(poses) != len(xyz_files):
        logger.warning(
            f"Number of poses ({len(poses)}) does not match number of XYZ files "
            f"({len(xyz_files)}); processing {n_pairs} pairs."
        )
    for idx in range(n_pairs):
        matrix = scale_translation(poses[idx], translation_scale)
        output_path = output_dir / f"{extract_scan_index(xyz_files[idx].stem)}.json"
        write_pose_json(matrix, output_path)
        logger.debug(f"Written {output_path.name}")
    logger.info(f"Wrote {n_pairs} JSON pose files to {output_dir}")


def main() -> None:
    """CLI entry point for prepare_scan_dataset."""
    parser = argparse.ArgumentParser(
        description=(
            "Convert XYZ point cloud files and a flat-text poses file to binary "
            "PLY and JSON format compatible with the registration scripts."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        type=str,
        help="Directory containing .xyz scan files and the poses file.",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default=None,
        help=(
            "Directory where PLY and JSON files will be written. "
            "Defaults to the input directory."
        ),
    )
    parser.add_argument(
        "--poses",
        "-p",
        type=str,
        default=None,
        help=(
            f"Path to the flat-text poses file. "
            f"Defaults to '{DEFAULT_POSES_FILENAME}' inside the input directory."
        ),
    )
    parser.add_argument(
        "--workers",
        "-w",
        type=int,
        default=DEFAULT_WORKERS,
        help="Number of parallel worker processes for XYZ-to-PLY conversion.",
    )
    parser.add_argument(
        "--scale",
        "-s",
        type=float,
        default=1.0,
        help="Scale factor applied to the translation part of each pose (default: 1.0).",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Logging level.",
    )

    args = parser.parse_args()
    setup_logging(level=getattr(logging, args.log_level))

    input_dir = Path(args.input)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir
    poses_file = Path(args.poses) if args.poses else input_dir / DEFAULT_POSES_FILENAME

    output_dir.mkdir(parents=True, exist_ok=True)

    if args.scale != 1.0:
        logger.info(f"Applying translation scale factor: {args.scale}")

    xyz_files = find_xyz_files(input_dir)
    poses = read_poses_from_txt(poses_file)

    t_start = time.perf_counter()
    convert_all_xyz_files(xyz_files, output_dir, scale=args.scale, workers=args.workers)
    elapsed = time.perf_counter() - t_start
    logger.info(
        f"XYZ-to-PLY conversion completed in {elapsed:.2f} s ({elapsed / len(xyz_files):.3f} s/scan)"
    )

    write_pose_jsons(poses, xyz_files, output_dir, translation_scale=args.scale)


if __name__ == "__main__":
    main()
