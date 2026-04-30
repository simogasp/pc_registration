"""Convert a row-major flat-text poses file to per-scan JSON pose files.

Each line of the input file contains 16 space-separated floats representing
a 4x4 transformation matrix in row-major order. The script writes one JSON
file per line (0.json, 1.json, ...) in the format consumed by
registration_common.load_transformation_matrix:

    { "H": [[r00, r01, r02, t0],
             [r10, r11, r12, t1],
             [r20, r21, r22, t2],
             [0,   0,   0,   1 ]] }
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List

# Add scripts directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np

from registration_common import write_pose_json

logger = logging.getLogger(__name__)

MATRIX_SIZE = 4
FLOATS_PER_LINE = MATRIX_SIZE * MATRIX_SIZE


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logger to write to stdout.

    Args:
        level: Logging level (e.g. logging.INFO).
    """
    logging.basicConfig(
        stream=sys.stdout,
        level=level,
        format="[%(asctime)s][%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_line(line: str, line_number: int) -> np.ndarray:
    """Parse one line of text into a 4x4 matrix.

    Args:
        line: Space-separated string of 16 floats.
        line_number: 1-based line index used in error messages.

    Returns:
        4x4 numpy array in row-major order.

    Raises:
        ValueError: If the line does not contain exactly 16 floats.
    """
    values = line.split()
    if len(values) != FLOATS_PER_LINE:
        raise ValueError(
            f"Line {line_number}: expected {FLOATS_PER_LINE} values, "
            f"got {len(values)}"
        )
    return np.array([float(v) for v in values], dtype=float).reshape(
        MATRIX_SIZE, MATRIX_SIZE
    )


def read_poses(poses_file: Path) -> List[np.ndarray]:
    """Read all poses from a flat text file.

    Blank lines and lines starting with '#' are skipped.

    Args:
        poses_file: Path to the text file.

    Returns:
        List of 4x4 numpy arrays, one per non-empty line.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If any line has an unexpected number of values.
    """
    if not poses_file.exists():
        raise FileNotFoundError(f"Poses file not found: {poses_file}")

    poses = []
    line_number = 0
    with open(poses_file, "r") as f:
        for raw_line in f:
            line_number += 1
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            poses.append(parse_line(line, line_number))

    logger.info(f"Read {len(poses)} poses from {poses_file}")
    return poses


def scale_translation(matrix: np.ndarray, scale: float) -> np.ndarray:
    """Return a copy of a 4x4 pose matrix with the translation scaled.

    Only the first three elements of the last column (tx, ty, tz) are
    affected; the rotation block and the homogeneous row are left unchanged.

    Args:
        matrix: 4x4 numpy array representing the pose.
        scale: Multiplicative factor applied to the translation components.

    Returns:
        New 4x4 numpy array with scaled translation.
    """
    scaled = matrix.copy()
    scaled[:3, 3] *= scale
    return scaled


def write_all_poses(
    poses: List[np.ndarray], output_dir: Path, translation_scale: float = 1.0
) -> None:
    """Write each pose as a numbered JSON file inside output_dir.

    Files are named 0.json, 1.json, ... in the same order as the input list.

    Args:
        poses: List of 4x4 pose matrices.
        output_dir: Directory where JSON files will be written.
        translation_scale: Scale factor applied to the translation part of
            each matrix before writing (default: 1.0, no scaling).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    for idx, matrix in enumerate(poses):
        output_path = output_dir / f"{idx}.json"
        write_pose_json(scale_translation(matrix, translation_scale), output_path)
        logger.debug(f"Written {output_path}")
    logger.info(f"Wrote {len(poses)} JSON files to {output_dir}")


def main() -> None:
    """CLI entry point for poses_txt_to_json."""
    parser = argparse.ArgumentParser(
        description=(
            "Convert a row-major flat-text poses file to per-scan JSON pose files. "
            "Each line of the input must contain 16 space-separated floats."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        type=str,
        help="Path to the input poses text file.",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default=None,
        help=(
            "Directory where JSON files will be written. "
            "Defaults to the same directory as the input file."
        ),
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

    input_path = Path(args.input)
    output_dir = Path(args.output_dir) if args.output_dir else input_path.parent

    poses = read_poses(input_path)
    if args.scale != 1.0:
        logger.info(f"Applying translation scale factor: {args.scale}")
    write_all_poses(poses, output_dir, translation_scale=args.scale)


if __name__ == "__main__":
    main()