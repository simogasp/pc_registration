#! /usr/bin/env python3
"""Flip and scale a point cloud along a specified axis."""

import argparse
import logging

import open3d as o3d
import numpy as np

from registration.utils.logging import setup_logging
from registration.visualization.viewer import print_point_cloud_info

from registration.utils.transforms import get_flip_transform

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    # add input file argument
    parser = argparse.ArgumentParser(
        description="Apply a scale wrt the origin and a flip transformation to a point cloud"
    )
    parser.add_argument("--input", type=str, help="Input file path", required=True)
    parser.add_argument("--output", type=str, help="Output file path", required=True)
    parser.add_argument(
        "--flip",
        type=str,
        choices=["x", "y", "z", "nx", "ny", "nz"],
        help="Flip axis (x, y, or z) by 90 degrees (prefix n for negative)",
        required=False,
    )
    parser.add_argument(
        "--scale", type=float, help="Scale factor (default: 1.0)", required=False
    )
    parser.add_argument(
        "-v",
        "--verbose",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set logging level (default: WARNING)",
    )
    args = parser.parse_args()

    # Set logging level based on user selection
    setup_logging(getattr(logging, args.verbose))

    pcd = o3d.io.read_point_cloud(args.input)
    print_point_cloud_info(pcd, args.input)

    if args.scale:
        logger.info(f"Scaling point cloud by factor {args.scale} wrt the origin")
        pcd.scale(args.scale, np.zeros(3))

    if args.flip:
        logger.info(f"Flipping point cloud along {args.flip} axis by 90 degrees")
        pcd.transform(get_flip_transform(args.flip))

    logger.info(f"Saving transformed point cloud to {args.output}")
    o3d.io.write_point_cloud(args.output, pcd)
