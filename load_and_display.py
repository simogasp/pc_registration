#! /usr/bin/env python3

import open3d as o3d
import argparse
import logging
from common import *

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    # add input file argument
    argparse = argparse.ArgumentParser(
        description="Load and display a point cloud with its bounding boxes"
    )
    argparse.add_argument("--input", type=str, help="Input file path", required=True)
    argparse.add_argument(
        "-v",
        "--verbose",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set logging level (default: WARNING)",
    )
    args = argparse.parse_args()

    # Set logging level based on user selection
    setup_logging(getattr(logging, args.verbose))

    pcd = o3d.io.read_point_cloud(args.input)
    print_point_cloud_info(pcd, args.input)
    # Flip it, otherwise the pointcloud will be upside down.
    # pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    # print(pcd)
    axis_aligned_bounding_box = pcd.get_axis_aligned_bounding_box()
    axis_aligned_bounding_box.color = (1, 0, 0)
    oriented_bounding_box = pcd.get_oriented_bounding_box()
    oriented_bounding_box.color = (0, 1, 0)
    logger.info(
        "Displaying axis_aligned_bounding_box in red and oriented bounding box in green ..."
    )
    o3d.visualization.draw([pcd, axis_aligned_bounding_box, oriented_bounding_box])
