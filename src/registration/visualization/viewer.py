"""Visualization utilities for point clouds."""

import copy
import logging

import numpy as np
import open3d as o3d


def draw_registration_result(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    transformation: np.ndarray,
    window_name: str,
) -> None:
    """Visualize the registration result by applying transformation to source.

    Creates a visualization showing both the transformed source point cloud
    and the target point cloud. The source is colored yellow and the target
    is colored cyan for easy distinction.

    Args:
        source: Source point cloud to be transformed and displayed.
        target: Target point cloud to be displayed.
        transformation: 4x4 transformation matrix to apply to the source.
        window_name: Name of the visualization window.

    Note:
        This function blocks execution until the visualization window is closed.
        The original point clouds are not modified; copies are used for display.
    """
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])  # yellow
    target_temp.paint_uniform_color([0, 0.651, 0.929])  # cyan
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries(  # type: ignore
        [source_temp, target_temp], window_name=window_name
    )


def print_point_cloud_info(
    pcd: o3d.geometry.PointCloud, name: str = "Point cloud"
) -> None:
    """Print basic information about a point cloud.

    Displays the number of points and the axis-aligned bounding box (min and max
    coordinates) of the point cloud. Useful for debugging and understanding the
    scale and size of point cloud data.

    Args:
        pcd: The point cloud to analyze.
        name: A descriptive name for the point cloud (used in log messages).

    Note:
        Information is logged at DEBUG level for the number of points and
        bounding box coordinates.
    """
    num_points = len(pcd.points)
    aabb = pcd.get_axis_aligned_bounding_box()
    obb = pcd.get_oriented_bounding_box()
    logging.info(f"Point Cloud '{name}':")
    logging.info(f"\tNumber of points: {num_points}")
    logging.info(f"\tHas normals: {pcd.has_normals()}")
    logging.info(f"\tPoint cloud size: {pcd.get_max_bound() - pcd.get_min_bound()}")
    logging.info(
        f"\tAxis-Aligned Bounding Box: min {aabb.min_bound}, max {aabb.max_bound}"
    )
    logging.info(f"\tOriented Bounding Box: center {obb.center}, extent {obb.extent}")
