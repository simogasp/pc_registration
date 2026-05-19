#! /usr/bin/env python3
"""Load and display a point cloud file."""

import argparse
import logging

import open3d as o3d

from registration.utils.logging import setup_logging
from registration.visualization.viewer import print_point_cloud_info
from registration.utils.transforms import (
    get_flip_transform,
)
from global_registration import rough_scale_point_cloud

logger = logging.getLogger(__name__)


class PointCloudViewer:
    """Interactive point cloud viewer with keyboard controls."""

    def __init__(self, pcd: o3d.geometry.PointCloud, frame_scale: float):
        """Initialize the viewer.

        Args:
            pcd: Point cloud to display.
            frame_scale: Scale for the coordinate frame.
        """
        self.pcd = pcd
        self.pcd_original = pcd  # Keep original for downsampling
        self.frame_scale = frame_scale
        self.show_frame = True
        self.show_axis_aligned_bbox = True
        self.show_oriented_bbox = True

        # Downsampling controls
        self.downsample_enabled = False
        self.voxel_size = 10.0  # 10mm default (data is in mm)
        self.voxel_size_step = 5.0  # 5mm increment

        # Create geometries
        self.axis_aligned_bbox = pcd.get_axis_aligned_bounding_box()
        self.axis_aligned_bbox.color = (1, 0, 0)
        self.oriented_bbox = pcd.get_minimal_oriented_bounding_box()
        self.oriented_bbox.color = (0, 1, 0)
        self.coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=frame_scale, origin=[0, 0, 0]
        )

        # Color the point cloud
        self.pcd.paint_uniform_color([1, 0.706, 0])  # yellow

        # Create visualizer
        self.vis = o3d.visualization.VisualizerWithKeyCallback()  # type: ignore[attr-defined]  # ty: ignore[possibly-missing-submodule]

    def toggle_frame(self, vis):
        """Toggle coordinate frame visibility."""
        self.show_frame = not self.show_frame
        if self.show_frame:
            vis.add_geometry(self.coordinate_frame, reset_bounding_box=False)
            logger.info("Coordinate frame: VISIBLE")
        else:
            vis.remove_geometry(self.coordinate_frame, reset_bounding_box=False)
            logger.info("Coordinate frame: HIDDEN")
        return False

    def toggle_axis_aligned_bbox(self, vis):
        """Toggle axis-aligned bounding box visibility."""
        self.show_axis_aligned_bbox = not self.show_axis_aligned_bbox
        if self.show_axis_aligned_bbox:
            vis.add_geometry(self.axis_aligned_bbox, reset_bounding_box=False)
            logger.info("Axis-aligned bounding box: VISIBLE")
        else:
            vis.remove_geometry(self.axis_aligned_bbox, reset_bounding_box=False)
            logger.info("Axis-aligned bounding box: HIDDEN")
        return False

    def toggle_oriented_bbox(self, vis):
        """Toggle oriented bounding box visibility."""
        self.show_oriented_bbox = not self.show_oriented_bbox
        if self.show_oriented_bbox:
            vis.add_geometry(self.oriented_bbox, reset_bounding_box=False)
            logger.info("Oriented bounding box: VISIBLE")
        else:
            vis.remove_geometry(self.oriented_bbox, reset_bounding_box=False)
            logger.info("Oriented bounding box: HIDDEN")
        return False

    def toggle_downsampling(self, vis):
        """Toggle point cloud downsampling."""
        self.downsample_enabled = not self.downsample_enabled
        self._update_downsampling()
        return False

    def increase_voxel_size(self, vis):
        """Increase voxel size (more sparse)."""
        if not self.downsample_enabled:
            logger.info("Downsampling not active (press D to enable)")
            return False

        self.voxel_size += self.voxel_size_step
        self._update_downsampling()
        return False

    def decrease_voxel_size(self, vis):
        """Decrease voxel size (more dense)."""
        if not self.downsample_enabled:
            logger.info("Downsampling not active (press D to enable)")
            return False

        # Minimum voxel size to avoid too much density (5mm)
        self.voxel_size = max(5.0, self.voxel_size - self.voxel_size_step)
        self._update_downsampling()
        return False

    def _update_downsampling(self):
        """Update point cloud with current downsampling settings."""
        # Remove current point cloud
        self.vis.remove_geometry(self.pcd, reset_bounding_box=False)

        # Apply downsampling or use original
        import copy

        if self.downsample_enabled:
            self.pcd = self.pcd_original.voxel_down_sample(voxel_size=self.voxel_size)
            # Re-apply color after downsampling
            self.pcd.paint_uniform_color([1, 0.706, 0])  # yellow
            num_points = len(self.pcd.points)
            num_original = len(self.pcd_original.points)
            logger.info(
                f"Downsampling: ON | Voxel: {self.voxel_size:.1f}mm | Points: {num_points}/{num_original}"
            )
        else:
            # Create a copy of the original
            self.pcd = copy.deepcopy(self.pcd_original)
            self.pcd.paint_uniform_color([1, 0.706, 0])  # yellow
            num_points = len(self.pcd.points)
            logger.info(f"Downsampling: OFF | Points: {num_points}")

        # Re-add point cloud
        self.vis.add_geometry(self.pcd, reset_bounding_box=False)

    def run(self):
        """Run the interactive visualizer."""
        # Create window
        self.vis.create_window(window_name="Point Cloud Viewer", width=1280, height=720)

        # Register key callbacks (will appear in Open3D's native help when pressing H)
        self.vis.register_key_callback(ord("F"), self.toggle_frame)
        self.vis.register_key_callback(ord("A"), self.toggle_axis_aligned_bbox)
        self.vis.register_key_callback(ord("B"), self.toggle_oriented_bbox)
        self.vis.register_key_callback(ord("D"), self.toggle_downsampling)
        self.vis.register_key_callback(ord("["), self.decrease_voxel_size)
        self.vis.register_key_callback(ord("]"), self.increase_voxel_size)

        # Add geometries
        self.vis.add_geometry(self.pcd)
        self.vis.add_geometry(self.axis_aligned_bbox)
        self.vis.add_geometry(self.oriented_bbox)
        self.vis.add_geometry(self.coordinate_frame)

        # Set view options
        self.vis.get_render_option().point_size = 2.0
        self.vis.get_render_option().background_color = [0.1, 0.1, 0.1]

        logger.info("Viewer started")
        logger.info("=" * 80)
        logger.info("KEYBOARD CONTROLS:")
        logger.info("  F:       Toggle coordinate frame")
        logger.info("  A:       Toggle axis-aligned bounding box (red)")
        logger.info("  B:       Toggle oriented bounding box (green)")
        logger.info("  D:       Toggle downsampling")
        logger.info("  [ / ]:   Decrease/Increase downsampling voxel size")
        logger.info("  +/-:     Increase/Decrease point size (native Open3D)")
        logger.info("  H:       Show native Open3D help (includes all controls)")
        logger.info("  Q/ESC:   Quit")
        logger.info("=" * 80)
        logger.info("Displaying:")
        logger.info("  - Axis-aligned bounding box in red")
        logger.info("  - Oriented bounding box in green")
        logger.info("  - Coordinate frame (toggle with F)")

        # Run visualizer
        self.vis.run()
        self.vis.destroy_window()


if __name__ == "__main__":
    # add input file argument
    parser = argparse.ArgumentParser(
        description="Load and display a point cloud with its bounding boxes"
    )
    parser.add_argument("--input", type=str, help="Input file path", required=True)
    parser.add_argument(
        "--flip",
        type=str,
        choices=["x", "y", "z", "nx", "ny", "nz"],
        help="Flip axis (x, y, or z) by 90 degrees (prefix n for negative)",
        required=False,
    )
    parser.add_argument(
        "--recompute-normals",
        action="store_true",
        help="Recompute point cloud normals after loading",
    )
    parser.add_argument(
        "--normal-radius",
        type=float,
        default=30.0,
        help="Radius for normal estimation in mm (default: 30.0)",
    )
    parser.add_argument(
        "--normal-max-nn",
        type=int,
        default=30,
        help="Maximum number of neighbors for normal estimation (default: 30)",
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

    # Recompute normals if requested
    if args.recompute_normals:
        logger.info(
            f"Recomputing normals (radius={args.normal_radius}mm, max_nn={args.normal_max_nn})"
        )
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=args.normal_radius, max_nn=args.normal_max_nn
            )
        )
        logger.info(f"Normals computed: {len(pcd.normals)} normals")

    frame_scale = rough_scale_point_cloud(pcd)

    if args.flip:
        pcd.transform(get_flip_transform(args.flip))

    # Create and run viewer
    viewer = PointCloudViewer(pcd, frame_scale)
    viewer.run()
