#!/usr/bin/env python3
"""Visualize scan acquisition sequence with animation.

This script displays an animated sequence of point cloud scans, showing:
- World coordinate frame (origin)
- LiDAR coordinate frame (moving with the sensor)
- LiDAR trajectory path (yellow line showing sensor movement)
- Current scan in the sequence
- Optional fused map overlay

Keyboard Controls:
    SPACE: Pause/Resume animation
    LEFT/RIGHT ARROW: Move to previous/next scan (when paused)
    F: Print current frame statistics
    T: Toggle trajectory visibility
    M: Toggle fused map visibility
    D: Toggle fused map downsampling on/off
    [ / ]: Decrease/Increase downsampling voxel size
    +/-: Increase/Decrease point size (native Open3D)
    Q/ESC: Quit

The animation loops continuously through the scan sequence.
"""

import sys
import time
import logging
from pathlib import Path
from typing import List, Tuple, Optional
import argparse

import numpy as np
import open3d as o3d

from registration.utils.logging import setup_logging

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from registration_common import (
    find_scan_pairs,
    load_transformation_matrix,
    load_poses_from_file,
)

logger = logging.getLogger(__name__)


# Animation parameters
DEFAULT_ANIMATION_SPEED = 1.0  # seconds per frame
COORDINATE_FRAME_SIZE = 500.0  # mm


def calculate_trajectory_length(points: np.ndarray) -> float:
    """Calculate total trajectory length from array of 3D points.

    Args:
        points: Nx3 numpy array of 3D points.

    Returns:
        Total length in mm (sum of distances between consecutive points).
    """
    if len(points) < 2:
        return 0.0

    distances = np.linalg.norm(np.diff(points, axis=0), axis=1)
    return float(np.sum(distances))


class SequenceVisualizer:
    """Visualizer for animating point cloud scan sequences.

    Displays each scan in sequence with world and LiDAR coordinate frames,
    with optional fused map overlay and keyboard controls for navigation.
    """

    def __init__(
        self,
        scan_pairs: List[Tuple[Path, Path]],
        fused_map_path: Optional[str] = None,
        animation_speed: float = DEFAULT_ANIMATION_SPEED,
        start_scan: Optional[int] = None,
        end_scan: Optional[int] = None,
        step: int = 1,
        poses_file: Optional[str] = None,
    ):
        """Initialize the sequence visualizer.

        Args:
            scan_pairs: List of (ply_path, json_path) tuples.
            fused_map_path: Optional path to fused map PLY file.
            animation_speed: Time in seconds between frames.
            start_scan: Index of first scan (0-based, inclusive).
            end_scan: Index of last scan (0-based, inclusive). None = last scan.
            step: Process every nth scan within the selected range (default: 1 = all scans).
            poses_file: Optional path to JSON file containing all poses (like optimized_poses.json).
        """
        self.scan_pairs = scan_pairs
        self.animation_speed = animation_speed

        # Apply scan range
        self.start_idx = start_scan if start_scan is not None else 0
        self.end_idx = end_scan if end_scan is not None else len(scan_pairs) - 1
        if step < 1:
            raise ValueError(f"step must be >= 1, got {step}")
        self._validate_scan_range()

        self.scan_pairs = self.scan_pairs[self.start_idx : self.end_idx + 1 : step]
        self.num_scans = len(self.scan_pairs)

        # Animation state
        self.current_scan_idx = 0
        self.is_playing = True
        self.last_update_time = time.time()

        # Visualization objects
        self.vis = None
        self.current_scan_geometry = None
        self.lidar_frame_geometry = None
        self.world_frame_geometry = None
        self.fused_map_geometry = None
        self.fused_map_geometry_original = None  # Original before downsampling
        self.trajectory_geometry = None
        self.show_fused_map = True
        self.show_trajectory = True

        # Fused map downsampling controls
        self.downsample_enabled = False
        self.voxel_size = 10.0  # 10mm default (data is in mm)
        self.voxel_size_step = 5.0  # 5mm increment

        # Load poses from file if provided
        self.poses = None
        if poses_file:
            self.poses = load_poses_from_file(poses_file, self.num_scans)
            if self.poses is None:
                logger.warning(
                    "Failed to load poses file, falling back to individual JSON files"
                )

        # Load all transformations to build trajectory
        self._load_trajectory()

        # Load fused map if provided
        if fused_map_path:
            self._load_fused_map(fused_map_path)

        logger.info(f"Initialized visualizer with {self.num_scans} scans")
        logger.info(f"Scan range: {self.start_idx} to {self.end_idx} (step={step})")
        logger.info(f"Animation speed: {self.animation_speed}s per frame")

    def _validate_scan_range(self):
        """Validate scan range parameters."""
        total_scans = len(self.scan_pairs)

        if self.start_idx < 0 or self.start_idx >= total_scans:
            raise ValueError(
                f"start_scan ({self.start_idx}) must be in range [0, {total_scans - 1}]"
            )
        if self.end_idx < 0 or self.end_idx >= total_scans:
            raise ValueError(
                f"end_scan ({self.end_idx}) must be in range [0, {total_scans - 1}]"
            )
        if self.start_idx > self.end_idx:
            raise ValueError(
                f"start_scan ({self.start_idx}) must be <= end_scan ({self.end_idx})"
            )

    def _load_trajectory(self):
        """Load all transformations and create trajectory line."""
        trajectory_points = []

        for scan_idx in range(self.num_scans):
            try:
                # Use loaded poses if available, otherwise load from individual JSON files
                if self.poses is not None:
                    if scan_idx < len(self.poses):
                        H = self.poses[scan_idx]
                    else:
                        logger.warning(
                            f"Pose index {scan_idx} out of range in poses file"
                        )
                        continue
                else:
                    _, json_path = self.scan_pairs[scan_idx]
                    H = load_transformation_matrix(json_path)

                # Extract translation (position) from transformation matrix
                position = H[:3, 3]
                trajectory_points.append(position)
            except Exception as e:
                logger.warning(
                    f"Could not load transformation for scan {scan_idx}: {e}"
                )

        if len(trajectory_points) < 2:
            logger.warning("Not enough points to create trajectory")
            return

        # Create LineSet for trajectory
        trajectory_points_array = np.array(trajectory_points)
        lines = [[i, i + 1] for i in range(len(trajectory_points) - 1)]

        self.trajectory_geometry = o3d.geometry.LineSet()
        self.trajectory_geometry.points = o3d.utility.Vector3dVector(
            trajectory_points_array
        )
        self.trajectory_geometry.lines = o3d.utility.Vector2iVector(lines)

        # Set trajectory color to yellow for visibility
        colors = [[1, 1, 0] for _ in range(len(lines))]  # Yellow
        self.trajectory_geometry.colors = o3d.utility.Vector3dVector(colors)

        # Calculate and log trajectory length
        trajectory_length = calculate_trajectory_length(trajectory_points_array)
        logger.info(f"Created trajectory with {len(trajectory_points)} points")
        logger.info(f"Trajectory length: {trajectory_length:.2f} mm")

    def _load_fused_map(self, fused_map_path: str):
        """Load the fused map from file."""
        map_path = Path(fused_map_path)
        if not map_path.exists():
            logger.warning(f"Fused map file not found: {fused_map_path}")
            return

        logger.info(f"Loading fused map from: {fused_map_path}")
        self.fused_map_geometry_original = o3d.io.read_point_cloud(str(map_path))

        if not self.fused_map_geometry_original.has_points():
            logger.warning("Fused map has no points")
            self.fused_map_geometry_original = None
            return

        num_points = len(self.fused_map_geometry_original.points)

        # Keep original pristine (no color modifications) for proper downsampling
        # Create a copy for initial display
        import copy

        self.fused_map_geometry = copy.deepcopy(self.fused_map_geometry_original)
        self.fused_map_geometry.paint_uniform_color([0.3, 0.3, 0.3])  # Dim gray

        logger.info(
            f"Loaded fused map with {num_points} points (downsampling: OFF, use D to toggle)"
        )

    def _create_coordinate_frame(self, size: float = COORDINATE_FRAME_SIZE):
        """Create a coordinate frame mesh.

        Args:
            size: Size of the coordinate frame axes in mm.

        Returns:
            TriangleMesh representing the coordinate frame.
        """
        return o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)

    def _load_scan_at_index(self, scan_idx: int):
        """Load and transform scan at given index.

        Args:
            scan_idx: Index of scan to load.

        Returns:
            Tuple of (transformed_scan, H_transform) or (None, None) on error.
        """
        if scan_idx < 0 or scan_idx >= self.num_scans:
            logger.error(f"Invalid scan index: {scan_idx}")
            return None, None

        ply_path, json_path = self.scan_pairs[scan_idx]

        try:
            # Load point cloud
            scan = o3d.io.read_point_cloud(str(ply_path))
            if not scan.has_points():
                logger.warning(f"Scan {scan_idx} has no points: {ply_path}")
                return None, None

            # Load transformation from poses file or individual JSON file
            if self.poses is not None:
                if scan_idx < len(self.poses):
                    H = self.poses[scan_idx]
                else:
                    logger.error(f"Pose index {scan_idx} out of range in poses file")
                    return None, None
            else:
                H = load_transformation_matrix(json_path)

            # Transform scan to world frame
            scan.transform(H)

            return scan, H

        except Exception as e:
            logger.error(f"Error loading scan {scan_idx}: {e}")
            return None, None

    def _update_visualization(self):
        """Update visualization with current scan."""
        if self.vis is None:
            logger.warning("Visualizer not initialized yet")
            return  # Early return if not initialized

        # Load current scan
        scan, H = self._load_scan_at_index(self.current_scan_idx)

        if scan is None:
            logger.warning(f"Could not load scan {self.current_scan_idx}, skipping")
            return

        # Remove previous scan if exists
        if self.current_scan_geometry is not None:
            self.vis.remove_geometry(
                self.current_scan_geometry, reset_bounding_box=False
            )

        # Remove previous LiDAR frame if exists
        if self.lidar_frame_geometry is not None:
            self.vis.remove_geometry(
                self.lidar_frame_geometry, reset_bounding_box=False
            )

        # Add current scan with bright color to stand out from fused map
        self.current_scan_geometry = scan
        # Paint bright cyan to make it clearly visible over the dim fused map
        self.current_scan_geometry.paint_uniform_color([0.0, 0.9, 1.0])  # Bright cyan
        self.vis.add_geometry(self.current_scan_geometry, reset_bounding_box=False)

        # Create and add LiDAR frame at current pose
        self.lidar_frame_geometry = self._create_coordinate_frame()
        self.lidar_frame_geometry.transform(H)
        self.vis.add_geometry(self.lidar_frame_geometry, reset_bounding_box=False)

    def _on_key_space(self, vis):
        """Callback for SPACE key - toggle play/pause."""
        self.is_playing = not self.is_playing
        status = "PLAYING" if self.is_playing else "PAUSED"
        logger.info(f"Animation {status}")
        return False

    def _on_key_left(self, vis):
        """Callback for LEFT arrow - previous scan."""
        if not self.is_playing:
            self.current_scan_idx = (self.current_scan_idx - 1) % self.num_scans
            self._update_visualization()
        return False

    def _on_key_right(self, vis):
        """Callback for RIGHT arrow - next scan."""
        if not self.is_playing:
            self.current_scan_idx = (self.current_scan_idx + 1) % self.num_scans
            self._update_visualization()
        return False

    def _on_key_m(self, vis):
        """Callback for M key - toggle fused map visibility."""
        if self.fused_map_geometry is None:
            logger.info("No fused map loaded")
            return False

        if self.vis is None:
            logger.warning("Visualizer not initialized yet")
            return  # Early return if not initialized

        self.show_fused_map = not self.show_fused_map

        if self.show_fused_map:
            self.vis.add_geometry(self.fused_map_geometry, reset_bounding_box=False)
            logger.info("Fused map: VISIBLE")
        else:
            self.vis.remove_geometry(self.fused_map_geometry, reset_bounding_box=False)
            logger.info("Fused map: HIDDEN")

        return False

    def _on_key_t(self, vis):
        """Callback for T key - toggle trajectory visibility."""
        if self.trajectory_geometry is None:
            logger.info("No trajectory available")
            return False

        if self.vis is None:
            logger.warning("Visualizer not initialized yet")
            return  # Early return if not initialized

        self.show_trajectory = not self.show_trajectory

        if self.show_trajectory:
            self.vis.add_geometry(self.trajectory_geometry, reset_bounding_box=False)
            logger.info("Trajectory: VISIBLE")
        else:
            self.vis.remove_geometry(self.trajectory_geometry, reset_bounding_box=False)
            logger.info("Trajectory: HIDDEN")

        return False

    def _on_key_d(self, vis):
        """Callback for D key - toggle fused map downsampling."""
        if self.fused_map_geometry_original is None:
            logger.info("No fused map loaded")
            return False

        self.downsample_enabled = not self.downsample_enabled
        self._update_fused_map_downsampling()

        return False

    def _on_key_left_bracket(self, vis):
        """Callback for [ key - decrease voxel size (more dense)."""
        if self.fused_map_geometry_original is None or not self.downsample_enabled:
            logger.info("Downsampling not active")
            return False

        # Minimum voxel size to avoid too much density (5mm)
        self.voxel_size = max(5.0, self.voxel_size - self.voxel_size_step)
        self._update_fused_map_downsampling()

        return False

    def _on_key_right_bracket(self, vis):
        """Callback for ] key - increase voxel size (more sparse)."""
        if self.fused_map_geometry_original is None or not self.downsample_enabled:
            logger.info("Downsampling not active")
            return False

        self.voxel_size += self.voxel_size_step
        self._update_fused_map_downsampling()

        return False

    def _on_key_f(self, vis):
        """Callback for F key - print current frame statistics."""
        actual_scan_num = self.start_idx + self.current_scan_idx
        total_displayed = self.end_idx - self.start_idx + 1
        status = "PLAYING" if self.is_playing else "PAUSED"
        logger.info(
            f">>> Scan {actual_scan_num} (frame {self.current_scan_idx + 1}/{total_displayed}) - {status}"
        )
        return False

    def _update_fused_map_downsampling(self):
        """Update fused map with current downsampling settings.

        Removes the current map, applies downsampling if enabled, and re-adds it.
        """
        if self.fused_map_geometry_original is None:
            return

        if self.vis is None:
            logger.warning("Visualizer not initialized yet")
            return  # Early return if not initialized

        # Remove current map if visible
        was_visible = self.show_fused_map
        if was_visible and self.fused_map_geometry is not None:
            self.vis.remove_geometry(self.fused_map_geometry, reset_bounding_box=False)

        # Apply downsampling or use original (always create a fresh copy)
        import copy

        if self.downsample_enabled:
            # Always start from the pristine original
            self.fused_map_geometry = (
                self.fused_map_geometry_original.voxel_down_sample(
                    voxel_size=self.voxel_size
                )
            )
            # Apply color after downsampling
            self.fused_map_geometry.paint_uniform_color([0.3, 0.3, 0.3])
            num_points = len(self.fused_map_geometry.points)
            num_original = len(self.fused_map_geometry_original.points)
            logger.info(
                f"Downsampling: ON | Voxel: {self.voxel_size:.1f}mm | Points: {num_points}/{num_original}"
            )
        else:
            # Create a copy of the original and paint it
            self.fused_map_geometry = copy.deepcopy(self.fused_map_geometry_original)
            self.fused_map_geometry.paint_uniform_color([0.3, 0.3, 0.3])
            num_points = len(self.fused_map_geometry.points)
            logger.info(f"Downsampling: OFF | Points: {num_points}")

        # Re-add if it was visible
        if was_visible:
            self.vis.add_geometry(self.fused_map_geometry, reset_bounding_box=False)

    def _animation_callback(self, vis):
        """Animation callback for automatic playback.

        Called periodically to advance to next scan when playing.
        """
        current_time = time.time()

        if self.is_playing:
            # Check if enough time has passed for next frame
            if current_time - self.last_update_time >= self.animation_speed:
                self.current_scan_idx = (self.current_scan_idx + 1) % self.num_scans
                self._update_visualization()
                self.last_update_time = current_time

        return False

    def run(self):
        """Run the interactive visualization."""
        # Create visualizer
        self.vis = o3d.visualization.VisualizerWithKeyCallback()  # type: ignore[attr-defined]
        self.vis.create_window(
            window_name="Scan Sequence Viewer",
            width=1280,
            height=720,
        )

        # Set dark background color (dark gray/black)
        render_option = self.vis.get_render_option()
        render_option.background_color = np.array([0.15, 0.15, 0.15])  # Very dark gray
        render_option.point_size = 1.5  # Smaller point size for better detail

        # Register keyboard callbacks
        self.vis.register_key_callback(ord(" "), self._on_key_space)  # SPACE
        self.vis.register_key_callback(262, self._on_key_right)  # RIGHT ARROW
        self.vis.register_key_callback(263, self._on_key_left)  # LEFT ARROW
        self.vis.register_key_callback(ord("M"), self._on_key_m)  # M key
        self.vis.register_key_callback(ord("T"), self._on_key_t)  # T key
        self.vis.register_key_callback(ord("D"), self._on_key_d)  # D key
        self.vis.register_key_callback(ord("["), self._on_key_left_bracket)  # [ key
        self.vis.register_key_callback(ord("]"), self._on_key_right_bracket)  # ] key
        self.vis.register_key_callback(ord("F"), self._on_key_f)  # F key

        # Register animation callback
        self.vis.register_animation_callback(self._animation_callback)

        # Create and add world coordinate frame (origin)
        self.world_frame_geometry = self._create_coordinate_frame()
        self.vis.add_geometry(self.world_frame_geometry)

        # Add trajectory if available and visible by default
        if self.trajectory_geometry is not None and self.show_trajectory:
            self.vis.add_geometry(self.trajectory_geometry)
            logger.info("Trajectory visible (press T to toggle)")

        # Add fused map if loaded and visible by default
        if self.fused_map_geometry is not None and self.show_fused_map:
            self.vis.add_geometry(self.fused_map_geometry)
            logger.info("Fused map visible (press M to toggle)")

        # Load and display first scan
        self._update_visualization()

        # Compute bounding box for proper camera setup
        all_geometries = []
        if self.world_frame_geometry is not None:
            all_geometries.append(self.world_frame_geometry)
        if self.trajectory_geometry is not None:
            all_geometries.append(self.trajectory_geometry)
        if self.fused_map_geometry is not None:
            all_geometries.append(self.fused_map_geometry)

        # Reset view to fit all geometry
        if all_geometries:
            self.vis.reset_view_point(True)

        # Show instructions
        logger.info("=" * 80)
        logger.info("KEYBOARD CONTROLS:")
        logger.info("  SPACE:       Pause/Resume animation")
        logger.info("  LEFT/RIGHT:  Previous/Next scan (when paused)")
        logger.info("  F:           Print current frame statistics")
        logger.info("  T:           Toggle trajectory visibility")
        logger.info("  M:           Toggle fused map visibility")
        logger.info("  D:           Toggle fused map downsampling")
        logger.info("  [ / ]:       Decrease/Increase downsampling voxel size")
        logger.info("  +/-:         Increase/Decrease point size (native Open3D)")
        logger.info("  Q/ESC:       Quit")
        logger.info("=" * 80)

        # Run visualization loop
        self.vis.run()
        self.vis.destroy_window()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Visualize scan acquisition sequence with animation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Input directory containing .ply and .json scan pairs",
    )

    parser.add_argument(
        "--fused-map",
        "-m",
        default=None,
        help="Optional path to fused map PLY file for overlay",
    )

    parser.add_argument(
        "--speed",
        "-s",
        type=float,
        default=DEFAULT_ANIMATION_SPEED,
        help="Animation speed in seconds per frame",
    )

    parser.add_argument(
        "--start-scan",
        type=int,
        default=None,
        help="Index of first scan to display (0-based, inclusive)",
    )

    parser.add_argument(
        "--end-scan",
        type=int,
        default=None,
        help="Index of last scan to display (0-based, inclusive)",
    )

    parser.add_argument(
        "--step",
        type=int,
        default=1,
        help="Process every nth scan within the selected range (1=consecutive, 2=every other scan, etc.)",
    )

    parser.add_argument(
        "--poses",
        "-p",
        default=None,
        help="Optional path to JSON file containing all poses (like optimized_poses.json from multiway_registration.py). If provided, poses from this file are used instead of individual ground truth JSON files.",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging()

    data_path = Path(args.input)

    # Validate input directory
    if not data_path.exists():
        logger.error(f"Input directory not found: {args.input}")
        sys.exit(1)

    # Find all scan pairs
    logger.info(f"Searching for scan pairs in: {data_path}")
    pairs = find_scan_pairs(data_path)

    if not pairs:
        logger.error(f"No matching .ply/.json pairs found in {args.input}")
        sys.exit(1)

    logger.info(f"Found {len(pairs)} scan pairs")

    try:
        # Create and run visualizer
        visualizer = SequenceVisualizer(
            scan_pairs=pairs,
            fused_map_path=args.fused_map,
            animation_speed=args.speed,
            start_scan=args.start_scan,
            end_scan=args.end_scan,
            step=args.step,
            poses_file=args.poses,
        )
        visualizer.run()

    except Exception as e:
        logger.error(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()
