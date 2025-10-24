import open3d as o3d
import copy
import logging
import sys


# Custom formatter with colored log levels
class ColoredFormatter(logging.Formatter):
    """Custom formatter that adds color to log level names only."""
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record):
        # Save the original levelname
        levelname = record.levelname
        
        # Add color to levelname only
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{levelname}{self.RESET}"
        
        # Format the message
        result = super().format(record)
        
        # Restore original levelname
        record.levelname = levelname
        
        return result


def setup_logging(level=logging.INFO):
    """Set up logging configuration with colored output.
    
    Args:
        level: The logging level (default: logging.INFO).
    
    Returns:
        The root logger instance.
    """
    # Create formatter with timestamp and level
    formatter = ColoredFormatter(
        fmt='[%(asctime)s][%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Set up console handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.handlers.clear()  # Remove any existing handlers
    root_logger.addHandler(handler)
    
    return root_logger


# Initialize logging when module is imported
setup_logging()


def draw_registration_result(source, target, transformation, window_name : str):
    """Visualize the registration result between source and target point clouds.

    Args:
        source: Source point cloud to be transformed.
        target: Target point cloud (reference).
        transformation: 4x4 transformation matrix to apply to source.
        window_name: Name of the visualization window.
    """
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp], window_name=window_name)
                                      # zoom=0.4459,
                                      # front=[0.9288, -0.2951, -0.2242],
                                      # lookat=[1.6784, 2.0612, 1.4451],
                                      # up=[-0.3402, -0.9189, -0.1996])

# print stats about a point cloud
def print_point_cloud_info(pcd: o3d.geometry.PointCloud, name: str):
    """Prints information about a point cloud including number of points and bounding box.

    Args:
        pcd: The point cloud to analyze.
        name: Name of the point cloud for identification in the output.
    """
    num_points = len(pcd.points)
    aabb = pcd.get_axis_aligned_bounding_box()
    obb = pcd.get_oriented_bounding_box()
    logging.info(f"Point Cloud '{name}':")
    logging.info(f"  Number of points: {num_points}")
    logging.info(f"  Axis-Aligned Bounding Box: min {aabb.min_bound}, max {aabb.max_bound}")
    logging.info(f"  Oriented Bounding Box: center {obb.center}, extent {obb.extent}")