"""Annotate point clouds with semantic labels."""

import json
import logging
import shutil
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Optional

import open3d as o3d

from registration.utils.logging import setup_logging

logger = logging.getLogger(__name__)


def load_point_cloud(ply_path: str) -> o3d.geometry.PointCloud:
    """Load a point cloud from a PLY file.

    Args:
        ply_path: Path to input .ply file.

    Returns:
        The loaded Open3D point cloud.
    """
    pcd = o3d.io.read_point_cloud(ply_path)
    if not pcd.has_points():
        raise ValueError("Point cloud is empty.")
    return pcd


def fit_plane(pcd: o3d.geometry.PointCloud) -> Tuple[np.ndarray, list]:
    """Fit a plane to the cloud using RANSAC.

    Args:
        pcd: Input point cloud.

    Returns:
        plane_params: Plane equation parameters [a, b, c, d].
        inliers: List of inlier indices for the plane.
    """
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=20, ransac_n=3, num_iterations=2000
    )
    a, b, c, d = plane_model
    return np.array([a, b, c, d]), inliers


def fit_cylinder(pcd: o3d.geometry.PointCloud) -> Tuple[Dict, list]:
    """Fit a cylinder using RANSAC.

    Args:
        pcd: Input point cloud.

    Returns:
        params: Cylinder dictionary: {"center": [..], "axis": [..], "radius": r}
        inliers: List of inlier indices for the cylinder.
    """
    # Convert to numpy
    pts = np.asarray(pcd.points)

    # Estimate axis via PCA
    pts_mean = pts.mean(axis=0)
    _, _, vh = np.linalg.svd(pts - pts_mean)
    axis = vh[0]

    # Distance from axis gives radius
    proj = (pts - pts_mean) @ axis
    closest = pts_mean + np.outer(proj, axis)
    radial = np.linalg.norm(pts - closest, axis=1)
    radius = np.median(radial)

    # Inliers: close to radius
    inliers = np.where(np.abs(radial - radius) < 170)[0].tolist()

    return {
        "center": pts_mean.tolist(),
        "axis": axis.tolist(),
        "radius": float(radius),
    }, inliers


def compute_obb(pcd: o3d.geometry.PointCloud) -> Dict:
    """Compute an oriented bounding box for a point cloud.

    Args:
        pcd: Input point cloud.

    Returns:
        obb: Dict containing center, axes, and extents.
    """
    try:
        obb = pcd.get_oriented_bounding_box()
        return {
            "center": obb.center.tolist(),
            "axes": np.asarray(obb.R).tolist(),
            "extents": obb.extent.tolist(),
        }
    except RuntimeError as e:
        if "qhull" in str(e).lower() or "simplex" in str(e).lower():
            # Fallback: data might be degenerate, use axis-aligned bbox
            aabb = pcd.get_axis_aligned_bounding_box()
            center = aabb.get_center()
            extents = aabb.get_extent()
            return {
                "center": center.tolist(),
                "axes": np.eye(3).tolist(),
                "extents": extents.tolist(),
            }
        else:
            raise


def compute_planar_obb(
    pcd: o3d.geometry.PointCloud, plane_normal: np.ndarray, margin: float = 25
) -> Dict:
    """Compute an oriented bounding box for a planar point cloud.

    For planar point clouds (where all points are coplanar), computes a 2D
    bounding box in the plane and adds a small margin in the normal direction
    to avoid qhull errors with perfectly flat data.

    Args:
        pcd: Input point cloud.
        plane_normal: Normal vector of the plane [a, b, c] from plane equation.
        margin: Margin to add in normal direction (default: 0.01).

    Returns:
        obb: Dict containing center, axes, and extents.
    """
    pts = np.asarray(pcd.points)

    # Normalize the plane normal
    normal = plane_normal / np.linalg.norm(plane_normal)

    # Create coordinate system in the plane
    # Find two orthogonal vectors in the plane
    if np.abs(normal[2]) < 0.9:
        u = np.cross(normal, np.array([0, 0, 1]))
    else:
        u = np.cross(normal, np.array([0, 1, 0]))
    u = u / np.linalg.norm(u)
    v = np.cross(normal, u)
    v = v / np.linalg.norm(v)
    logger.debug(f"Plane basis vectors u: {u}, v: {v}")

    # Compute centroid
    center = pts.mean(axis=0)
    # logger.debug(f"Centroid of points: {center}")
    # logger.debug(f"Original points:\n{pts}")

    # Project points onto plane coordinate system
    pts_centered = pts - center
    # logger.debug(f"Centered points:\n{pts_centered}")
    u_coords = pts_centered @ u
    v_coords = pts_centered @ v
    # logger.debug(f"Projected coordinates u: {u_coords}")

    # Compute 2D bounding box
    u_min, u_max = u_coords.min(), u_coords.max()
    v_min, v_max = v_coords.min(), v_coords.max()

    # Extents in each direction
    u_extent = u_max - u_min
    v_extent = v_max - v_min

    # Add small margin in normal direction
    normal_extent = 2 * margin

    # Adjust center to be at the middle of the bounding box
    u_center = (u_min + u_max) / 2
    v_center = (v_min + v_max) / 2
    center = center + u_center * u + v_center * v

    # Build rotation matrix (axes as columns)
    R = np.column_stack([u, v, normal])

    return {
        "center": center.tolist(),
        "axes": R.T.tolist(),
        "extents": [u_extent, v_extent, normal_extent],
    }


def load_annotated_ply(
    ply_path: str,
) -> Tuple[Optional[o3d.geometry.PointCloud], Optional[Dict[str, np.ndarray]]]:
    """Load an existing annotated PLY file with custom properties.

    Args:
        ply_path: Path to annotated PLY file.

    Returns:
        Tuple of (point cloud, annotations dict) or (None, None) if file doesn't exist.
    """
    if not Path(ply_path).exists():
        return None, None

    # Read the PLY file manually to extract custom properties
    with open(ply_path, "r") as f:
        lines = f.readlines()

    # Parse header
    header_end = 0
    n_points = 0
    properties = []

    for i, line in enumerate(lines):
        if line.startswith("element vertex"):
            n_points = int(line.split()[-1])
        elif line.startswith("property"):
            parts = line.split()
            prop_type = parts[1]
            prop_name = parts[2]
            properties.append((prop_name, prop_type))
        elif line.strip() == "end_header":
            header_end = i + 1
            break

    # Read data
    data_lines = lines[header_end : header_end + n_points]

    # Parse points
    points = []
    normals = []
    colors = []
    semantic_class = []
    semantic_instance = []
    geom_id = []

    has_normals = any(name == "nx" for name, _ in properties)
    has_colors = any(name == "red" for name, _ in properties)
    has_annotations = any(name == "semantic_class" for name, _ in properties)

    for line in data_lines:
        values = line.split()
        idx = 0

        # x, y, z
        points.append(
            [float(values[idx]), float(values[idx + 1]), float(values[idx + 2])]
        )
        idx += 3

        # normals
        if has_normals:
            normals.append(
                [float(values[idx]), float(values[idx + 1]), float(values[idx + 2])]
            )
            idx += 3

        # colors
        if has_colors:
            colors.append(
                [
                    int(values[idx]) / 255.0,
                    int(values[idx + 1]) / 255.0,
                    int(values[idx + 2]) / 255.0,
                ]
            )
            idx += 3

        # annotations
        if has_annotations:
            semantic_class.append(int(values[idx]))
            semantic_instance.append(int(values[idx + 1]))
            geom_id.append(int(values[idx + 2]))

    # Create point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(points))

    if has_normals and normals:
        pcd.normals = o3d.utility.Vector3dVector(np.array(normals))

    if has_colors and colors:
        pcd.colors = o3d.utility.Vector3dVector(np.array(colors))

    annotations = None
    if has_annotations:
        annotations = {
            "semantic_class": np.array(semantic_class, dtype=np.uint8),
            "semantic_instance": np.array(semantic_instance, dtype=np.uint16),
            "geom_id": np.array(geom_id, dtype=np.uint16),
        }

    return pcd, annotations


def merge_point_clouds_with_annotations(
    existing_pcd: o3d.geometry.PointCloud,
    existing_annotations: Dict[str, np.ndarray],
    new_pcd: o3d.geometry.PointCloud,
    new_annotations: Dict[str, np.ndarray],
) -> Tuple[o3d.geometry.PointCloud, Dict[str, np.ndarray]]:
    """Merge two point clouds with their annotations.

    Args:
        existing_pcd: Existing point cloud.
        existing_annotations: Existing annotations.
        new_pcd: New point cloud to add.
        new_annotations: New annotations to add.

    Returns:
        Tuple of (merged point cloud, merged annotations).
    """
    # Merge point clouds
    merged_pcd = existing_pcd + new_pcd

    # Merge annotations
    merged_annotations = {
        "semantic_class": np.concatenate(
            [existing_annotations["semantic_class"], new_annotations["semantic_class"]]
        ),
        "semantic_instance": np.concatenate(
            [
                existing_annotations["semantic_instance"],
                new_annotations["semantic_instance"],
            ]
        ),
        "geom_id": np.concatenate(
            [existing_annotations["geom_id"], new_annotations["geom_id"]]
        ),
    }

    return merged_pcd, merged_annotations


def add_annotations(
    pcd: o3d.geometry.PointCloud,
    inliers: list,
    geom_id: int,
    class_id: int,
    instance_id: int,
) -> Tuple[o3d.geometry.PointCloud, Dict[str, np.ndarray]]:
    """Prepare annotation attributes for a point cloud.

    Args:
        pcd: Input point cloud.
        inliers: List of inlier indices belonging to a primitive.
        geom_id: Primitive ID.
        class_id: Semantic Class ID.
        instance_id: Semantic Instance ID.

    Returns:
        A tuple containing:
            - The original point cloud (unchanged)
            - A dictionary of annotation arrays: {"semantic_class": array, "semantic_instance": array, "geom_id": array}
    """
    n = len(pcd.points)

    geom = np.zeros(n, dtype=np.uint16)
    sem_class = np.zeros(n, dtype=np.uint8)
    sem_inst = np.zeros(n, dtype=np.uint16)

    geom[inliers] = geom_id
    sem_class[inliers] = class_id
    sem_inst[inliers] = instance_id

    annotations = {
        "semantic_class": sem_class,
        "semantic_instance": sem_inst,
        "geom_id": geom,
    }

    return pcd, annotations


def save_ply_with_annotations(
    pcd: o3d.geometry.PointCloud, annotations: Dict[str, np.ndarray], out_path: str
):
    """Save a PLY file with custom annotation properties.

    Writes a PLY file with the point cloud data plus custom annotation fields.
    The PLY format preserves all standard properties (position, color, normal)
    and adds the annotation fields.

    Args:
        pcd: Input point cloud.
        annotations: Dictionary mapping property names to numpy arrays.
        out_path: Output file path ending in .ply
    """
    points = np.asarray(pcd.points)
    n_points = len(points)

    # Check for optional attributes
    has_colors = pcd.has_colors()
    has_normals = pcd.has_normals()

    colors = np.asarray(pcd.colors) if has_colors else None
    normals = np.asarray(pcd.normals) if has_normals else None

    # Write PLY file manually
    with open(out_path, "w") as f:
        # Header
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {n_points}\n")

        # Standard properties
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")

        if has_normals:
            f.write("property float nx\n")
            f.write("property float ny\n")
            f.write("property float nz\n")

        if has_colors:
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")

        # Custom annotation properties
        f.write("property uchar semantic_class\n")
        f.write("property ushort semantic_instance\n")
        f.write("property ushort geom_id\n")

        f.write("end_header\n")

        # Data
        for i in range(n_points):
            # Position
            f.write(f"{points[i, 0]} {points[i, 1]} {points[i, 2]}")

            # Normal
            if has_normals and normals is not None:
                f.write(f" {normals[i, 0]} {normals[i, 1]} {normals[i, 2]}")

            # Color
            if has_colors and colors is not None:
                r = int(colors[i, 0] * 255)
                g = int(colors[i, 1] * 255)
                b = int(colors[i, 2] * 255)
                f.write(f" {r} {g} {b}")

            # Annotations
            f.write(f" {annotations['semantic_class'][i]}")
            f.write(f" {annotations['semantic_instance'][i]}")
            f.write(f" {annotations['geom_id'][i]}")
            f.write("\n")


def load_primitives(json_path: str) -> list:
    """Load existing primitives from JSON file.

    Args:
        json_path: Path to the primitives JSON file.

    Returns:
        List of existing primitives, or empty list if file doesn't exist.
    """
    if Path(json_path).exists():
        with open(json_path, "r") as f:
            data = json.load(f)
            return data.get("primitives", [])
    return []


def save_primitives(primitives: list, out_path: str):
    """Save primitives list to JSON file with schema reference.

    Args:
        primitives: List of primitive dictionaries.
        out_path: Output path.
    """
    data = {"$schema": "./primitives_schema.json", "primitives": primitives}
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)

    # Copy schema file to output directory
    schema_src = Path(__file__).parent / "primitives_schema.json"
    schema_dst = Path(out_path).parent / "primitives_schema.json"
    if schema_src.exists():
        shutil.copy2(schema_src, schema_dst)


def load_semantic_classes(json_path: str) -> Dict[str, str]:
    """Load semantic classes from JSON file.

    Args:
        json_path: Path to the semantic classes JSON file.

    Returns:
        Dictionary mapping class IDs to class names.
    """
    if Path(json_path).exists():
        with open(json_path, "r") as f:
            data = json.load(f)
            return data.get("semantic_classes", {})
    return {}


def load_semantic_instances(json_path: str) -> list:
    """Load existing semantic instances from JSON file.

    Args:
        json_path: Path to the semantic JSON file.

    Returns:
        List of existing semantic instances.
    """
    if Path(json_path).exists():
        with open(json_path, "r") as f:
            data = json.load(f)
            return data.get("semantic_instances", [])
    return []


def save_semantic(
    semantic_classes: Dict[str, str], semantic_instances: list, out_path: str
):
    """Save semantic information to JSON file.

    Args:
        semantic_classes: Dictionary mapping class IDs to names.
        semantic_instances: List of semantic instance dictionaries.
        out_path: Output path.
    """
    data = {
        "semantic_classes": semantic_classes,
        "semantic_instances": semantic_instances,
    }
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)


def save_json(data: Dict, out_path: str):
    """Save JSON dictionary to file.

    Args:
        data: Dictionary to save.
        out_path: Output path.
    """
    with open(out_path, "w") as f:
        json.dump(data, f, indent=4)


def annotate(
    ply_path: str,
    primitive_type: str,
    geom_id: int,
    class_id: int,
    instance_id: int,
    out_dir: str,
    semantic_classes_path: Optional[str] = None,
):
    """Main annotation pipeline.

    Args:
        ply_path: Path to input PLY.
        primitive_type: "plane" or "cylinder".
        geom_id: Primitive ID to assign in output.
        class_id: Semantic class ID.
        instance_id: Instance ID for that class.
        out_dir: Output folder.
        semantic_classes_path: Path to semantic classes JSON file.
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    pcd = load_point_cloud(ply_path)

    if primitive_type == "plane":
        params, inliers = fit_plane(pcd)
        # Extract fitted points
        inlier_cloud = pcd.select_by_index(inliers)
        # For planes, use 2D bounding box with margin
        plane_normal = params[:3]  # [a, b, c] from plane equation
        logger.info(f"Fitting plane with normal: {plane_normal}, d: {params[3]}")
        logger.info(f"Inlier ratio: {len(inliers)} / {len(pcd.points)}")
        obb = compute_planar_obb(inlier_cloud, plane_normal)

        print(f"obb: {obb}")

        primitive = {
            "id": geom_id,
            "type": "plane",
            "equation": params.tolist(),
            "obb": obb,
        }
    elif primitive_type == "cylinder":
        params, inliers = fit_cylinder(pcd)
        # Extract fitted points
        inlier_cloud = pcd.select_by_index(inliers)
        logger.info(
            f"Fitting cylinder with inlier ratio: {len(inliers)} / {len(pcd.points)}"
        )
        # For cylinders, use standard 3D OBB
        obb = compute_obb(inlier_cloud)

        primitive = {
            "id": geom_id,
            "type": "cylinder",
            "center": params["center"],
            "axis": params["axis"],
            "radius": params["radius"],
            "obb": obb,
        }
    else:
        raise ValueError("Unknown primitive type")

    # Add point annotations
    annotated_pcd, annotations = add_annotations(
        pcd, inliers, geom_id, class_id, instance_id
    )

    # Load existing annotated PLY if it exists and merge
    ply_out = str(Path(out_dir) / "annotated.ply")
    existing_pcd, existing_annotations = load_annotated_ply(ply_out)

    if existing_pcd is not None and existing_annotations is not None:
        # Merge with existing
        annotated_pcd, annotations = merge_point_clouds_with_annotations(
            existing_pcd, existing_annotations, annotated_pcd, annotations
        )
        logger.info(
            f"Merged with existing annotated PLY ({len(existing_pcd.points)} existing points)"
        )

    # Save outputs
    save_ply_with_annotations(annotated_pcd, annotations, ply_out)

    # Load existing primitives and add new one
    primitives_path = str(Path(out_dir) / "primitives.json")
    primitives = load_primitives(primitives_path)

    # Check if primitive with this ID already exists and update it
    existing_idx = None
    for idx, p in enumerate(primitives):
        if p.get("id") == geom_id:
            existing_idx = idx
            break

    if existing_idx is not None:
        primitives[existing_idx] = primitive
    else:
        primitives.append(primitive)

    save_primitives(primitives, primitives_path)

    # Load semantic classes
    if semantic_classes_path is None:
        semantic_classes_path = str(Path(__file__).parent / "semantic_classes.json")
    semantic_classes = load_semantic_classes(semantic_classes_path)

    # Load existing semantic instances
    semantic_path = str(Path(out_dir) / "semantic.json")
    semantic_instances = load_semantic_instances(semantic_path)

    # Get class name
    class_name = semantic_classes.get(str(class_id), "unknown")
    instance_name = f"{class_name}_{instance_id}"

    # Create new semantic instance
    new_instance = {
        "class_id": class_id,
        "instance_id": instance_id,
        "name": instance_name,
        "obb": obb,
    }

    # Check if instance already exists and update it
    existing_idx = None
    for idx, inst in enumerate(semantic_instances):
        if inst.get("class_id") == class_id and inst.get("instance_id") == instance_id:
            existing_idx = idx
            break

    if existing_idx is not None:
        semantic_instances[existing_idx] = new_instance
    else:
        semantic_instances.append(new_instance)

    # Save semantic information
    save_semantic(semantic_classes, semantic_instances, semantic_path)

    logger.info(f"Saved annotated PLY to {ply_out}")
    logger.info(f"Updated primitives in {primitives_path}")
    logger.info(f"Updated semantic information in {semantic_path}")


if __name__ == "__main__":
    import argparse

    # Setup logging to see output on console
    setup_logging()

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="Input .ply file")
    parser.add_argument(
        "--type", help="plane or cylinder", choices=["plane", "cylinder"]
    )
    parser.add_argument("--geom_id", type=int, help="geometric primitive id")
    parser.add_argument("--class_id", type=int, help="semantic class id")
    parser.add_argument("--instance_id", type=int, help="semantic instance id")
    parser.add_argument("--out", default="output", help="Output folder")
    parser.add_argument(
        "--semantic_classes", help="Path to semantic classes JSON file", default=None
    )

    args = parser.parse_args()

    annotate(
        args.input,
        args.type,
        args.geom_id,
        args.class_id,
        args.instance_id,
        args.out,
        args.semantic_classes,
    )
