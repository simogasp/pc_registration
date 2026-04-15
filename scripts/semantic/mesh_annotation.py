"""Mesh annotation script for geometric and semantic labeling.

This script annotates triangular meshes with geometric primitives and semantic information.
It fits primitives to mesh vertices and annotates both vertices and faces.
"""

import json
import logging
import shutil
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Optional

import open3d as o3d


logger = logging.getLogger(__name__)


def load_mesh(ply_path: str) -> o3d.geometry.TriangleMesh:
    """Load a triangular mesh from a PLY file.

    Args:
        ply_path: Path to input .ply file.

    Returns:
        The loaded Open3D triangular mesh.
    """
    mesh = o3d.io.read_triangle_mesh(ply_path)
    if not mesh.has_vertices():
        raise ValueError("Mesh has no vertices.")
    if not mesh.has_triangles():
        raise ValueError("Mesh has no triangles.")
    return mesh


def fit_plane(vertices: np.ndarray) -> Tuple[np.ndarray, list]:
    """Fit a plane to vertices using RANSAC.

    Args:
        vertices: Nx3 array of vertex positions.

    Returns:
        plane_params: Plane equation parameters [a, b, c, d].
        inliers: List of inlier indices for the plane.
    """
    # Create temporary point cloud for fitting
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vertices)

    plane_model, inliers = pcd.segment_plane(
        distance_threshold=20, ransac_n=3, num_iterations=2000
    )
    a, b, c, d = plane_model
    return np.array([a, b, c, d]), inliers


def fit_cylinder(vertices: np.ndarray) -> Tuple[Dict, list]:
    """Fit a cylinder using PCA-based estimation.

    Args:
        vertices: Nx3 array of vertex positions.

    Returns:
        params: Cylinder dictionary: {"center": [..], "axis": [..], "radius": r}
        inliers: List of inlier indices for the cylinder.
    """
    # Estimate axis via PCA
    pts_mean = vertices.mean(axis=0)
    u, s, vh = np.linalg.svd(vertices - pts_mean)
    axis = vh[0]

    # Distance from axis gives radius
    proj = (vertices - pts_mean) @ axis
    closest = pts_mean + np.outer(proj, axis)
    radial = np.linalg.norm(vertices - closest, axis=1)
    radius = np.median(radial)

    # Inliers: close to radius
    inliers = np.where(np.abs(radial - radius) < 170)[0].tolist()

    return {
        "center": pts_mean.tolist(),
        "axis": axis.tolist(),
        "radius": float(radius),
    }, inliers


def compute_obb(vertices: np.ndarray) -> Dict:
    """Compute an oriented bounding box for vertices.

    Args:
        vertices: Nx3 array of vertex positions.

    Returns:
        obb: Dict containing center, axes, and extents.
    """
    # Create temporary point cloud for OBB computation
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vertices)

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
    vertices: np.ndarray, plane_normal: np.ndarray, margin: float = 25
) -> Dict:
    """Compute an oriented bounding box for planar vertices.

    For planar data, computes a 2D bounding box in the plane and adds a small
    margin in the normal direction to avoid qhull errors.

    Args:
        vertices: Nx3 array of vertex positions.
        plane_normal: Normal vector of the plane [a, b, c] from plane equation.
        margin: Margin to add in normal direction (default: 0.01).

    Returns:
        obb: Dict containing center, axes, and extents.
    """
    # Normalize the plane normal
    normal = plane_normal / np.linalg.norm(plane_normal)

    # Create coordinate system in the plane
    if np.abs(normal[2]) < 0.9:
        u = np.cross(normal, np.array([0, 0, 1]))
    else:
        u = np.cross(normal, np.array([1, 0, 0]))
    u = u / np.linalg.norm(u)
    v = np.cross(normal, u)
    v = v / np.linalg.norm(v)

    # Compute centroid
    center = vertices.mean(axis=0)

    # Project points onto plane coordinate system
    pts_centered = vertices - center
    u_coords = pts_centered @ u
    v_coords = pts_centered @ v

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


def add_annotations(
    mesh: o3d.geometry.TriangleMesh,
    inliers: list,
    geom_id: int,
    class_id: int,
    instance_id: int,
) -> Tuple[o3d.geometry.TriangleMesh, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Prepare annotation attributes for vertices and faces.

    Args:
        mesh: Input triangular mesh.
        inliers: List of inlier vertex indices belonging to a primitive.
        geom_id: Primitive ID.
        class_id: Semantic Class ID.
        instance_id: Semantic Instance ID.

    Returns:
        A tuple containing:
            - The original mesh (unchanged)
            - A dictionary of vertex annotation arrays
            - A dictionary of face annotation arrays
    """
    n_vertices = len(mesh.vertices)
    n_faces = len(mesh.triangles)

    # Initialize vertex annotations
    vertex_geom = np.zeros(n_vertices, dtype=np.uint16)
    vertex_sem_class = np.zeros(n_vertices, dtype=np.uint8)
    vertex_sem_inst = np.zeros(n_vertices, dtype=np.uint16)

    # Set vertex annotations for inliers
    vertex_geom[inliers] = geom_id
    vertex_sem_class[inliers] = class_id
    vertex_sem_inst[inliers] = instance_id

    # Initialize face annotations
    face_geom = np.zeros(n_faces, dtype=np.uint16)
    face_sem_class = np.zeros(n_faces, dtype=np.uint8)
    face_sem_inst = np.zeros(n_faces, dtype=np.uint16)

    # Annotate faces: a face gets annotation if all 3 vertices have the same annotation
    triangles = np.asarray(mesh.triangles)
    for i, (v0, v1, v2) in enumerate(triangles):
        # Check geometric annotation
        if (
            vertex_geom[v0] == vertex_geom[v1] == vertex_geom[v2]
            and vertex_geom[v0] != 0
        ):
            face_geom[i] = vertex_geom[v0]

        # Check semantic class annotation
        if (
            vertex_sem_class[v0] == vertex_sem_class[v1] == vertex_sem_class[v2]
            and vertex_sem_class[v0] != 0
        ):
            face_sem_class[i] = vertex_sem_class[v0]

        # Check semantic instance annotation
        if (
            vertex_sem_inst[v0] == vertex_sem_inst[v1] == vertex_sem_inst[v2]
            and vertex_sem_inst[v0] != 0
        ):
            face_sem_inst[i] = vertex_sem_inst[v0]

    vertex_annotations = {
        "semantic_class": vertex_sem_class,
        "semantic_instance": vertex_sem_inst,
        "geom_id": vertex_geom,
    }

    face_annotations = {
        "semantic_class": face_sem_class,
        "semantic_instance": face_sem_inst,
        "geom_id": face_geom,
    }

    return mesh, vertex_annotations, face_annotations


def save_ply_with_annotations(
    mesh: o3d.geometry.TriangleMesh,
    vertex_annotations: Dict[str, np.ndarray],
    face_annotations: Dict[str, np.ndarray],
    out_path: str,
):
    """Save a PLY file with custom annotation properties for vertices and faces.

    Args:
        mesh: Input triangular mesh.
        vertex_annotations: Dictionary mapping vertex property names to numpy arrays.
        face_annotations: Dictionary mapping face property names to numpy arrays.
        out_path: Output file path ending in .ply
    """
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    n_vertices = len(vertices)
    n_faces = len(triangles)

    # Check for optional attributes
    has_vertex_colors = mesh.has_vertex_colors()
    has_vertex_normals = mesh.has_vertex_normals()

    vertex_colors = np.asarray(mesh.vertex_colors) if has_vertex_colors else None
    vertex_normals = np.asarray(mesh.vertex_normals) if has_vertex_normals else None

    # Write PLY file manually
    with open(out_path, "w") as f:
        # Header
        f.write("ply\n")
        f.write("format ascii 1.0\n")

        # Vertex element
        f.write(f"element vertex {n_vertices}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")

        if has_vertex_normals:
            f.write("property float nx\n")
            f.write("property float ny\n")
            f.write("property float nz\n")

        if has_vertex_colors:
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")

        # Custom vertex annotation properties
        f.write("property uchar semantic_class\n")
        f.write("property ushort semantic_instance\n")
        f.write("property ushort geom_id\n")

        # Face element
        f.write(f"element face {n_faces}\n")
        f.write("property list uchar int vertex_indices\n")

        # Custom face annotation properties
        f.write("property uchar semantic_class\n")
        f.write("property ushort semantic_instance\n")
        f.write("property ushort geom_id\n")

        f.write("end_header\n")

        # Vertex data
        for i in range(n_vertices):
            # Position
            f.write(f"{vertices[i, 0]} {vertices[i, 1]} {vertices[i, 2]}")

            # Normal
            if has_vertex_normals and vertex_normals is not None:
                f.write(
                    f" {vertex_normals[i, 0]} {vertex_normals[i, 1]} {vertex_normals[i, 2]}"
                )

            # Color
            if has_vertex_colors and vertex_colors is not None:
                r = int(vertex_colors[i, 0] * 255)
                g = int(vertex_colors[i, 1] * 255)
                b = int(vertex_colors[i, 2] * 255)
                f.write(f" {r} {g} {b}")

            # Vertex annotations
            f.write(f" {vertex_annotations['semantic_class'][i]}")
            f.write(f" {vertex_annotations['semantic_instance'][i]}")
            f.write(f" {vertex_annotations['geom_id'][i]}")
            f.write("\n")

        # Face data
        for i in range(n_faces):
            # Vertex indices (always 3 for triangular mesh)
            f.write(f"3 {triangles[i, 0]} {triangles[i, 1]} {triangles[i, 2]}")

            # Face annotations
            f.write(f" {face_annotations['semantic_class'][i]}")
            f.write(f" {face_annotations['semantic_instance'][i]}")
            f.write(f" {face_annotations['geom_id'][i]}")
            f.write("\n")


def load_annotated_ply(
    ply_path: str,
) -> Tuple[
    Optional[o3d.geometry.TriangleMesh],
    Optional[Dict[str, np.ndarray]],
    Optional[Dict[str, np.ndarray]],
]:
    """Load an existing annotated PLY mesh with custom properties.

    Args:
        ply_path: Path to annotated PLY file.

    Returns:
        Tuple of (mesh, vertex_annotations, face_annotations) or (None, None, None) if file doesn't exist.
    """
    if not Path(ply_path).exists():
        return None, None, None

    # Read the PLY file manually to extract custom properties
    with open(ply_path, "r") as f:
        lines = f.readlines()

    # Parse header
    header_end = 0
    n_vertices = 0
    n_faces = 0
    vertex_properties = []
    face_properties = []
    current_element = None

    for i, line in enumerate(lines):
        if line.startswith("element vertex"):
            n_vertices = int(line.split()[-1])
            current_element = "vertex"
        elif line.startswith("element face"):
            n_faces = int(line.split()[-1])
            current_element = "face"
        elif line.startswith("property"):
            parts = line.split()
            if current_element == "vertex":
                if parts[1] == "list":
                    continue  # Skip list properties in vertex
                prop_type = parts[1]
                prop_name = parts[2]
                vertex_properties.append((prop_name, prop_type))
            elif current_element == "face":
                if parts[1] == "list":
                    face_properties.append(("vertex_indices", "list"))
                else:
                    prop_type = parts[1]
                    prop_name = parts[2]
                    face_properties.append((prop_name, prop_type))
        elif line.strip() == "end_header":
            header_end = i + 1
            break

    # Read data
    data_lines = lines[header_end:]

    # Parse vertices
    vertices = []
    vertex_normals = []
    vertex_colors = []
    vertex_semantic_class = []
    vertex_semantic_instance = []
    vertex_geom_id = []

    has_normals = any(name == "nx" for name, _ in vertex_properties)
    has_colors = any(name == "red" for name, _ in vertex_properties)
    has_annotations = any(name == "semantic_class" for name, _ in vertex_properties)

    for j in range(n_vertices):
        values = data_lines[j].split()
        idx = 0

        # x, y, z
        vertices.append(
            [float(values[idx]), float(values[idx + 1]), float(values[idx + 2])]
        )
        idx += 3

        # normals
        if has_normals:
            vertex_normals.append(
                [float(values[idx]), float(values[idx + 1]), float(values[idx + 2])]
            )
            idx += 3

        # colors
        if has_colors:
            vertex_colors.append(
                [
                    int(values[idx]) / 255.0,
                    int(values[idx + 1]) / 255.0,
                    int(values[idx + 2]) / 255.0,
                ]
            )
            idx += 3

        # annotations
        if has_annotations:
            vertex_semantic_class.append(int(values[idx]))
            vertex_semantic_instance.append(int(values[idx + 1]))
            vertex_geom_id.append(int(values[idx + 2]))

    # Parse faces
    triangles = []
    face_semantic_class = []
    face_semantic_instance = []
    face_geom_id = []

    has_face_annotations = any(
        name == "semantic_class"
        for name, _ in face_properties
        if name != "vertex_indices"
    )

    for j in range(n_faces):
        values = data_lines[n_vertices + j].split()
        idx = 0

        # Vertex indices (skip the count, assume 3)
        n_verts = int(values[idx])
        idx += 1
        triangles.append([int(values[idx]), int(values[idx + 1]), int(values[idx + 2])])
        idx += n_verts

        # Face annotations
        if has_face_annotations:
            face_semantic_class.append(int(values[idx]))
            face_semantic_instance.append(int(values[idx + 1]))
            face_geom_id.append(int(values[idx + 2]))

    # Create mesh
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(np.array(vertices))
    mesh.triangles = o3d.utility.Vector3iVector(np.array(triangles))

    if has_normals and vertex_normals:
        mesh.vertex_normals = o3d.utility.Vector3dVector(np.array(vertex_normals))

    if has_colors and vertex_colors:
        mesh.vertex_colors = o3d.utility.Vector3dVector(np.array(vertex_colors))

    vertex_annotations = None
    if has_annotations:
        vertex_annotations = {
            "semantic_class": np.array(vertex_semantic_class, dtype=np.uint8),
            "semantic_instance": np.array(vertex_semantic_instance, dtype=np.uint16),
            "geom_id": np.array(vertex_geom_id, dtype=np.uint16),
        }

    face_annotations = None
    if has_face_annotations:
        face_annotations = {
            "semantic_class": np.array(face_semantic_class, dtype=np.uint8),
            "semantic_instance": np.array(face_semantic_instance, dtype=np.uint16),
            "geom_id": np.array(face_geom_id, dtype=np.uint16),
        }

    return mesh, vertex_annotations, face_annotations


def merge_meshes_with_annotations(
    existing_mesh: o3d.geometry.TriangleMesh,
    existing_vertex_annotations: Dict[str, np.ndarray],
    existing_face_annotations: Dict[str, np.ndarray],
    new_mesh: o3d.geometry.TriangleMesh,
    new_vertex_annotations: Dict[str, np.ndarray],
    new_face_annotations: Dict[str, np.ndarray],
) -> Tuple[o3d.geometry.TriangleMesh, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Merge two meshes with their annotations.

    Args:
        existing_mesh: Existing mesh.
        existing_vertex_annotations: Existing vertex annotations.
        existing_face_annotations: Existing face annotations.
        new_mesh: New mesh to add.
        new_vertex_annotations: New vertex annotations to add.
        new_face_annotations: New face annotations to add.

    Returns:
        Tuple of (merged mesh, merged vertex annotations, merged face annotations).
    """
    # Merge meshes
    merged_mesh = existing_mesh + new_mesh

    # Merge vertex annotations
    merged_vertex_annotations = {
        "semantic_class": np.concatenate(
            [
                existing_vertex_annotations["semantic_class"],
                new_vertex_annotations["semantic_class"],
            ]
        ),
        "semantic_instance": np.concatenate(
            [
                existing_vertex_annotations["semantic_instance"],
                new_vertex_annotations["semantic_instance"],
            ]
        ),
        "geom_id": np.concatenate(
            [existing_vertex_annotations["geom_id"], new_vertex_annotations["geom_id"]]
        ),
    }

    # Merge face annotations
    merged_face_annotations = {
        "semantic_class": np.concatenate(
            [
                existing_face_annotations["semantic_class"],
                new_face_annotations["semantic_class"],
            ]
        ),
        "semantic_instance": np.concatenate(
            [
                existing_face_annotations["semantic_instance"],
                new_face_annotations["semantic_instance"],
            ]
        ),
        "geom_id": np.concatenate(
            [existing_face_annotations["geom_id"], new_face_annotations["geom_id"]]
        ),
    }

    return merged_mesh, merged_vertex_annotations, merged_face_annotations


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


def annotate(
    ply_path: str,
    primitive_type: str,
    geom_id: int,
    class_id: int,
    instance_id: int,
    out_dir: str,
    semantic_classes_path: Optional[str] = None,
):
    """Main annotation pipeline for meshes.

    Args:
        ply_path: Path to input PLY mesh.
        primitive_type: "plane" or "cylinder".
        geom_id: Primitive ID to assign in output.
        class_id: Semantic class ID.
        instance_id: Instance ID for that class.
        out_dir: Output folder.
        semantic_classes_path: Path to semantic classes JSON file.
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    mesh = load_mesh(ply_path)
    vertices = np.asarray(mesh.vertices)

    if primitive_type == "plane":
        params, inliers = fit_plane(vertices)
        # Extract fitted vertices
        inlier_vertices = vertices[inliers]
        # For planes, use 2D bounding box with margin
        plane_normal = params[:3]  # [a, b, c] from plane equation
        obb = compute_planar_obb(inlier_vertices, plane_normal)

        primitive = {
            "id": geom_id,
            "type": "plane",
            "equation": params.tolist(),
            "obb": obb,
        }
    elif primitive_type == "cylinder":
        params, inliers = fit_cylinder(vertices)
        # Extract fitted vertices
        inlier_vertices = vertices[inliers]
        # For cylinders, use standard 3D OBB
        obb = compute_obb(inlier_vertices)

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

    # Add annotations
    annotated_mesh, vertex_annotations, face_annotations = add_annotations(
        mesh, inliers, geom_id, class_id, instance_id
    )

    # Load existing annotated PLY if it exists and merge
    ply_out = str(Path(out_dir) / "annotated.ply")
    existing_mesh, existing_vertex_annotations, existing_face_annotations = (
        load_annotated_ply(ply_out)
    )

    if (
        existing_mesh is not None
        and existing_vertex_annotations is not None
        and existing_face_annotations is not None
    ):
        # Merge with existing
        annotated_mesh, vertex_annotations, face_annotations = (
            merge_meshes_with_annotations(
                existing_mesh,
                existing_vertex_annotations,
                existing_face_annotations,
                annotated_mesh,
                vertex_annotations,
                face_annotations,
            )
        )
        logger.info(
            f"Merged with existing annotated mesh ({len(existing_mesh.vertices)} existing vertices)"
        )

    # Save outputs
    save_ply_with_annotations(
        annotated_mesh, vertex_annotations, face_annotations, ply_out
    )

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

    logger.info(f"Saved annotated mesh to {ply_out}")
    logger.info(f"Updated primitives in {primitives_path}")
    logger.info(f"Updated semantic information in {semantic_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="Input .ply mesh file")
    parser.add_argument(
        "--type", help="plane or cylinder", choices=["plane", "cylinder"]
    )
    parser.add_argument("--geom_id", type=int)
    parser.add_argument("--class_id", type=int)
    parser.add_argument("--instance_id", type=int)
    parser.add_argument("--out", default="output")
    parser.add_argument(
        "--semantic_classes",
        help="Path to semantic classes JSON file",
        default=None,
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
