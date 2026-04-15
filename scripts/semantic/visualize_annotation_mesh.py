"""Visualize annotated meshes with multiple visualization modes.

This script loads annotated mesh data and provides interactive visualization
with three modes:
1. Original 3D model
2. Geometric segmentation (colored by primitive type with OBBs)
3. Semantic segmentation (colored by semantic class with OBBs)

Controls:
- Press '1' or 'M': Original model view
- Press '2' or 'G': Geometric segmentation view
- Press '3' or 'S': Semantic segmentation view
- Press 'Q' or ESC: Quit
"""

import json
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import argparse

import open3d as o3d
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb

from registration.utils.logging import setup_logging

logger = logging.getLogger(__name__)

# Color for unlabeled faces (RGB in range [0, 1])
UNLABELED_COLOR = np.array([0.5, 0.5, 0.5])


def load_annotated_mesh(
    ply_path: str,
) -> Tuple[o3d.geometry.TriangleMesh, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Load annotated mesh PLY file with custom vertex and face properties.

    Args:
        ply_path: Path to annotated mesh PLY file.

    Returns:
        Tuple of (mesh, vertex_annotations dict, face_annotations dict).
    """
    if not Path(ply_path).exists():
        raise FileNotFoundError(f"Annotated mesh PLY not found: {ply_path}")

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
    has_vertex_annotations = any(
        name == "semantic_class" for name, _ in vertex_properties
    )

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
        if has_vertex_annotations:
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
    if has_vertex_annotations:
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


def load_primitives(json_path: str) -> List[Dict]:
    """Load primitives from JSON file.

    Args:
        json_path: Path to primitives JSON file.

    Returns:
        List of primitive dictionaries.
    """
    if not Path(json_path).exists():
        raise FileNotFoundError(f"Primitives JSON not found: {json_path}")

    with open(json_path, "r") as f:
        data = json.load(f)

    return data.get("primitives", [])


def load_semantic(json_path: str) -> Tuple[Dict[str, str], List[Dict]]:
    """Load semantic information from JSON file.

    Args:
        json_path: Path to semantic JSON file.

    Returns:
        Tuple of (semantic_classes dict, semantic_instances list).
    """
    if not Path(json_path).exists():
        raise FileNotFoundError(f"Semantic JSON not found: {json_path}")

    with open(json_path, "r") as f:
        data = json.load(f)

    return data.get("semantic_classes", {}), data.get("semantic_instances", [])


def get_n_colors_set3(n: int) -> np.ndarray:
    """Get n colors from matplotlib's Set3 colormap.

    Args:
        n: Number of colors to get.

    Returns:
        Array of shape (n, 3) with RGB colors in range [0, 1].
    """
    cmap = plt.get_cmap("Set3")
    colors = np.array([to_rgb(cmap(i)) for i in range(n)])
    return colors


def generate_colors(n: int, seed: int = 42) -> np.ndarray:
    """Generate n distinct random colors.

    Args:
        n: Number of colors to generate.
        seed: Random seed for reproducibility.

    Returns:
        Array of shape (n, 3) with RGB colors in range [0, 1].
    """
    np.random.seed(seed)
    colors = np.random.rand(n, 3)
    # Ensure colors are not too dark
    colors = np.clip(colors * 0.7 + 0.3, 0, 1)
    return colors


def get_n_colors(n: int, seed: int = 42) -> np.ndarray:
    """Get n distinct colors, preferring Set3 colormap if n <= 12.

    Args:
        n: Number of colors to generate.
        seed: Random seed for random generation if needed.

    Returns:
        Array of shape (n, 3) with RGB colors in range [0, 1].
    """
    SET3_SIZE = 12  # Set3 colormap has 12 colors

    if n <= SET3_SIZE:
        return get_n_colors_set3(n)
    else:
        logger.warning(
            f"Requested {n} colors exceeds Set3 colormap size ({SET3_SIZE}), using random colors"
        )
        return generate_colors(n, seed=seed)


def create_obb_lineset(obb_data: Dict) -> o3d.geometry.LineSet:
    """Create a LineSet representing an oriented bounding box.

    Args:
        obb_data: Dictionary with 'center', 'axes', and 'extents'.

    Returns:
        Open3D LineSet geometry.
    """
    center = np.array(obb_data["center"])
    axes = np.array(obb_data["axes"])
    extents = np.array(obb_data["extents"])

    # Create 8 corners of the box
    corners = []
    for i in [-1, 1]:
        for j in [-1, 1]:
            for k in [-1, 1]:
                corner = (
                    center
                    + i * extents[0] / 2 * axes[0]
                    + j * extents[1] / 2 * axes[1]
                    + k * extents[2] / 2 * axes[2]
                )
                corners.append(corner)

    corners = np.array(corners)

    # Define edges connecting corners
    lines = [
        [0, 1],
        [0, 2],
        [0, 4],
        [1, 3],
        [1, 5],
        [2, 3],
        [2, 6],
        [3, 7],
        [4, 5],
        [4, 6],
        [5, 7],
        [6, 7],
    ]

    # Create LineSet
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(corners)
    line_set.lines = o3d.utility.Vector2iVector(lines)

    # Set color (yellow for visibility)
    colors = [[1, 1, 0] for _ in range(len(lines))]
    line_set.colors = o3d.utility.Vector3dVector(colors)

    return line_set


def apply_face_colors_to_mesh(
    mesh: o3d.geometry.TriangleMesh, face_colors: np.ndarray
) -> o3d.geometry.TriangleMesh:
    """Apply per-face colors to a mesh by converting to vertex colors.

    Open3D doesn't support per-face colors directly, so we duplicate vertices
    and assign colors per-triangle.

    Args:
        mesh: Input mesh.
        face_colors: Array of shape (n_faces, 3) with RGB colors in range [0, 1].

    Returns:
        New mesh with per-face colors applied as vertex colors.
    """
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    normals = np.asarray(mesh.vertex_normals) if mesh.has_vertex_normals() else None

    # Create new vertices and triangles with duplicated data
    new_vertices = []
    new_normals = []
    new_colors = []
    new_triangles = []

    for i, (v0, v1, v2) in enumerate(triangles):
        # Get the color for this face
        color = face_colors[i]

        # Add the three vertices of this triangle
        idx_start = len(new_vertices)
        new_vertices.extend([vertices[v0], vertices[v1], vertices[v2]])
        new_colors.extend([color, color, color])

        if normals is not None:
            new_normals.extend([normals[v0], normals[v1], normals[v2]])

        # Add new triangle with new indices
        new_triangles.append([idx_start, idx_start + 1, idx_start + 2])

    # Create new mesh
    new_mesh = o3d.geometry.TriangleMesh()
    new_mesh.vertices = o3d.utility.Vector3dVector(np.array(new_vertices))
    new_mesh.triangles = o3d.utility.Vector3iVector(np.array(new_triangles))
    new_mesh.vertex_colors = o3d.utility.Vector3dVector(np.array(new_colors))

    if normals is not None:
        new_mesh.vertex_normals = o3d.utility.Vector3dVector(np.array(new_normals))

    return new_mesh


class MeshAnnotationVisualizer:
    """Interactive visualizer for annotated meshes."""

    def __init__(self, data_dir: str):
        """Initialize the visualizer.

        Args:
            data_dir: Directory containing annotated.ply, primitives.json, and semantic.json.
        """
        self.data_dir = Path(data_dir)

        # Load data
        logger.info(f"Loading data from {data_dir}")
        ply_path = self.data_dir / "annotated.ply"
        primitives_path = self.data_dir / "primitives.json"
        semantic_path = self.data_dir / "semantic.json"

        self.mesh, self.vertex_annotations, self.face_annotations = load_annotated_mesh(
            str(ply_path)
        )
        self.primitives = load_primitives(str(primitives_path))
        self.semantic_classes, self.semantic_instances = load_semantic(
            str(semantic_path)
        )

        # Store original colors
        if self.mesh.has_vertex_colors():
            self.original_vertex_colors = np.asarray(self.mesh.vertex_colors).copy()
        else:
            # Default gray color
            self.original_vertex_colors = np.ones((len(self.mesh.vertices), 3)) * 0.5

        # Visualization mode: 0=original, 1=geometric, 2=semantic
        self.mode = 0

        # Create visualizer
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.geometries = []
        self.current_mesh = None

        logger.info(f"Loaded {len(self.mesh.vertices)} vertices")
        logger.info(f"Loaded {len(self.mesh.triangles)} faces")
        logger.info(f"Loaded {len(self.primitives)} primitives")
        logger.info(f"Loaded {len(self.semantic_instances)} semantic instances")

    def apply_original_colors(self):
        """Apply original model colors."""
        logger.info("Mode 1: Original model view")

        # Create mesh with original vertex colors
        self.current_mesh = o3d.geometry.TriangleMesh(self.mesh)
        self.current_mesh.vertex_colors = o3d.utility.Vector3dVector(
            self.original_vertex_colors
        )
        if self.mesh.has_vertex_normals():
            self.current_mesh.compute_vertex_normals()

        # Remove OBBs
        self.clear_geometries()
        self.vis.add_geometry(self.current_mesh, reset_bounding_box=False)
        self.vis.update_geometry(self.current_mesh)

    def apply_geometric_colors(self):
        """Apply geometric segmentation colors with OBBs (colored by type)."""
        logger.info("Mode 2: Geometric segmentation view")

        # Get unique primitive types from primitives JSON
        primitive_types = {
            primitive.get("type")
            for primitive in self.primitives
            if "type" in primitive
        }
        unique_types = sorted([t for t in primitive_types if t is not None])

        # Generate colors for each type
        n_types = len(unique_types)
        type_colors = get_n_colors(n_types, seed=42)

        # Create type to color mapping
        type_color_map = {
            prim_type: type_colors[i] for i, prim_type in enumerate(unique_types)
        }

        # Create geom_id to type mapping
        geom_id_to_type = {}
        for primitive in self.primitives:
            if "id" in primitive and "type" in primitive:
                geom_id_to_type[primitive["id"]] = primitive["type"]

        # Create color mapping for geom_ids based on their type
        color_map = {}
        for geom_id, prim_type in geom_id_to_type.items():
            color_map[geom_id] = type_color_map.get(prim_type, UNLABELED_COLOR)
        color_map[0] = UNLABELED_COLOR  # Gray for unlabeled

        # Apply colors to faces
        n_faces = len(self.mesh.triangles)
        face_colors = np.zeros((n_faces, 3))
        for i, geom_id in enumerate(self.face_annotations["geom_id"]):
            face_colors[i] = color_map.get(geom_id, UNLABELED_COLOR)

        # Create mesh with per-face colors
        self.current_mesh = apply_face_colors_to_mesh(self.mesh, face_colors)
        if self.mesh.has_vertex_normals():
            self.current_mesh.compute_vertex_normals()

        # Clear and add geometries
        self.clear_geometries()
        self.vis.add_geometry(self.current_mesh, reset_bounding_box=False)

        # Add OBBs for each primitive
        for primitive in self.primitives:
            if "obb" in primitive:
                obb_lineset = create_obb_lineset(primitive["obb"])
                self.geometries.append(obb_lineset)
                self.vis.add_geometry(obb_lineset, reset_bounding_box=False)

        self.vis.update_geometry(self.current_mesh)

    def apply_semantic_colors(self):
        """Apply semantic segmentation colors with OBBs."""
        logger.info("Mode 3: Semantic segmentation view")

        # Get unique semantic class IDs from face annotations
        unique_class_ids = np.unique(self.face_annotations["semantic_class"])
        unique_class_ids = unique_class_ids[
            unique_class_ids > 0
        ]  # Exclude unlabeled (0)

        # Generate colors for each class
        n_classes = len(unique_class_ids)
        class_colors = get_n_colors(n_classes, seed=123)

        # Create color mapping
        color_map = {
            class_id: class_colors[i] for i, class_id in enumerate(unique_class_ids)
        }
        color_map[0] = UNLABELED_COLOR  # Gray for unlabeled

        # Apply colors to faces
        n_faces = len(self.mesh.triangles)
        face_colors = np.zeros((n_faces, 3))
        for i, class_id in enumerate(self.face_annotations["semantic_class"]):
            face_colors[i] = color_map[class_id]

        # Create mesh with per-face colors
        self.current_mesh = apply_face_colors_to_mesh(self.mesh, face_colors)
        if self.mesh.has_vertex_normals():
            self.current_mesh.compute_vertex_normals()

        # Clear and add geometries
        self.clear_geometries()
        self.vis.add_geometry(self.current_mesh, reset_bounding_box=False)

        # Add OBBs for each semantic instance
        for instance in self.semantic_instances:
            if "obb" in instance:
                obb_lineset = create_obb_lineset(instance["obb"])
                self.geometries.append(obb_lineset)
                self.vis.add_geometry(obb_lineset, reset_bounding_box=False)

        self.vis.update_geometry(self.current_mesh)

    def clear_geometries(self):
        """Clear all geometries."""
        for geom in self.geometries:
            self.vis.remove_geometry(geom, reset_bounding_box=False)
        self.geometries.clear()
        if self.current_mesh is not None:
            self.vis.remove_geometry(self.current_mesh, reset_bounding_box=False)

    def key_callback_1(self, vis):
        """Callback for key '1' or 'M': Original model."""
        self.mode = 0
        self.apply_original_colors()
        return False

    def key_callback_2(self, vis):
        """Callback for key '2' or 'G': Geometric segmentation."""
        self.mode = 1
        self.apply_geometric_colors()
        return False

    def key_callback_3(self, vis):
        """Callback for key '3' or 'S': Semantic segmentation."""
        self.mode = 2
        self.apply_semantic_colors()
        return False

    def run(self):
        """Run the interactive visualizer."""
        # Create window
        self.vis.create_window(
            window_name="Mesh Annotation Visualizer", width=1280, height=720
        )

        # Register key callbacks
        self.vis.register_key_callback(ord("1"), self.key_callback_1)
        self.vis.register_key_callback(ord("M"), self.key_callback_1)
        self.vis.register_key_callback(ord("2"), self.key_callback_2)
        self.vis.register_key_callback(ord("G"), self.key_callback_2)
        self.vis.register_key_callback(ord("3"), self.key_callback_3)
        self.vis.register_key_callback(ord("S"), self.key_callback_3)

        # Prepare original mesh for initial view
        self.current_mesh = o3d.geometry.TriangleMesh(self.mesh)
        self.current_mesh.vertex_colors = o3d.utility.Vector3dVector(
            self.original_vertex_colors
        )
        if self.mesh.has_vertex_normals():
            self.current_mesh.compute_vertex_normals()

        # Add initial geometry (original view)
        self.vis.add_geometry(self.current_mesh)

        # Set view
        self.vis.get_render_option().mesh_show_wireframe = False
        self.vis.get_render_option().mesh_show_back_face = True
        self.vis.get_render_option().background_color = np.array([0.1, 0.1, 0.1])

        logger.info("Visualizer started")
        logger.info("Controls:")
        logger.info("  1 or M: Original model view")
        logger.info("  2 or G: Geometric segmentation view (with OBBs)")
        logger.info("  3 or S: Semantic segmentation view (with OBBs)")
        logger.info("  Q or ESC: Quit")

        # Run visualizer
        self.vis.run()
        self.vis.destroy_window()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Visualize annotated meshes with multiple modes"
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Input directory containing annotated.ply, primitives.json, and semantic.json",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging()

    try:
        visualizer = MeshAnnotationVisualizer(args.input)
        visualizer.run()
    except Exception as e:
        logger.error(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()
