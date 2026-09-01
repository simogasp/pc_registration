# Point Cloud Annotation Tool

A tool for annotating 3D point clouds with geometric primitives and semantic information. This script processes point cloud data (`.ply` files) representing CAD model parts and generates annotated outputs with both geometric and semantic metadata.

---

## Overview

This tool allows you to:

1. **Fit geometric primitives** (planes, cylinders, spheres) to point cloud data using RANSAC
2. **Annotate points** with geometric IDs and semantic labels
3. **Generate structured metadata** in JSON format with schema validation
4. **Build incrementally** by adding multiple primitives to the same output directory

### Key Features

- **Preserves all points**: Annotates inliers while keeping outliers (with zero labels)
- **Incremental annotation**: Add multiple primitives to the same scene
- **Schema validation**: Auto-generates JSON schema for primitives
- **Semantic classes**: Maps class IDs to human-readable names
- **Oriented bounding boxes**: Computes OBB for each primitive and semantic instance

## Output Files

The tool generates four main files in the output directory:

| File | Description |
|------|-------------|
| `annotated.ply` | Point cloud with annotation properties (semantic_class, semantic_instance, geom_id) |
| `primitives.json` | Geometric primitives with parameters and OBBs |
| `semantic.json` | Semantic classes and instances with OBBs |
| `primitives_schema.json` | JSON schema for primitives validation (auto-copied) |

---

## Usage

### Basic Usage

```bash
python pc_annotation.py \
  --input <input.ply> \
  --type <plane|cylinder> \
  --geom_id <primitive_id> \
  --class_id <semantic_class_id> \
  --instance_id <instance_id> \
  --out <output_directory>
```

### Parameters

- `--input`: Path to input PLY file
- `--type`: Primitive type to fit (`plane` or `cylinder`)
- `--geom_id`: Unique ID for this geometric primitive (integer)
- `--class_id`: Semantic class ID (0=unlabeled, 1=wall, 2=bolt, etc.)
- `--instance_id`: Instance number for this class
- `--out`: Output directory (default: `output`)
- `--semantic_classes`: Path to semantic classes JSON (optional, defaults to `scripts/semantic_classes.json`)

---

## Examples

### Example 1: Annotate a Single Plane

```bash
# First primitive: annotate a wall (class_id=1, instance_id=0)
python pc_annotation.py \
  --input data/wall_scan.ply \
  --type plane \
  --geom_id 1 \
  --class_id 1 \
  --instance_id 0 \
  --out output/scene1
```

**Output:**

- `output/scene1/annotated.ply` - All points with wall inliers labeled
- `output/scene1/primitives.json` - Contains one plane primitive
- `output/scene1/semantic.json` - Contains "wall_0" instance
- `output/scene1/primitives_schema.json` - JSON schema for validation

### Example 2: Add More Primitives to the Same Scene

```bash
# Second primitive: add a bolt (class_id=2, instance_id=0)
python pc_annotation.py \
  --input data/bolt_scan.ply \
  --type cylinder \
  --geom_id 2 \
  --class_id 2 \
  --instance_id 0 \
  --out output/scene1

# Third primitive: add another wall (class_id=1, instance_id=1)
python pc_annotation.py \
  --input data/wall2_scan.ply \
  --type plane \
  --geom_id 3 \
  --class_id 1 \
  --instance_id 1 \
  --out output/scene1
```

**Result:**

- `primitives.json` now contains 3 primitives (2 planes, 1 cylinder)
- `semantic.json` contains 3 instances (wall_0, bolt_0, wall_1)
- `annotated.ply` is updated with the latest primitive's annotations

### Example 3: Update an Existing Primitive

```bash
# Re-run with same geom_id to update primitive parameters
python pc_annotation.py \
  --input data/wall_refined.ply \
  --type plane \
  --geom_id 1 \
  --class_id 1 \
  --instance_id 0 \
  --out output/scene1
```

**Result:**

- Primitive with `id=1` is updated (not duplicated)
- Semantic instance with `class_id=1, instance_id=0` is updated

---

## Output Format Details

### 1. Annotated PLY (`annotated.ply`)

ASCII PLY file with custom properties:

```none
ply
format ascii 1.0
element vertex <N>
property float x
property float y
property float z
property float nx
property float ny
property float nz
property uchar semantic_class
property ushort semantic_instance
property ushort geom_id
end_header
<x> <y> <z> <nx> <ny> <nz> <class> <instance> <geom>
...
```

**Important**:

- **All points preserved**: The output contains ALL points from the input
- **Inliers labeled**: Points fitting the primitive get non-zero annotation values
- **Outliers preserved**: Non-fitting points remain in the file with zero labels
- **Semantic_class**: 0 = unlabeled, 1+ = labeled class
- **Semantic_instance**: Instance number within the class
- **Geom_id**: Geometric primitive ID (matches `primitives.json`)

### 2. Primitives JSON (`primitives.json`)

```json
{
  "$schema": "./primitives_schema.json",
  "primitives": [
    {
      "id": 1,
      "type": "plane",
      "equation": [a, b, c, d],
      "obb": {
        "center": [x, y, z],
        "axes": [[...], [...], [...]],
        "extents": [width, height, depth]
      }
    },
    {
      "id": 2,
      "type": "cylinder",
      "center": [x, y, z],
      "axis": [dx, dy, dz],
      "radius": r,
      "obb": {...}
    }
  ]
}
```

**Plane**: Equation `ax + by + cz + d = 0`
**Cylinder**: Center point, unit axis vector, radius
**Sphere** (future): Center point, radius
**OBB**: Oriented bounding box (center, 3 axes, extents)

### 3. Semantic JSON (`semantic.json`)

```json
{
  "semantic_classes": {
    "0": "unlabeled",
    "1": "wall",
    "2": "bolt",
    "3": "rivet",
    "4": "stiffener",
    "5": "post"
  },
  "semantic_instances": [
    {
      "class_id": 1,
      "instance_id": 0,
      "name": "wall_0",
      "obb": {...}
    },
    {
      "class_id": 2,
      "instance_id": 0,
      "name": "bolt_0",
      "obb": {...}
    }
  ]
}
```

- **semantic_classes**: Loaded from `semantic_classes.json` (customizable)
- **semantic_instances**: Array of instances with names and OBBs
- **Instance names**: Automatically generated as `{class_name}_{instance_id}`

---

## 🔄 Incremental Annotation Workflow

The tool supports building complex scenes incrementally by running multiple times on the same output directory:

### How It Works

1. **First run**: Creates new output directory and all files
2. **Subsequent runs**:
   - Loads existing `primitives.json` and `semantic.json`
   - Appends new primitives/instances OR updates existing ones
   - Overwrites `annotated.ply` with current primitive's annotations

### ID-Based Update Logic

**Primitives**:

- If a primitive with the same `geom_id` exists → **UPDATE** it
- If no primitive with that `geom_id` exists → **APPEND** new primitive

**Semantic Instances**:

- If an instance with same `class_id` AND `instance_id` exists → **UPDATE** it
- Otherwise → **APPEND** new instance

### Example Workflow

```bash
# Step 1: Annotate first wall
python pc_annotation.py --input wall1.ply --type plane \
  --geom_id 1 --class_id 1 --instance_id 0 --out scene1

# primitives.json: [plane #1]
# semantic.json: [wall_0]

# Step 2: Add a bolt
python pc_annotation.py --input bolt1.ply --type cylinder \
  --geom_id 2 --class_id 2 --instance_id 0 --out scene1

# primitives.json: [plane #1, cylinder #2]
# semantic.json: [wall_0, bolt_0]

# Step 3: Add second wall
python pc_annotation.py --input wall2.ply --type plane \
  --geom_id 3 --class_id 1 --instance_id 1 --out scene1

# primitives.json: [plane #1, cylinder #2, plane #3]
# semantic.json: [wall_0, bolt_0, wall_1]

# Step 4: Update first wall with refined data
python pc_annotation.py --input wall1_refined.ply --type plane \
  --geom_id 1 --class_id 1 --instance_id 0 --out scene1

# primitives.json: [plane #1 UPDATED, cylinder #2, plane #3]
# semantic.json: [wall_0 UPDATED, bolt_0, wall_1]
```

### Best Practices

- **Use sequential geom_id**: 1, 2, 3, ... for easy tracking
- **Plan class_id**: Consistent IDs for object types (1=wall, 2=bolt, etc.)
- **Use sequential instance_id per class**: wall_0, wall_1, ...; bolt_0, bolt_1, ...
- **Keep input files**: PLY is overwritten each run, keep originals to re-process
- **Version control**: Consider keeping different scene versions in separate directories

### Important Notes

⚠️ **PLY File Limitation**: The `annotated.ply` file is **overwritten** on each run and contains annotations for **only the latest primitive**. To build a complete annotated point cloud with all primitives, you would need to:

- Process all primitives in sequence
- Merge the point clouds programmatically
- Or modify the script to accumulate annotations

Currently, the JSON files (`primitives.json` and `semantic.json`) accumulate data across runs, but the PLY file does not.

---

## Customization

### Custom Semantic Classes

Create a custom `semantic_classes.json`:

```json
{
  "semantic_classes": {
    "0": "unlabeled",
    "1": "pipe",
    "2": "valve",
    "3": "flange",
    "4": "support"
  }
}
```

Place it in the `scripts/` directory or specify with `--semantic_classes`:

```bash
python pc_annotation.py \
  --semantic_classes config/custom_classes.json \
  --input pipe.ply --type cylinder --geom_id 1 --class_id 1 --instance_id 0 \
  --out output
```

### RANSAC Parameters

Edit `pc_annotation.py` to adjust fitting parameters:

```python
# Plane fitting (line ~36)
plane_model, inliers = pcd.segment_plane(
    distance_threshold=0.005,  # Inlier distance threshold (meters)
    ransac_n=3,  # Min points for plane
    num_iterations=2000,  # RANSAC iterations
)

# Cylinder fitting (line ~76)
inliers = np.where(np.abs(radial - radius) < 0.01)[0]  # Radial tolerance
```

### OBB Margin for Planes

Planes use a 2D bounding box with a small margin in the normal direction:

```python
# Planar OBB computation (line ~113)
def compute_planar_obb(pcd, plane_normal, margin=0.01):  # Adjust margin
    ...
```

---

## Use Cases

### Industrial Robotics

- **Welding robots**: Locate stiffeners, seams, and weld paths
- **Assembly robots**: Identify bolt holes, flanges, and mounting points
- **Inspection robots**: Measure dimensions and detect defects

### CAD Model Understanding

- **Reverse engineering**: Extract geometric primitives from scans
- **Quality control**: Compare as-built vs. as-designed
- **Digital twin creation**: Build semantic 3D models of facilities

### Machine Learning

- **Training data**: Create labeled datasets for 3D object detection
- **Synthetic data validation**: Verify procedural generation pipelines
- **Benchmark creation**: Ground truth for segmentation algorithms

---

## 🛠 Technical Details

### Algorithms

- **Plane fitting**: Open3D's `segment_plane` (RANSAC-based)
- **Cylinder fitting**: PCA-based axis estimation + radial distance thresholding
- **OBB computation**:
  - 3D primitives: Open3D's `get_oriented_bounding_box`
  - Planar primitives: Custom 2D projection with normal margin

### Data Structures

- **PLY format**: ASCII with custom properties (semantic_class, semantic_instance, geom_id)
- **JSON schema**: Draft 07 with conditional validation for primitive types
- **Coordinate system**: Right-handed, same as input PLY

### Error Handling

- **Empty point clouds**: Raises `ValueError`
- **Qhull errors**: Falls back to axis-aligned bounding box
- **Missing files**: Creates new files if they don't exist
- **Invalid IDs**: No validation (user responsibility)

---

## File Structure

```none
scripts/
├── pc_annotation.py           # Main annotation script
├── pc_annotation.md           # This documentation
├── semantic_classes.json      # Default semantic class definitions
├── primitives_schema.json     # JSON schema for primitives
├── verify_annotated_ply.py    # Verification utility
└── create_test_cylinder.py    # Test data generator
```

---

## Limitations

1. **PLY overwrite**: Each run overwrites `annotated.ply` (only latest primitive annotated)
2. **Manual execution**: Requires running script separately for each primitive
3. **No multi-primitive fitting**: Cannot fit multiple primitives in one command
4. **Input segmentation**: Assumes input PLY contains one dominant primitive
5. **Limited primitive types**: Currently supports planes and cylinders only (sphere in schema)
6. **No visualization**: No built-in 3D viewer (use CloudCompare, MeshLab, etc.)
