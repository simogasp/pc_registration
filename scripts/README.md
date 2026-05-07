# Scripts

## flip_and_scale_pc.py

This script is used to flip and scale point clouds.
It can be used to preprocess point cloud to bring it in the reference frame of the visualizer.

### Usage

For example to flip a point cloud coming from the sampling of the maquette (stp file)

```bash
uv run ./scripts/flip_and_scale_pc.py --input data/maquette12k.ply --output data/sameref/maquette12k.ply --flip z
```

To flip a scan coming from the ROS simulator (flip + scale back to mm)

```bash
uv run ./scripts/flip_and_scale_pc.py --input data/y_-0.75m/pcl_out_time104-116000000.ply --output data/sameref/y_-0.75m.ply --scale 1000 --flip nx
```

## load_and_display.py

This script loads a point cloud from file and displays it using the Open3D visualizer.

## global_registration.py

This script performs global registration between two point clouds using RANSAC followed by ICP refinement.

- it generates a random initial transformation to be applied to the source point cloud
- it then tries to correct the rotation around the gravity axis (y-axis) using a simple heuristic
- aligns the centers of the two point clouds to improve convergence
- performs RANSAC global registration checking that
  - the fitness (ratio of inliers) is above a threshold `--min-fitness`
  - the solution is physically plausible (the gravity axis is not flipped)
- refines the result with ICP

Example usage:

```bash
# align scan to maquette E Shape
uv run ./scripts/global_registration.py --source data/sameref/ry_0_degres.ply --target  data/sameref/E_shape_maq15k.ply --max_iter_icp 100000 --voxel-size 40 --min-fitness 0.35

# align scan to maquette full
uv run ./scripts/global_registration.py --source data/sameref/ry_0_degres.ply --target  data/sameref/maquette27k.ply --max_iter_icp 100000 --voxel-size 30 --min-fitness 0.4
uv run ./scripts/global_registration.py --source data/sameref/ry_0_degres.ply --target  data/sameref/maquette27k.ply --max_iter_icp 100000 --voxel-size 30 --min-fitness 0.53
```

## sample_mesh.py

Samples a point cloud from the surface of a triangle mesh. Supports any mesh format readable by Open3D (`.ply`, `.obj`, `.stl`, `.off`, `.gltf`, `.glb`, …). Surface normals are transferred from the mesh to the sampled points.

Two sampling strategies are available:

- **`poisson-disk`** (default): spatially uniform distribution — no two output points are closer than a minimum radius derived from the requested density. Produces clean, well-spread clouds suitable for registration and reconstruction.
- **`uniform`**: Monte Carlo sampling proportional to triangle area. Fast, produces exactly the requested number of points, but may show local clustering artefacts.

### Parameters

| Argument | Default | Description |
| --- | --- | --- |
| `--input` / `-i` | required | Input mesh file |
| `--output` / `-o` | required | Output `.ply` point cloud |
| `--num-points` / `-n` | 100 000 | Point count. For `uniform`: exact output size. For `poisson-disk`: target that sets the minimum inter-point distance; actual count may be lower. Mutually exclusive with `--spacing`. |
| `--spacing` / `-d` | — | Desired average distance between points in mesh units (e.g. `0.5` for 0.5 mm). Derives `N` automatically from the mesh surface area. Mutually exclusive with `--num-points`. |
| `--method` / `-m` | `poisson-disk` | `uniform` or `poisson-disk` |
| `--init-factor` | 5 | Poisson-disk only: oversampling factor before elimination (≥ 5 recommended). Higher values improve spatial uniformity. |
| `--use-triangle-normal` | off | Poisson-disk only: assign flat face normals instead of interpolated vertex normals. |
| `--log-level` | `INFO` | Logging verbosity (`DEBUG`, `INFO`, `WARNING`, `ERROR`). |

### Usage

```bash
# Poisson-disk sampling at 0.5 mm spacing (mesh in mm)
uv run ./scripts/sample_mesh.py --input data/maquette.obj --output data/maquette_sampled.ply --spacing 0.5

# Poisson-disk sampling with an explicit point count
uv run ./scripts/sample_mesh.py --input data/maquette.obj --output data/maquette_sampled.ply --num-points 200000

# Uniform sampling (exact count, faster)
uv run ./scripts/sample_mesh.py --input data/maquette.stl --output data/maquette_sampled.ply --num-points 100000 --method uniform
```
