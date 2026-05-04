# Sequence Registration Scripts

This directory contains scripts for processing, registering, and fusing sequential point cloud scans.

## Scripts

### validate_ground_truth.py

**Purpose:** Validates ground truth transformations by comparing them with ICP registration results.

**Key Features:**

- Performs pairwise ICP registration between scans at configurable intervals
- Computes rotation error (degrees) and translation error (mm) vs ground truth
- Generates comprehensive statistics (mean, median, std, min, max)
- Optional JSON output with per-pair details

**Usage:**

```bash
# Basic validation (consecutive scans)
uv run ./validate_ground_truth.py --input data/scans

# Validate with custom step interval (e.g., every 2nd scan)
uv run ./validate_ground_truth.py --input data/scans --step 2

# Save detailed results to JSON
uv run ./validate_ground_truth.py --input data/scans --output validation_results.json

# Without ground truth initialization (identity matrix)
uv run ./validate_ground_truth.py --input data/scans --no-gt-init

# Using Generalized ICP (GICP) instead of classic ICP
uv run ./validate_ground_truth.py --input data/scans --use-gicp
```

**Parameters:**

- `--input`: Directory containing `.ply` and `.json` scan pairs
- `--voxel-size`: Voxel size (mm) for downsampling (default: 50.0)
- `--max-correspondence-distance`: ICP correspondence distance (default: 150.0)
- `--step`: Step size for selecting pairs (1=consecutive, 2=every other scan, etc.) (default: 1)
- `--no-gt-init`: Use identity initialization instead of ground truth
- `--use-gicp`: Use Generalized ICP (GICP) instead of classic ICP for registration
- `--output`: Optional JSON file for detailed results
- `--log-level`: Set logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL) (default: INFO)
- `--start-scan`: Index of first scan to process (0-based, inclusive)
- `--end-scan`: Index of last scan to process (0-based, inclusive)

---

### fuse_scans.py

**Purpose:** Simple fusion of multiple scans using ground truth transformations without optimization.

**Key Features:**

- Direct transformation and concatenation of scans using ground truth poses
- Creates raw map (all points) and fused map (voxel downsampled)
- Optional statistical outlier removal
- Configurable scan stride to fuse every nth scan
- Binary PLY output format
- Saves execution parameters to `parameters.json` for reproducibility

**Output Files:**

- `raw_map.ply`: Binary PLY file with all transformed points concatenated
- `fused_map.ply`: Binary PLY file with voxel downsampling applied
- `parameters.json`: Execution parameters including timestamps and scan ranges

**Usage:**

```bash
# Basic fusion
uv run ./fuse_scans.py --input data/scans --output output/fused

# With outlier removal
uv run ./fuse_scans.py --input data/scans --output output/fused --remove-outliers

# Only fused map (skip raw)
uv run ./fuse_scans.py --input data/scans --output output/fused --skip-raw

# Fuse every 5th scan (faster, lower density)
uv run ./fuse_scans.py --input data/scans --output output/fused --step 5
```

**Parameters:**

- `--input`: Directory containing `.ply` and `.json` scan pairs
- `--output`: Output directory (default: output/fused_scans)
- `--voxel-size`: Voxel size (mm) for fusion downsampling (default: 10.0)
- `--skip-raw`: Skip saving raw concatenated map
- `--skip-fused`: Skip saving fused map
- `--remove-outliers`: Apply statistical outlier removal
- `--outlier-nb-neighbors`: Number of neighbors for outlier detection (default: 20)
- `--outlier-std-ratio`: Standard deviation ratio threshold (default: 2.0)
- `--filter-distant`: Filter out points that are too far from the point cloud centroid
- `--max-distance`: Maximum distance (mm) from centroid for filtering
- `--distance-percentile`: Distance percentile threshold for filtering (default: 99.0)
- `--start-scan`: Index of first scan to process (0-based, inclusive)
- `--end-scan`: Index of last scan to process (0-based, inclusive)
- `--step`: Process every nth scan within the selected range (default: 1 = all scans). For example, `--step 5` fuses scans 0, 5, 10, 15, ...

---

### prepare_scan_dataset.py

**Purpose:** Convert a directory of XYZ point cloud scans and a flat-text poses file to binary PLY and JSON files compatible with all registration scripts.

**Key Features:**

- Discovers all `.xyz` files in the input directory, sorted numerically by stem
- Converts each XYZ scan to binary PLY format preserving the original stem name (`0.xyz` -> `0.ply`)
- Reads a flat-text poses file (default: `poses.txt`) and writes one JSON pose file per scan in the `{"H": [[...]]}` format
- Optional `--scale` factor applied consistently to both point cloud coordinates and pose translations
- Output directory defaults to the same directory as the input files

**Usage:**

```bash
# Convert in-place (PLY and JSON written alongside XYZ files)
uv run ./prepare_scan_dataset.py --input data/my_scans

# Write to a separate output directory
uv run ./prepare_scan_dataset.py --input data/my_scans --output-dir data/my_scans/converted

# Apply a scale factor (e.g. metres to millimetres)
uv run ./prepare_scan_dataset.py --input data/my_scans --output-dir data/my_scans/converted --scale 1000.0

# Custom poses file path
uv run ./prepare_scan_dataset.py --input data/my_scans --poses data/my_scans/ground_truth.txt
```

**Parameters:**

- `--input` / `-i`: Directory containing `.xyz` scan files and the poses file (required)
- `--output-dir` / `-o`: Directory where PLY and JSON files will be written (default: same as input directory)
- `--poses` / `-p`: Path to the flat-text poses file (default: `poses.txt` inside the input directory)
- `--scale` / `-s`: Scale factor applied to point cloud coordinates and pose translations (default: 1.0)
- `--log-level`: Set logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL) (default: INFO)

---

### poses_txt_to_json.py

**Purpose:** Convert a row-major flat-text poses file to per-scan JSON pose files compatible with all registration scripts.

**Key Features:**

- Reads a text file where each line holds 16 space-separated floats (one 4x4 matrix in row-major order)
- Writes one numbered JSON file per line (`0.json`, `1.json`, ...) in the format expected by `registration_common.load_transformation_matrix`
- Skips blank lines and `#` comment lines
- Output directory defaults to the same directory as the input file

**Usage:**

```bash
# Write JSON files alongside the poses file
uv run ./poses_txt_to_json.py --input data/dataset_real_lidar/poses.txt

# Write JSON files to a separate directory
uv run ./poses_txt_to_json.py --input data/dataset_real_lidar/poses.txt --output-dir data/dataset_real_lidar/poses_json

# Apply a scale factor to the translation components
uv run ./poses_txt_to_json.py --input data/dataset_real_lidar/poses.txt --scale 1000.0
```

**Parameters:**

- `--input`: Path to the input poses text file
- `--output-dir`: Directory where JSON files will be written (default: same directory as the input file)
- `--scale` / `-s`: Scale factor applied to the translation part of each pose (default: 1.0)
- `--log-level`: Set logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL) (default: INFO)

**Input format:**

Each non-empty, non-comment line must contain exactly 16 space-separated floats representing a 4x4 homogeneous transformation matrix in row-major order:

```none
r00 r01 r02 tx  r10 r11 r12 ty  r20 r21 r22 tz  0 0 0 1
```

---

### multiway_registration.py

**Purpose:** Performs multiway registration using pose graph optimization to align multiple sequential scans and create a fused map.

**Key Features:**

- Builds pose graph with odometry edges (consecutive scans) and optional loop closure detection
- Supports ground truth initialization or preliminary ICP-based registration
- Global optimization using Levenberg-Marquardt method
- Outputs optimized poses (JSON) and fused point cloud map (PLY)

**Usage:**

```bash
# With ground truth poses
uv run ./multiway_registration.py --input data/scans --output output/multiway

# Without ground truth (registration-only mode)
uv run ./multiway_registration.py --input data/scans --output output/multiway --no-ground-truth

# With loop closure detection
uv run ./multiway_registration.py --input data/scans --output output/multiway --loop-closure-distance 5000.0

# With outlier removal and distance filtering
uv run ./multiway_registration.py --input data/scans --output output/multiway --remove-outliers --filter-distant

# Using Generalized ICP (GICP) instead of classic ICP
uv run ./multiway_registration.py --input data/scans --output output/multiway --use-gicp
```

**Parameters:**

- `--input`: Directory containing `.ply` and `.json` scan pairs
- `--output`: Output directory for results
- `--voxel-size-input`: Voxel size (mm) for downsampling input scans **used only during registration** (pose graph construction and ICP). Set to `0` to disable. Smaller values improve map accuracy at the cost of higher memory and compute (default: 50.0).
- `--voxel-size-fusion`: Voxel size (mm) for downsampling the final fused map (default: 10.0). After pose optimisation, the script reloads all scans at **full resolution** and transforms them with the optimised poses before applying this voxel filter. This makes `--voxel-size-fusion` the sole parameter controlling output map density, independently of `--voxel-size-input`.
- `--max-correspondence-distance`: ICP correspondence distance threshold (default: 150.0)
- `--loop-closure-distance`: Distance threshold for loop closure detection (optional)
- `--no-ground-truth`: Disable ground truth initialization, use sequential ICP instead
- `--use-gicp`: Use Generalized ICP (GICP) instead of classic ICP for registration
- `--start-scan`: Index of first scan to process (0-based, inclusive)
- `--end-scan`: Index of last scan to process (0-based, inclusive)
- `--step`: Process every nth scan within the selected range (default: 1 = all scans). For example, `--step 5` processes scans 0, 5, 10, 15, ...
- `--remove-outliers`: Apply statistical outlier removal to fused map
- `--outlier-nb-neighbors`: Number of neighbors for outlier detection (default: 20)
- `--outlier-std-ratio`: Standard deviation ratio threshold (default: 2.0)
- `--filter-distant`: Filter out points that are too far from the point cloud centroid
- `--max-distance`: Maximum distance (mm) from centroid for filtering
- `--distance-percentile`: Distance percentile threshold for filtering (default: 99.0)

---

### localize_against_map.py

**Purpose:** Localize individual scans against a global map using RANSAC-based global registration with optional ICP/GICP refinement.

**Key Features:**

- RANSAC-based global registration using FPFH features for robust initial alignment
- Optional two-stage pipeline: RANSAC (coarse) → ICP/GICP (fine refinement)
- Configurable refinement resolution: reuse RANSAC clouds, use original point clouds, or a separate voxel size
- Compares estimated poses with ground truth for accuracy evaluation
- Supports both individual JSON ground truth files and single poses file
- Generates comprehensive statistics across all scans
- Optional JSON output with per-scan details

**Usage:**

```bash
# Basic RANSAC-only localization
uv run ./localize_against_map.py --input data/scans --map output/fused_map.ply --output localization_results.json

# RANSAC + GICP refinement (recommended for improved accuracy)
uv run ./localize_against_map.py --input data/scans --map output/fused_map.ply --refine-poses --use-gicp --output localization_results.json

# RANSAC + ICP refinement
uv run ./localize_against_map.py --input data/scans --map output/fused_map.ply --refine-poses --output localization_results.json

# Using a single poses file for ground truth (e.g., optimized_poses.json)
uv run ./localize_against_map.py --input data/scans --map output/fused_map.ply --poses output/optimized_poses.json --output localization_results.json

# With ground truth initialization (for validation)
uv run ./localize_against_map.py --input data/scans --map output/fused_map.ply --use-gt-init --output localization_results.json

# Localize subset of scans
uv run ./localize_against_map.py --input data/scans --map output/fused_map.ply --start-scan 0 --end-scan 49 --output localization_results.json

# Custom ICP refinement distance
uv run ./localize_against_map.py --input data/scans --map output/fused_map.ply --refine-poses --use-gicp --icp-refinement-distance 100.0 --output localization_results.json

# RANSAC + GICP refinement at original (undownsampled) resolution
uv run ./localize_against_map.py --input data/scans --map output/fused_map.ply --refine-poses --use-gicp --refinement-voxel-size 0 --output localization_results.json

# RANSAC + GICP refinement at a custom finer voxel size (e.g. 20mm)
uv run ./localize_against_map.py --input data/scans --map output/fused_map.ply --refine-poses --use-gicp --refinement-voxel-size 20 --output localization_results.json

# Skip RANSAC and use pre-estimated poses as initialization for ICP refinement
uv run ./localize_against_map.py --input data/scans --map output/fused_map.ply --estimated-poses output/ransac_results/estimated_poses.json --refine-poses --output localization_results.json

# Skip RANSAC and use pre-estimated poses as initialization for GICP refinement at a custom voxel size
uv run ./localize_against_map.py --input data/scans --map output/fused_map.ply --estimated-poses output/ransac_results/estimated_poses.json --refine-poses --use-gicp --refinement-voxel-size 20 --output localization_results.json
```

**Parameters:**

- `--input`: Directory containing `.ply` and `.json` scan pairs
- `--map`: Path to global map PLY file (e.g., fused_map_optimized.ply)
- `--voxel-size`: Voxel size (mm) for downsampling scans and map (default: 50.0)
- `--max-correspondence-distance`: RANSAC correspondence distance (default: 150.0)
- `--refine-poses`: Enable ICP/GICP refinement after RANSAC (default: False)
- `--use-gicp`: Use Generalized ICP instead of point-to-plane ICP for refinement (default: False)
- `--icp-refinement-distance`: Max correspondence distance for ICP/GICP refinement (default: 50.0)
- `--refinement-voxel-size`: Voxel size (mm) for downsampling during refinement. If not set, reuses the RANSAC clouds (no extra loading). Set to `0` to use the original undownsampled point clouds (default: None)
- `--estimated-poses`: Path to a JSON file of pre-estimated poses (same format as `estimated_poses.json` produced by this script). When provided, RANSAC global registration is skipped and these poses are used as the initial estimate for refinement. Has no effect unless `--refine-poses` is also passed.
- `--use-gt-init`: Use ground truth as initialization (otherwise uses identity)
- `--output`: Optional JSON file for detailed results
- `--start-scan`: Index of first scan to process (0-based, inclusive)
- `--end-scan`: Index of last scan to process (0-based, inclusive)
- `--poses`: Optional single JSON file containing all ground truth poses (like optimized_poses.json)
- `--log-level`: Set logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL) (default: INFO)

**Output Files:**

When `--output` is specified, the script creates:

- Main results JSON file (specified path): Contains all localization results and statistics
- `parameters.json`: Execution parameters including timestamps and scan ranges
- `estimated_poses.json`: All estimated poses in the same format as optimized_poses.json

**Output Statistics:**

- Rotation error (degrees): mean, median, std, min, max
- Translation error (mm): mean, median, std, min, max
- Localization fitness: mean, median, std, min, max
- Localization RMSE (mm): mean, median, std, min, max

**Use Cases:**

- **Map Quality Evaluation**: Test how well scans can be localized against the fused map
- **Localization Algorithm Testing**: Benchmark localization accuracy with/without initialization
- **Ground Truth Validation**: Compare different ground truth sources (individual JSONs vs optimized poses)
- **Subset Analysis**: Evaluate localization performance on specific scan ranges

---

### visualize_sequence.py

**Purpose:** Interactive visualization of scan acquisition sequence with animation.

**Key Features:**

- Animated playback of scan sequence showing acquisition progression
- Displays world coordinate frame (origin) and LiDAR frame (moving sensor)
- **LiDAR trajectory visualization** - yellow line showing the sensor path through all scans
- Optional fused map overlay for context
- **Fused map downsampling controls** - toggle downsampling and adjust voxel size interactively
- Interactive controls: pause/resume, manual navigation, map/trajectory toggle
- Continuous loop playback
- **Automatic camera bounds** - properly fits all geometry for easy navigation
- On-demand frame statistics - press F to display current scan number and frame position

**Usage:**

```bash
# Basic visualization
uv run ./visualize_sequence.py --input data/scans

# With fused map overlay
uv run ./visualize_sequence.py --input data/scans --fused-map output/fused_map.ply

# Slow animation (2 seconds per frame)
uv run ./visualize_sequence.py --input data/scans --speed 2.0

# Visualize subset of scans
uv run ./visualize_sequence.py --input data/scans --start-scan 0 --end-scan 49

# Use optimized poses instead of ground truth
uv run ./visualize_sequence.py --input data/scans --fused-map output/fused_map.ply --poses output/optimized_poses.json
```

**Parameters:**

- `--input`: Directory containing `.ply` and `.json` scan pairs
- `--fused-map`: Optional path to fused map PLY file for overlay
- `--speed`: Animation speed in seconds per frame (default: 1.0)
- `--start-scan`: Index of first scan to display (0-based, inclusive)
- `--end-scan`: Index of last scan to display (0-based, inclusive)
- `--poses`: Optional single JSON file containing all poses (like optimized_poses.json). If provided, uses these poses instead of individual ground truth JSON files

**Keyboard Controls:**

- `SPACE`: Pause/Resume animation
- `LEFT/RIGHT ARROW`: Previous/Next scan (when paused)
- `F`: Print current frame statistics (scan number, frame position, status)
- `T`: Toggle trajectory visibility (yellow line)
- `M`: Toggle fused map visibility
- `D`: Toggle fused map downsampling on/off
- `[ / ]`: Decrease/Increase downsampling voxel size (when downsampling is enabled)
- `+/-`: Increase/Decrease point size (native Open3D)
- `Q/ESC`: Quit

**Display:**

- **Yellow line**: LiDAR trajectory path
- **RGB axes (origin)**: World coordinate frame
- **RGB axes (moving)**: Current LiDAR pose
- **Colored points**: Current scan
- **Gray points**: Fused map (if loaded)

---

### plot_validation_errors.py

**Purpose:** Generate histogram plots for validation errors from validate_ground_truth.py results.

**Key Features:**

- Creates histogram distributions for rotation and translation errors
- Displays statistics (mean, median, std, min, max) on plots
- Uses common plotting utilities for consistency
- High-resolution output (300 DPI)

**Usage:**

```bash
# Generate plots from validation results
uv run ./plot_validation_errors.py --input validation_results.json --output output/validation_plots
```

**Parameters:**

- `--input`: Path to validation JSON file (from validate_ground_truth.py)
- `--output`: Output directory for histogram plots (default: output/validation_histograms)
- `--log-level`: Set logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL) (default: INFO)

**Output Files:**

- `histogram_rotation_error.png`: Distribution of rotation errors (degrees)
- `histogram_translation_error.png`: Distribution of translation errors (mm)

---

### plot_localization_errors.py

**Purpose:** Generate histogram plots for localization errors from localize_against_map.py results.

**Key Features:**

- Creates histogram distributions for rotation and translation errors
- Displays statistics (mean, median, std, min, max) on plots
- Uses common plotting utilities for consistency
- High-resolution output (300 DPI)

**Usage:**

```bash
# Generate plots from localization results
uv run ./plot_localization_errors.py --input localization_results.json --output output/localization_plots
```

**Parameters:**

- `--input`: Path to localization JSON file (from localize_against_map.py)
- `--output`: Output directory for histogram plots (default: output/localization_histograms)
- `--log-level`: Set logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL) (default: INFO)

**Output Files:**

- `histogram_rotation_error.png`: Distribution of rotation errors (degrees)
- `histogram_translation_error.png`: Distribution of translation errors (mm)

---

### plot_localization_comparison.py

**Purpose:** Generate comparative box plots for localization results across different voxel sizes and methods (RANSAC, RANSAC+ICP, RANSAC+GICP).

**Key Features:**

- Automatically discovers and loads localization results from multiple directory configurations
- Creates grouped box plots comparing methods across voxel sizes
- Supports filtering to specific voxel sizes for focused analysis
- Parses directory names to extract method and voxel size information
- Interactive filtering with validation and helpful error messages

**Usage:**

```bash
# Generate comparison plots for all voxel sizes
uv run ./plot_localization_comparison.py --input output/localization/comparison --output output/comparison_plots

# Compare only specific voxel sizes
uv run ./plot_localization_comparison.py --input output/localization/comparison --output output/comparison_plots --voxel-sizes "100,150,450"

# With custom logging level
uv run ./plot_localization_comparison.py --input output/localization/comparison --output output/comparison_plots --log-level DEBUG
```

**Parameters:**

- `--input`: Base directory containing localization result subdirectories (e.g., `ransac_vs100`, `ransac_gicp_vs450`)
- `--output`: Output directory for comparison plots (default: output/localization_comparison_plots)
- `--voxel-sizes`: Comma-separated list of voxel sizes to plot (e.g., '50,100,450'). If not specified, all voxel sizes are plotted
- `--log-level`: Set logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL) (default: INFO)

**Directory Naming Convention:**

The script expects subdirectories following the pattern `{method}_vs{voxel_size}`:

- `ransac_vs100`: RANSAC-only localization with 100mm voxel size
- `ransac_icp_vs150`: RANSAC+ICP refinement with 150mm voxel size
- `ransac_gicp_vs450`: RANSAC+GICP refinement with 450mm voxel size

**Output Files:**

The script generates both combined comparison plots and individual plots for each voxel size.

When no voxel size filter is applied:

- `comparison_rotation_error.png`: Combined grouped box plot of rotation errors across all voxel sizes
- `comparison_translation_error.png`: Combined grouped box plot of translation errors across all voxel sizes
- `comparison_rotation_error_{voxel_size}.png`: Individual rotation error plots for each voxel size (e.g., `comparison_rotation_error_50.png`, `comparison_rotation_error_100.png`, etc.)
- `comparison_translation_error_{voxel_size}.png`: Individual translation error plots for each voxel size

When `--voxel-sizes` filter is applied (e.g., "100,150,450"):

- `comparison_rotation_error_100_150_450.png`: Combined rotation error comparison for filtered voxel sizes
- `comparison_translation_error_100_150_450.png`: Combined translation error comparison for filtered voxel sizes
- `comparison_rotation_error_100.png`, `comparison_rotation_error_150.png`, `comparison_rotation_error_450.png`: Individual rotation error plots for each filtered voxel size
- `comparison_translation_error_100.png`, `comparison_translation_error_150.png`, `comparison_translation_error_450.png`: Individual translation error plots for each filtered voxel size

**Example:** For 6 voxel sizes (50, 100, 150, 200, 300, 450), the script generates 14 plots total: 2 combined plots and 12 individual plots (2 per voxel size).

**Color Scheme:**

- Red: RANSAC only
- Orange: RANSAC + ICP
- Green: RANSAC + GICP

**Use Cases:**

- **Method Comparison**: Compare RANSAC-only vs RANSAC+ICP vs RANSAC+GICP refinement
- **Voxel Size Analysis**: Understand how downsampling affects localization accuracy
- **Parameter Optimization**: Identify optimal voxel size for a specific method
- **Focused Analysis**: Use `--voxel-sizes` to compare specific configurations without regenerating all plots

---

### plotting_common.py

**Purpose:** Shared plotting utilities for error visualization across validation and localization scripts.

**Key Functions:**

- `create_histogram()`: Create histogram plot with statistics overlay
- `create_error_histograms()`: Generate both rotation and translation error histograms
- `create_grouped_boxplot()`: Create grouped box plot for comparing multiple methods across different conditions

**Features:**

- Consistent styling across all plots
- Mean and median overlay lines (histograms)
- Statistics text box with key metrics
- Grouped box plots with customizable colors and labels
- High-resolution output (300 DPI)

---

### registration_common.py

**Purpose:** Shared utility functions used by all registration scripts.

**Key Functions:**

- `find_scan_pairs()`: Find matching `.ply` and `.json` file pairs
- `load_transformation_matrix()`: Load 4x4 transformation from JSON
- `load_point_cloud()`: Load PLY with optional downsampling and normal estimation
- `load_and_transform_scan()`: Load a PLY at full resolution and apply a rigid transformation (used for map fusion)
- `pairwise_registration()`: Perform ICP or GICP registration between two point clouds (supports both classic ICP and Generalized ICP)
- `remove_outliers()`: Remove statistical outliers from point cloud
- `filter_distant_points()`: Filter points beyond distance threshold or percentile
- `save_point_cloud_binary()`: Save a point cloud as binary PLY
- `save_parameters()`: Save execution parameters to a JSON file
- `rotation_error_degrees()`: Compute rotation error between two rotation matrices
- `translation_error()`: Compute Euclidean distance between translation vectors
- `load_poses_from_file()`: Load all poses from a single JSON file (like optimized_poses.json)
- `save_poses_to_file()`: Save poses to a JSON file

**Constants:**

- `NORMAL_ESTIMATION_RADIUS_MULTIPLIER = 2`: For computing normal estimation radius
- `DEFAULT_NORMAL_ESTIMATION_RADIUS_MM = 100.0`: Default radius when no voxel size specified
- `NORMAL_ESTIMATION_MAX_NEIGHBORS = 30`: Max neighbors for normal estimation

---

## Input File Format

All scripts expect paired files:

- **`.ply` files**: Point cloud data (XYZ coordinates, binary or ASCII format)
- **`.json` files**: Ground truth transformations with matching filenames

JSON format:

```json
{
  "H": [
    [r11, r12, r13, tx],
    [r21, r22, r23, ty],
    [r31, r32, r33, tz],
    [0.0, 0.0, 0.0, 1.0]
  ]
}
```

Where `H` is the 4x4 homogeneous transformation matrix (world frame).

---

## Scan Range Selection

All main scripts (`multiway_registration.py`, `validate_ground_truth.py`, `fuse_scans.py`, `localize_against_map.py`, and `visualize_sequence.py`) support processing a subset of scans from a sequence using the `--start-scan` and `--end-scan` options.

**Usage:**

```bash
# Process scans 0-49 (first 50 scans)
uv run ./multiway_registration.py --input data/scans --output output/subset --start-scan 0 --end-scan 49

# Validate only scans 10-30
uv run ./validate_ground_truth.py --input data/scans --start-scan 10 --end-scan 30

# Localize scans 0-99 against map
uv run ./localize_against_map.py --input data/scans --map output/fused_map.ply --start-scan 0 --end-scan 99

# Fuse scans starting from index 5 to the end
uv run ./fuse_scans.py --input data/scans --output output/partial --start-scan 5

# Visualize specific scan range
uv run ./visualize_sequence.py --input data/scans --start-scan 10 --end-scan 50
```

**Parameters:**

- `--start-scan`: Index of first scan to process (0-based, inclusive). Default: 0
- `--end-scan`: Index of last scan to process (0-based, inclusive). Default: last scan

**Notes:**

- Both indices are 0-based and inclusive
- Omitting `--start-scan` starts from the beginning (index 0)
- Omitting `--end-scan` processes until the last available scan
- The scripts validate that `start_scan <= end_scan` and both are within valid range
- Useful for testing on subsets of large datasets or processing specific segments

---

## Workflow Recommendations

### To create a gloabal map

```bash
uv run ./fuse_scans.py --input data/dataset_real_lidar/ --output output/sequence_registration/fuse/filtered_distance_full --filter-distant --distance-percentile 99.999
```

### For Visualizing Scan Sequence

To inspect the scan acquisition sequence and verify alignment:

```bash
# Visualize with animation
uv run ./visualize_sequence.py --input data/scans

# With fused map overlay for quality check
uv run ./visualize_sequence.py --input data/scans --fused-map output/fused_map.ply

# Visualize with optimized poses
uv run ./visualize_sequence.py --input data/scans --fused-map output/optimized/fused_map_optimized.ply --poses output/optimized/optimized_poses.json
```

### For Localization Evaluation

Evaluate how well individual scans can be localized against a global map:

```bash
# 1. Create a global map using multiway registration
 uv run ./fuse_scans.py --input data/dataset_real_lidar/ --output output/sequence_registration/fuse/filtered_distance_full --filter-distant --distance-percentile 99.999

# 2. Test localization against the map (RANSAC only)
uv run ./localize_against_map.py --input data/scans --map output/map_optimized/fused_map_optimized.ply --output output/localization_ransac.json

# 3. Test with GICP refinement
uv run ./localize_against_map.py --input data/scans --map output/map_optimized/fused_map_optimized.ply --refine-poses --use-gicp --output output/localization_gicp.json

# 4. Generate error distribution plots for each method
uv run ./plot_localization_errors.py --input output/localization_ransac.json --output output/localization_plots/ransac
uv run ./plot_localization_errors.py --input output/localization_gicp.json --output output/localization_plots/gicp

# 5. Test localization with ground truth initialization (for comparison)
uv run ./localize_against_map.py --input data/scans --map output/map_optimized/fused_map_optimized.ply --use-gt-init --output output/localization_gt_init.json

# 6. Compare localization using optimized poses as ground truth
uv run ./localize_against_map.py --input data/scans --map output/map_optimized/fused_map_optimized.ply --poses output/map_optimized/optimized_poses.json --output output/localization_vs_optimized.json
```

### For Method and Voxel Size Comparison

Compare localization performance across different methods (RANSAC, RANSAC+ICP, RANSAC+GICP) and voxel sizes:

```bash
# 1. Create a global map
uv run ./fuse_scans.py --input data/dataset_real_lidar/ --output output/sequence_registration/fuse/filtered_distance_full --filter-distant --distance-percentile 99.999


# 2. Run localization with different configurations
# For each voxel size (50, 100, 150, 200, 300, 450), run:
#   - RANSAC only
#   - RANSAC + ICP
#   - RANSAC + GICP
# Example for voxel size 100mm:
uv run ./localize_against_map.py --input data/scans --map output/map_optimized/fused_map_optimized.ply --voxel-size 100 --output output/comparison/ransac_vs100/localization_results.json
uv run ./localize_against_map.py --input data/scans --map output/map_optimized/fused_map_optimized.ply --voxel-size 100 --refine-poses --output output/comparison/ransac_icp_vs100/localization_results.json
uv run ./localize_against_map.py --input data/scans --map output/map_optimized/fused_map_optimized.ply --voxel-size 100 --refine-poses --use-gicp --output output/comparison/ransac_gicp_vs100/localization_results.json
# Optionally, refine at original resolution to compare:
uv run ./localize_against_map.py --input data/scans --map output/map_optimized/fused_map_optimized.ply --voxel-size 100 --refine-poses --use-gicp --refinement-voxel-size 0 --output output/comparison/ransac_gicp_orig_vs100/localization_results.json

# 3. Generate comprehensive comparison plots
uv run ./plot_localization_comparison.py --input output/comparison --output output/comparison_plots

# 4. Generate focused comparison for specific voxel sizes
uv run ./plot_localization_comparison.py --input output/comparison --output output/comparison_plots_focused --voxel-sizes "100,150,450"
```

### For Validation Analysis

Analyze validation results with error distribution plots:

```bash
# 1. Run validation on ground truth
uv run ./validate_ground_truth.py --input data/scans --output output/validation_results.json

# 2. Generate error distribution plots
uv run ./plot_validation_errors.py --input output/validation_results.json --output output/validation_plots
```

### For Registration Without Ground Truth

```bash
uv run ./multiway_registration.py --input data/scans --output output/no_gt --no-ground-truth
```
