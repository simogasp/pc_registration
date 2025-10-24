# Point Cloud Registration

This project provides implementations of various point cloud registration algorithms using Open3D, including:

- **ICP (Iterative Closest Point)**: Point-to-point and point-to-plane variants
- **Global Registration**: RANSAC-based feature matching with FPFH features
- **Point Cloud Visualization**: Load and display point clouds with bounding boxes

## Prerequisites

- Python 3.12 or higher
- [uv](https://github.com/astral-sh/uv) (recommended) or pip

## Setup Instructions

### Option 1: Using uv (Recommended)

[uv](https://github.com/astral-sh/uv) is a fast Python package installer and resolver that manages dependencies and virtual environments automatically.

#### Install uv

```bash
# On macOS and Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or with pip
pip install uv
```

#### Set up the project

```bash
# Clone the repository
git clone <repository-url>
cd registration

# Install dependencies (uv will automatically create a virtual environment in .venv)
uv sync
```

That's it! uv will:

- Create a virtual environment in `.venv/`
- Install all dependencies from `pyproject.toml`
- Keep everything synchronized

### Option 2: Using pip (Traditional Method)

```bash
# Clone the repository
git clone <repository-url>
cd registration

# Create a virtual environment
python3 -m venv .venv

# Activate the virtual environment
# On macOS/Linux:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
# Or install from pyproject.toml:
pip install -e .
```

## Usage

### Running Scripts with uv

With uv, you don't need to activate the virtual environment manually. Just prefix commands with `uv run`:

#### 1. Global Registration (RANSAC + ICP)

```bash
uv run python global_registration.py \
  --source data/source.ply \
  --target data/target.ply \
  --voxel-size 0.05 \
  -v INFO
```

**Options:**

- `--source`: Path to source point cloud file (required)
- `--target`: Path to target point cloud file (required)
- `--voxel-size`: Voxel size for downsampling (default: 0.05)
- `--max_iter_icp`: Maximum ICP iterations (default: 2000)
- `-v, --verbose`: Logging level - DEBUG, INFO, WARNING, ERROR, CRITICAL (default: INFO)

#### 2. Vanilla ICP Registration

```bash
uv run python icp_vanilla.py \
  --source data/source.ply \
  --target data/target.ply \
  --threshold 0.02 \
  -v INFO
```

**Options:**

- `--source`: Path to source point cloud file (required)
- `--target`: Path to target point cloud file (required)
- `--threshold`: Distance threshold for ICP (default: 0.02)
- `--max_iter_icp`: Maximum ICP iterations (default: 2000)
- `-v, --verbose`: Logging level (default: INFO)

#### 3. Load and Display Point Cloud

```bash
uv run python load_and_display.py \
  --input data/pointcloud.ply \
  -v INFO
```

**Options:**

- `--input`: Path to point cloud file (required)
- `-v, --verbose`: Logging level (default: WARNING)

### Running Scripts with pip/venv

If using the traditional pip method, make sure your virtual environment is activated first:

```bash
# Activate virtual environment
source .venv/bin/activate  # macOS/Linux
# or
.venv\Scripts\activate  # Windows

# Run scripts
python global_registration.py --source data/source.ply --target data/target.ply --voxel-size 0.05
python icp_vanilla.py --source data/source.ply --target data/target.ply --threshold 0.02
python load_and_display.py --input data/pointcloud.ply
```
