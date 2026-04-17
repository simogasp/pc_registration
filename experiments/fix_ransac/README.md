# Experiment: fix_ransac

## Goal

Evaluate whether running RANSAC+FPFH global registration once and reusing the
estimated poses as initialization for ICP and GICP refinement yields better
localization accuracy than running RANSAC alone.

The experiment sweeps over:

- **RANSAC voxel sizes**: `50 100 150 200 300 450` mm — controls the resolution
  at which FPFH features are computed and RANSAC is run.
- **Refinement voxel sizes**: `10 20 50` mm — controls the resolution at which
  ICP/GICP operates (finer than the RANSAC voxel size).
- **Refinement methods**: point-to-plane ICP (`ransac_icp`) and Generalized ICP
  (`ransac_gicp`).

RANSAC is run only once per voxel size, and its output poses are reused for all
refinement configurations, avoiding redundant computation.

## Output structure

```none
output/sequence_registration/localization/comparison/comparison_fix_ransac/
├── ransac_base/
│   ├── ransac_vs50/
│   ├── ransac_vs100/
│   └── ...
└── comparison_ref_vs{REF_VOXEL_SIZE}/
    ├── ransac_vs{VOXEL_SIZE}/        <- symlink to ransac_base/
    ├── ransac_icp_vs{VOXEL_SIZE}/
    ├── ransac_gicp_vs{VOXEL_SIZE}/
    └── comparison_plots/
```

## Files

| File | Purpose |
| --- | --- |
| `config.sh` | Dispatcher: sources the file named by `CONFIG_FILE` (default: `config_defaults.sh`). |
| `config_defaults.sh` | Default parameters (voxel sizes, methods, paths, scan range). Edit to change defaults. |
| `config_step10.sh` | Example variant: same as defaults but `STEP=10` and a separate output directory. |
| `submit.sh` | Entry point: submits all SLURM phases with correct dependencies. Accepts `--config`. |
| `slurm_ransac.sh` | SLURM worker — phase 1: RANSAC global registration. |
| `slurm_refinement.sh` | SLURM worker — phase 2: ICP/GICP refinement. |
| `slurm_link_ransac.sh` | SLURM worker — phase 3: symlinks + comparison plots. |
| `link_ransac_results.sh` | Manual equivalent of phase 3, run locally after SLURM. |

## How to run (SLURM cluster)

From the repository root:

```bash
# Default configuration
bash experiments/fix_ransac/submit.sh

# Custom configuration variant
bash experiments/fix_ransac/submit.sh --config config_step10.sh
```

This submits three job phases with automatic dependencies:

1. **Phase 1** — one RANSAC task per voxel size (6 tasks).
2. **Phase 2** — for each voxel size, refinement starts as soon as its own
   RANSAC task finishes, without waiting for other voxel sizes (6 x 6 tasks).
3. **Phase 3** — once all refinement tasks succeed, symlinks are created and
   comparison plots are generated (1 task).

## How to run (local sequential)

```bash
bash cli_localization_fix_ransac.sh
```

After it finishes, replace the `cp -r` symlinks with actual symlinks and
re-generate comparison plots:

```bash
# Default configuration
bash experiments/fix_ransac/link_ransac_results.sh

# Custom configuration variant
CONFIG_FILE=config_step10.sh bash experiments/fix_ransac/link_ransac_results.sh
```

## Configuration variants

Parameters are split across two layers:

- **`config_defaults.sh`** — all parameters with their default values.  Edit
  this file to change defaults that apply to every run.
- **Variant files** (e.g. `config_step10.sh`) — source `config_defaults.sh`
  and then override only the parameters that differ.  Create a new file for
  each logical variation of the experiment.

Pass the variant to `submit.sh` via `--config <filename>`.  The filename is
exported as `CONFIG_FILE` and picked up automatically by every SLURM worker
through `config.sh`.

To create a new variant (e.g. fewer voxel sizes):

```bash
cp experiments/fix_ransac/config_step10.sh experiments/fix_ransac/config_small.sh
# edit config_small.sh: change VOXEL_SIZES, ROOT_DIR, etc.
bash experiments/fix_ransac/submit.sh --config config_small.sh
```

## Modifying the defaults

Edit `experiments/fix_ransac/config_defaults.sh`.  All scripts pick up changes
automatically on the next run.  No other file needs to be changed.
