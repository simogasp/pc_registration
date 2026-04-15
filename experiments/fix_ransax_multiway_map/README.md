# Experiment: fix_ransac_multiway_map

## Goal

Evaluate whether running RANSAC+FPFH global registration once and reusing the
estimated poses as initialization for ICP and GICP refinement yields better
localization accuracy than running RANSAC alone.
This instance uses a global map obtained by multiway registration with the ground truth poses instead of just fusing together the input scans using the gt poses.

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
| `config.sh` | Single source of truth for all parameters. Edit only this file. |
| `submit.sh` | Entry point: submits all SLURM phases with correct dependencies. |
| `slurm_ransac.sh` | SLURM worker — phase 1: RANSAC global registration. |
| `slurm_refinement.sh` | SLURM worker — phase 2: ICP/GICP refinement. |
| `slurm_link_ransac.sh` | SLURM worker — phase 3: symlinks + comparison plots. |
| `link_ransac_results.sh` | Manual equivalent of phase 3, run locally after SLURM. |

## How to run (SLURM cluster)

From the repository root:

```bash
bash experiments/fix_ransac/submit.sh
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
bash experiments/fix_ransac/link_ransac_results.sh
```

## Modifying the experiment

Edit `experiments/fix_ransac/config.sh`. All scripts source it automatically.

For example, to add voxel size 600 mm:

```bash
VOXEL_SIZES=(50 100 150 200 300 450 600)
```

No other file needs to be changed.
