#!/bin/bash
# Phase 3: Create symlinks and generate comparison plots.
# Symlinks make the RANSAC baseline visible under each per-ref-voxel-size
# comparison directory, then plot_localization_comparison.py is run for each.
#
# This is a single non-array job that runs after all refinement jobs finish.
# Do not submit this directly. Use slurm_submit.sh.

#SBATCH --job-name=localization_link
#SBATCH --output=logs/%x_%A.out
#SBATCH --error=logs/%x_%A.err
#SBATCH --time=00:30:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1

#SBATCH --mail-type=END
#SBATCH --mail-user=simone.gasparini@toulouse-inp.fr

cd $SLURM_SUBMIT_DIR
bash "${EXPERIMENT_DIR}/link_ransac_results.sh"
