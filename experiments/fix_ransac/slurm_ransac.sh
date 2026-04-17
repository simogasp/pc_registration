#!/bin/bash
# Phase 1: RANSAC global registration.
# Runs once per voxel size, independent of refinement settings.
# Output poses are reused by slurm_refinement.sh.
#
# The array size is computed from config.sh by slurm_submit.sh, which passes
# --array=0-N on the sbatch command line, overriding any #SBATCH --array below.
# Do not submit this directly. Use slurm_submit.sh.

#SBATCH --job-name=localization_ransac
#SBATCH --output=logs/%x_%A/%x_%A_%a.out
#SBATCH --error=logs/%x_%A/%x_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4

#SBATCH --mail-type=END
#SBATCH --mail-user=simone.gasparini@toulouse-inp.fr

cd $SLURM_SUBMIT_DIR
source "${EXPERIMENT_DIR}/config.sh"

VOXEL_SIZE=${VOXEL_SIZES[$SLURM_ARRAY_TASK_ID]}
OUTPUT_DIR="${RANSAC_BASE_DIR}/ransac_vs${VOXEL_SIZE}"
LOG_FILE="${OUTPUT_DIR}/log.log"

echo "Task $SLURM_ARRAY_TASK_ID: VOXEL_SIZE=${VOXEL_SIZE}"
echo "Running on node: $(hostname)"

mkdir -p "${OUTPUT_DIR}" logs

START_TIME=$(date +%s)

uv run ./scripts/sequence_registration/localize_against_map.py \
    --input "${INPUT_DIR}" \
    --map "${MAP_FILE}" \
    --output "${OUTPUT_DIR}/localization_results.json" \
    --start-scan ${START_SCAN} \
    --end-scan ${END_SCAN} \
    --step ${STEP} \
    --voxel-size "${VOXEL_SIZE}" \
    > "${LOG_FILE}"

uv run ./scripts/sequence_registration/plot_localization_errors.py \
    --input "${OUTPUT_DIR}/localization_results.json" \
    --output "${OUTPUT_DIR}/plots"

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINUTES=$(((ELAPSED % 3600) / 60))
SECONDS=$((ELAPSED % 60))

echo "Task $SLURM_ARRAY_TASK_ID completed at: $(date)"
echo "Elapsed time: ${HOURS}h ${MINUTES}m ${SECONDS}s (${ELAPSED} seconds total)"
