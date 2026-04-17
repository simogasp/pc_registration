#!/bin/bash
# Phase 2: ICP/GICP refinement using pre-estimated RANSAC poses.
# Must run after slurm_ransac.sh has completed (all tasks).
#
# The array size is computed from config.sh by slurm_submit.sh, which passes
# --array=0-N on the sbatch command line, overriding any #SBATCH --array below.
#
# Task layout (N_VOXEL * N_REF * N_METHODS total):
#   task_id = VOXEL_IDX * (N_REF * N_METHODS) + REF_IDX * N_METHODS + METHOD_IDX
#
# Tasks for a given voxel size form a contiguous block of size (N_REF * N_METHODS),
# which allows slurm_submit.sh to submit each voxel's refinement batch independently
# with --array=START-END and a per-voxel dependency on its own RANSAC task.
#
# Do not submit this directly. Use slurm_submit.sh.

#SBATCH --job-name=localization_refine
#SBATCH --output=logs/%x_%A/%x_%A_%a.out
#SBATCH --error=logs/%x_%A/%x_%A_%a.err
#SBATCH --time=01:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4

#SBATCH --mail-type=END
#SBATCH --mail-user=simone.gasparini@toulouse-inp.fr

cd $SLURM_SUBMIT_DIR
source "${EXPERIMENT_DIR}/config.sh"

N_METHODS=${#METHODS[@]}
N_REF=${#REF_VOXEL_SIZES[@]}
STRIDE=$((N_REF * N_METHODS))

VOXEL_IDX=$((SLURM_ARRAY_TASK_ID / STRIDE))
WITHIN=$((SLURM_ARRAY_TASK_ID % STRIDE))
REF_IDX=$((WITHIN / N_METHODS))
METHOD_IDX=$((WITHIN % N_METHODS))

VOXEL_SIZE=${VOXEL_SIZES[$VOXEL_IDX]}
METHOD=${METHODS[$METHOD_IDX]}
REF_VOXEL_SIZE=${REF_VOXEL_SIZES[$REF_IDX]}

echo "Task $SLURM_ARRAY_TASK_ID: VOXEL_SIZE=${VOXEL_SIZE} METHOD=${METHOD} REF_VOXEL_SIZE=${REF_VOXEL_SIZE}"
echo "Running on node: $(hostname)"

RANSAC_DIR="${RANSAC_BASE_DIR}/ransac_vs${VOXEL_SIZE}"
OUTPUT_DIR="${ROOT_DIR}/comparison_ref_vs${REF_VOXEL_SIZE}/${METHOD}_vs${VOXEL_SIZE}"
LOG_FILE="${OUTPUT_DIR}/log.log"

mkdir -p "${OUTPUT_DIR}" logs

START_TIME=$(date +%s)

CMD="uv run ./scripts/sequence_registration/localize_against_map.py \
    --input ${INPUT_DIR} \
    --map ${MAP_FILE} \
    --output ${OUTPUT_DIR}/localization_results.json \
    --voxel-size ${VOXEL_SIZE} \
    --estimated-poses ${RANSAC_DIR}/estimated_poses.json \
    --start-scan ${START_SCAN} \
    --end-scan ${END_SCAN} \
    --step ${STEP} \
    --refine-poses \
    --refinement-voxel-size ${REF_VOXEL_SIZE} \
    --icp-refinement-distance ${ICP_REFINEMENT_DISTANCE}"

if [[ "$METHOD" == "ransac_gicp" ]]; then
    CMD="$CMD --use-gicp"
fi

echo "Running: $CMD"
eval $CMD > "${LOG_FILE}"

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