#!/bin/bash
# Submit the localization job with fine-grained per-voxel dependencies.
# Array sizes are computed dynamically from config.sh.
#
# For each voxel size independently:
#   - One RANSAC task is submitted.
#   - Its refinement batch (N_REF * N_METHODS tasks) is submitted immediately
#     with a dependency only on that single RANSAC task.
# This means refinement for a fast voxel size can start as soon as its RANSAC
# finishes, without waiting for slower (smaller) voxel sizes.
#
# A final symlink job (slurm_link_ransac.sh) runs once all refinement jobs finish.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.sh"

mkdir -p logs

N_VOXEL=${#VOXEL_SIZES[@]}
N_METHODS=${#METHODS[@]}
N_REF=${#REF_VOXEL_SIZES[@]}
N_REFINE_PER_VOXEL=$((N_METHODS * N_REF))

echo "Config: ${N_VOXEL} voxel sizes, ${N_METHODS} methods, ${N_REF} ref voxel sizes"
echo "  RANSAC tasks: ${N_VOXEL} (one per voxel size)"
echo "  Refinement tasks per voxel: ${N_REFINE_PER_VOXEL} (total: $((N_VOXEL * N_REFINE_PER_VOXEL)))"
echo ""

REFINE_JOBS=()

for VOXEL_IDX in $(seq 0 $((N_VOXEL - 1))); do
    VOXEL_SIZE=${VOXEL_SIZES[$VOXEL_IDX]}

    RANSAC_JOB=$(sbatch --parsable --array=${VOXEL_IDX} \
        --export=ALL,EXPERIMENT_DIR="${SCRIPT_DIR}" \
        "${SCRIPT_DIR}/slurm_ransac.sh")
    echo "Submitted RANSAC voxel=${VOXEL_SIZE}: job ${RANSAC_JOB} (task ${VOXEL_IDX})"

    START=$((VOXEL_IDX * N_REFINE_PER_VOXEL))
    END=$((START + N_REFINE_PER_VOXEL - 1))
    REFINE_JOB=$(sbatch --parsable --array=${START}-${END} \
        --export=ALL,EXPERIMENT_DIR="${SCRIPT_DIR}" \
        --dependency=afterok:${RANSAC_JOB} "${SCRIPT_DIR}/slurm_refinement.sh")
    echo "Submitted refinement voxel=${VOXEL_SIZE}: job ${REFINE_JOB} (tasks ${START}-${END})"

    REFINE_JOBS+=("${REFINE_JOB}")
done

# The symlink phase waits for every refinement job to succeed.
REFINE_DEP=$(IFS=:; echo "${REFINE_JOBS[*]}")
LINK_JOB=$(sbatch --parsable --dependency=afterok:${REFINE_DEP} \
    --export=ALL,EXPERIMENT_DIR="${SCRIPT_DIR}" \
    "${SCRIPT_DIR}/slurm_link_ransac.sh")
echo ""
echo "Submitted symlink phase: job ${LINK_JOB} (depends on all refinement jobs)"

echo ""
echo "Monitor progress with:"
echo "  squeue --jobs=$(IFS=,; echo "${REFINE_JOBS[*]}"),${LINK_JOB}"
