#!/bin/bash
# Create symlinks from each per-comparison directory to the corresponding
# RANSAC baseline directory, then generate per-ref-voxel-size comparison plots.
#
# Run this script manually after all SLURM jobs have completed.

set -e

source "$(dirname "$0")/config.sh"

for REF_VOXEL_SIZE in "${REF_VOXEL_SIZES[@]}"; do
    COMPARISON_DIR="${ROOT_DIR}/comparison_ref_vs${REF_VOXEL_SIZE}"
    mkdir -p "${COMPARISON_DIR}"

    for VOXEL_SIZE in "${VOXEL_SIZES[@]}"; do
        LINK="${COMPARISON_DIR}/ransac_vs${VOXEL_SIZE}"
        TARGET="../ransac_base/ransac_vs${VOXEL_SIZE}"

        if [[ -L "${LINK}" ]]; then
            echo "Symlink already exists, skipping: ${LINK}"
            continue
        fi

        if [[ -e "${LINK}" ]]; then
            echo "ERROR: path exists and is not a symlink, skipping: ${LINK}" >&2
            continue
        fi

        ln -s "${TARGET}" "${LINK}"
        echo "Created: ${LINK} -> ${TARGET}"
    done

    echo "Generating comparison plots for REF_VOXEL_SIZE=${REF_VOXEL_SIZE}..."
    uv run ./scripts/sequence_registration/plot_localization_comparison.py \
        --input "${COMPARISON_DIR}" \
        --output "${COMPARISON_DIR}/comparison_plots"
done

echo "Done"
