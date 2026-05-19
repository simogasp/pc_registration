#!/bin/bash

# This script runs localization against a global map with different voxel size values for both the initial RANSAC registration and the subsequent ICP/GICP refinement.
# For each combination of RANSAC voxel size, it runs localization with RANSAC only.
# Then, for each refinement voxel size, it runs ICP and GICP refinements using the RANSAC poses as input.
# The results are saved in separate directories for each configuration, and comparison plots are generated at the end of each refinement voxel size loop.

echo "Running localization against map with different configurations..."

ROOT_DIR="output/sequence_registration/localization/comparison/comparison_fix_ransac"
MAP="output/sequence_registration/fuse/filtered_distance_full/fused_map.ply"
INPUT="data/dataset_real_lidar"

VOXEL_SIZES=(50 100 150 200 300 450)
# VOXEL_SIZES=(200 450)
REF_VOXEL_SIZES=(10 20 50)
# REF_VOXEL_SIZES=(75 100 150 175 200 250 300)
RANSAC_BASE_DIR="${ROOT_DIR}/ransac_base"

# Run RANSAC once per voxel size (independent of refinement voxel size)
echo "========================================"
echo "Running RANSAC global registration"
echo "========================================"

for VOXEL_SIZE in "${VOXEL_SIZES[@]}"; do
    echo ""
    echo "  VOXEL_SIZE=$VOXEL_SIZE"
    OUTPUT_DIR="${RANSAC_BASE_DIR}/ransac_vs${VOXEL_SIZE}"
    LOG_FILE="${OUTPUT_DIR}/log.log"
    mkdir -p "${OUTPUT_DIR}"
    uv run ./scripts/sequence_registration/localize_against_map.py \
        --input "${INPUT}" \
        --map "${MAP}" \
        --output "${OUTPUT_DIR}/localization_results.json" \
        --voxel-size "${VOXEL_SIZE}" > "${LOG_FILE}"
    uv run ./scripts/sequence_registration/plot_localization_errors.py \
        --input "${OUTPUT_DIR}/localization_results.json" \
        --output "${OUTPUT_DIR}/plots"
done

# For each refinement voxel size, run ICP and GICP using the RANSAC poses as input
for REF_VOXEL_SIZE in "${REF_VOXEL_SIZES[@]}"; do
    echo ""
    echo "========================================"
    echo "Processing REF_VOXEL_SIZE=$REF_VOXEL_SIZE"
    echo "========================================"

    BASE_DIR="${ROOT_DIR}/comparison_ref_vs${REF_VOXEL_SIZE}"
    mkdir -p "${BASE_DIR}"

    for VOXEL_SIZE in "${VOXEL_SIZES[@]}"; do
        echo ""
        echo "  VOXEL_SIZE=$VOXEL_SIZE"
        RANSAC_DIR="${RANSAC_BASE_DIR}/ransac_vs${VOXEL_SIZE}"

        # Link RANSAC results into this comparison directory so the comparison
        # plot script can find all three methods under the same BASE_DIR
        # cp -r "${RANSAC_DIR}" "${BASE_DIR}/ransac_vs${VOXEL_SIZE}"
        LINK="${BASE_DIR}/ransac_vs${VOXEL_SIZE}"
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

        echo "  Running ICP refinement..."
        OUTPUT_DIR="${BASE_DIR}/ransac_icp_vs${VOXEL_SIZE}"
        LOG_FILE="${OUTPUT_DIR}/log.log"
        mkdir -p "${OUTPUT_DIR}"
        uv run ./scripts/sequence_registration/localize_against_map.py \
            --input "${INPUT}" \
            --map "${MAP}" \
            --output "${OUTPUT_DIR}/localization_results.json" \
            --voxel-size "${VOXEL_SIZE}" \
            --estimated-poses "${RANSAC_DIR}/estimated_poses.json" \
            --refine-poses \
            --refinement-voxel-size "${REF_VOXEL_SIZE}" \
            --icp-refinement-distance 50.0 > "${LOG_FILE}"
        uv run ./scripts/sequence_registration/plot_localization_errors.py \
            --input "${OUTPUT_DIR}/localization_results.json" \
            --output "${OUTPUT_DIR}/plots"

        echo "  Running GICP refinement..."
        OUTPUT_DIR="${BASE_DIR}/ransac_gicp_vs${VOXEL_SIZE}"
        LOG_FILE="${OUTPUT_DIR}/log.log"
        mkdir -p "${OUTPUT_DIR}"
        uv run ./scripts/sequence_registration/localize_against_map.py \
            --input "${INPUT}" \
            --map "${MAP}" \
            --output "${OUTPUT_DIR}/localization_results.json" \
            --voxel-size "${VOXEL_SIZE}" \
            --estimated-poses "${RANSAC_DIR}/estimated_poses.json" \
            --refine-poses \
            --refinement-voxel-size "${REF_VOXEL_SIZE}" \
            --icp-refinement-distance 50.0 \
            --use-gicp > "${LOG_FILE}"
        uv run ./scripts/sequence_registration/plot_localization_errors.py \
            --input "${OUTPUT_DIR}/localization_results.json" \
            --output "${OUTPUT_DIR}/plots"
    done

    echo ""
    echo "Generating comparison plots for REF_VOXEL_SIZE=$REF_VOXEL_SIZE..."
    uv run ./scripts/sequence_registration/plot_localization_comparison.py \
        --input "${BASE_DIR}" \
        --output "${BASE_DIR}/comparison_plots"

    echo "REF_VOXEL_SIZE=$REF_VOXEL_SIZE completed!"
done

echo ""
echo "========================================"
echo "All configurations completed!"
echo "========================================"

echo ""
echo "========================================"
echo "Generating final reports..."
echo "========================================"

uv run python scripts/sequence_registration/generate_localization_report.py   --input ${ROOT_DIR}