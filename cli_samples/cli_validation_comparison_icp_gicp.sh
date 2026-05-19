#!/bin/bash

# Run validation with different step and voxel-size combinations
# Step values: 10, 20, 80
# Voxel sizes: 50, 450

BASE_DIR="output/sequence_registration/validation/comparison/icp_vs_gicp"
DATASET_DIR="data/dataset_real_lidar"
METHODS=(icp gicp)

for METHOD in "${METHODS[@]}"; do
  METHOD_DIR="${BASE_DIR}/${METHOD}"
  mkdir -p "${METHOD_DIR}"

  for STEP in 10 20 80; do
    for VOXEL_SIZE in 50 150 300 450; do
    
      FILENAME="validation_no-gt_${METHOD}_step${STEP}_vs${VOXEL_SIZE}"
      
      OUTPUT_FILE="${METHOD_DIR}/${FILENAME}.json"
      echo "Running validation with method=${METHOD}, step=${STEP}, voxel-size=${VOXEL_SIZE}..."
      uv run ./scripts/sequence_registration/validate_ground_truth.py \
        --input "${DATASET_DIR}/" \
        --output "${OUTPUT_FILE}" \
        --no-gt-init \
        $( [[ "${METHOD}" == "gicp" ]] && echo "--use-gicp" ) \
        --step "${STEP}" \
        --voxel-size "${VOXEL_SIZE}" > "${METHOD_DIR}/${FILENAME}.log"

      uv run ./scripts/sequence_registration/plot_validation_errors.py \
        --input "${OUTPUT_FILE}" \
        --output "${METHOD_DIR}/${FILENAME}_histograms"
    done
  done

  uv run ./scripts/sequence_registration/plot_validation_comparison.py \
    --input "${METHOD_DIR}" \
    --output "${METHOD_DIR}"/comparison_plots
done


echo "All validations complete!"