#!/bin/bash

# Run validation with different step and voxel-size combinations
# Step values: 10, 20, 80
# Voxel sizes: 50, 450

for STEP in 10 20 80; do
  for VOXEL_SIZE in 50 450; do
    OUTPUT_FILE="output/sequence_registration/validation/test_gicp/validation_no-gt_gicp_step${STEP}_vs${VOXEL_SIZE}.json"
    echo "Running validation with step=${STEP}, voxel-size=${VOXEL_SIZE}..."
    uv run ./scripts/sequence_registration/validate_ground_truth.py \
      --input data/dataset_real_lidar/ \
      --output "${OUTPUT_FILE}" \
      --no-gt-init \
      --use-gicp \
      --step "${STEP}" \
      --log-level WARNING \
      --voxel-size "${VOXEL_SIZE}"
  done
done

echo "All validations complete!"