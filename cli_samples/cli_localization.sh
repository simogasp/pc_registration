#!/bin/bash

# This script runs localization against the a global map with different voxel sizes values.
# For each voxel size, it runs localization with RANSAC only, RANSAC with ICP refinement, and RANSAC with GICP refinement.
# The results are saved in separate directories for each configuration, and comparison plots are generated at the end.

echo "Running localization against map with different configurations..."

BASE_DIR="output/sequence_registration/localization/comparison/"
VOXEL_SIZES=(50 100 150 200 300 450)

for VOXEL_SIZE in "${VOXEL_SIZES[@]}"; do
    echo ""
    echo "========================================"
    echo "Processing VOXEL_SIZE=$VOXEL_SIZE"
    echo "========================================"
    
    echo "Running RANSAC only..."
    OUTPUT_DIR="${BASE_DIR}ransac_vs${VOXEL_SIZE}"
    LOG_FILE="$OUTPUT_DIR/log.log"
    mkdir -p $OUTPUT_DIR
    uv run ./scripts/sequence_registration/localize_against_map.py --input data/dataset_real_lidar --map output/sequence_registration/fuse/filtered_distance_full/fused_map.ply --output $OUTPUT_DIR/localization_results.json --voxel-size $VOXEL_SIZE > $LOG_FILE
    uv run ./scripts/sequence_registration/plot_localization_errors.py --input $OUTPUT_DIR/localization_results.json --output $OUTPUT_DIR/plots
    
    echo "Running RANSAC with ICP refinement..."
    OUTPUT_DIR="${BASE_DIR}ransac_icp_vs${VOXEL_SIZE}"
    LOG_FILE="$OUTPUT_DIR/log.log"
    mkdir -p $OUTPUT_DIR
    uv run ./scripts/sequence_registration/localize_against_map.py --input data/dataset_real_lidar --map output/sequence_registration/fuse/filtered_distance_full/fused_map.ply --output $OUTPUT_DIR/localization_results.json --voxel-size $VOXEL_SIZE --refine-poses --icp-refinement-distance 50.0 > $LOG_FILE
    uv run ./scripts/sequence_registration/plot_localization_errors.py --input $OUTPUT_DIR/localization_results.json --output $OUTPUT_DIR/plots
    
    echo "Running RANSAC with GICP refinement..."
    OUTPUT_DIR="${BASE_DIR}ransac_gicp_vs${VOXEL_SIZE}"
    LOG_FILE="$OUTPUT_DIR/log.log"
    mkdir -p $OUTPUT_DIR
    uv run ./scripts/sequence_registration/localize_against_map.py --input data/dataset_real_lidar --map output/sequence_registration/fuse/filtered_distance_full/fused_map.ply --output $OUTPUT_DIR/localization_results.json --voxel-size $VOXEL_SIZE --refine-poses --icp-refinement-distance 50.0 --use-gicp > $LOG_FILE
    uv run ./scripts/sequence_registration/plot_localization_errors.py --input $OUTPUT_DIR/localization_results.json --output $OUTPUT_DIR/plots
done

echo ""
echo "========================================"
echo "All configurations completed!"
echo "========================================"

echo ""
echo "========================================"
echo "Generating comparison plots..."
echo "========================================"
uv run ./scripts/sequence_registration/plot_localization_comparison.py --input $BASE_DIR --output ${BASE_DIR}comparison_plots