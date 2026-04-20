#!/bin/bash
# Default configuration for the fix_ransac experiment.
# All SLURM scripts and helper scripts load this file through config.sh.
#
# To override individual parameters, create a variant file (e.g. config_step10.sh)
# that sources this file first and then redefines the parameters you want to change.
# Pass the variant to submit.sh via --config <variant_file>.

VOXEL_SIZES=(50 100 150 200 300 450)
METHODS=(ransac_icp ransac_gicp)
REF_VOXEL_SIZES=(10 20 50 75 100 150 175 200 250 300)

INPUT_DIR="data/dataset_real_lidar"
MAP_FILE="output/sequence_registration/fuse/filtered_distance_full/fused_map.ply"
ICP_REFINEMENT_DISTANCE=50.0

ROOT_DIR="output/sequence_registration/localization/comparison/comparison_fix_ransac"
RANSAC_BASE_DIR="${ROOT_DIR}/ransac_base"

# params for the subset of scans to process
START_SCAN=0
END_SCAN=379
STEP=1
