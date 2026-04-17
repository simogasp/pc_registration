#!/bin/bash
# Shared configuration for localization comparison scripts.
# All SLURM scripts and helper scripts source this file.
#
# Edit ONLY this file when changing parameters (e.g. adding a voxel size).
# Changes are automatically picked up by all scripts.

VOXEL_SIZES=(50 100 150 200 300 450)
METHODS=(ransac_icp ransac_gicp)
REF_VOXEL_SIZES=(10 20 50)

INPUT_DIR="data/dataset_real_lidar"
MAP_FILE="output/sequence_registration/fuse/filtered_distance_full/fused_map.ply"
ICP_REFINEMENT_DISTANCE=50.0

ROOT_DIR="output/sequence_registration/localization/comparison/comparison_fix_ransac"
RANSAC_BASE_DIR="${ROOT_DIR}/ransac_base"

# params for the subset of scans to process
START_SCAN=0
END_SCAN=379
STEP=1