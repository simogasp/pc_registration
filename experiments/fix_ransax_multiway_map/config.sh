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
MAP_FILE="output/sequence_registration/multiway/gt_loop500_v300/fused_map_optimized.ply"
GT_POSES_FILE="output/sequence_registration/multiway/gt_loop500_v300/estimated_poses.json"
ICP_REFINEMENT_DISTANCE=50.0

ROOT_DIR="output/sequence_registration/localization/comparison/comparison_fix_ransac_multiway_map"
RANSAC_BASE_DIR="${ROOT_DIR}/ransac_base"
