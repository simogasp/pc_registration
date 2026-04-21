#!/bin/bash
# Config variant: use a map built from the first half of the sequence, and validate the localization all the scans of the sequence
#
# Usage:
#   bash experiments/fix_ransac/submit.sh --config config_half_map_full_validation.sh

_VARIANT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${_VARIANT_DIR}/config_defaults.sh"

MAP_FILE="output/sequence_registration/fuse/filtered_distance_0-204/fused_map.ply"

ROOT_DIR="output/sequence_registration/localization/comparison/comparison_fix_ransac_half_map_full_validation"
RANSAC_BASE_DIR="${ROOT_DIR}/ransac_base"