#!/bin/bash
# Config variant: process every 10th scan, separate output directory.
# All other parameters are inherited from config_defaults.sh.
#
# Usage:
#   bash experiments/fix_ransac/submit.sh --config config_step10.sh

_VARIANT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${_VARIANT_DIR}/config_defaults.sh"

STEP=10
ROOT_DIR="output/sequence_registration/localization/comparison/comparison_fix_ransac_step10"
RANSAC_BASE_DIR="${ROOT_DIR}/ransac_base"
