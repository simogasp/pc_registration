#!/bin/bash
# Configuration dispatcher for the fix_ransac experiment.
# All SLURM workers and helper scripts source this file.
#
# By default, config_defaults.sh is loaded.  To use a different configuration,
# export CONFIG_FILE=<filename> before sourcing this file, or pass --config to
# submit.sh, which sets and exports CONFIG_FILE automatically.
#
# Example:
#   CONFIG_FILE=config_step10.sh bash experiments/fix_ransac/link_ransac_results.sh

_CONFIG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${_CONFIG_DIR}/${CONFIG_FILE:-config_defaults.sh}"