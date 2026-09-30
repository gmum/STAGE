#!/bin/bash
# Submits one generate.py or compare.py run. Paths in the script arguments are relative to
# the repository root.
#
#   scripts/slurm/run.sh <generate|compare> [script args...]
#
# The time limit is $TIME (default 08:00:00). Other sbatch options can be set through their
# input environment variables, e.g. SBATCH_QOS.
#
# Example:
#   scripts/slurm/run.sh compare --method STAGE --axis material --concept wood \
#       --output-dir results/compare_STAGE_material_wood

set -euo pipefail

if [ $# -lt 1 ] || { [ "$1" != generate ] && [ "$1" != compare ]; }; then
    echo "Usage: $0 <generate|compare> [script args...]" >&2
    exit 1
fi
SCRIPT="$1"
shift

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source scripts/slurm/cluster.env
mkdir -p logs
sbatch --time="${TIME:-08:00:00}" --job-name="$SCRIPT" --output="logs/%x_%j.out" --error="logs/%x_%j.err" \
    scripts/slurm/job.slurm "$SCRIPT.py" "$@"
