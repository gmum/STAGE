#!/bin/bash -l
#SBATCH --export=NONE
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G

# Runs `python scripts/<script> <args>` on a GPU node with the settings of cluster.env.
# Submitted from the repository root by run.sh and submit_sweep.sh, which set the time limit,
# the job name and the log files in logs/. The logs are moved into the run's --output-dir
# when the job ends.

set -euo pipefail

SCRIPT="$1"
shift

cd "$SLURM_SUBMIT_DIR"
source scripts/slurm/cluster.env

LOGS=("logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out" "logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.err")
OUTPUT_DIR=""
args=("$@")
for i in "${!args[@]}"; do
    if [ "${args[$i]}" = "--output-dir" ]; then
        OUTPUT_DIR="${args[$((i + 1))]}"
        break
    fi
done

move_logs() {
    [ -z "$OUTPUT_DIR" ] && return 0
    mkdir -p "$OUTPUT_DIR"
    mv -f "${LOGS[@]}" "$OUTPUT_DIR/" 2>/dev/null || true
}
trap move_logs EXIT

for path in ${MODULE_PATHS:-}; do
    module use "$path"
done
if [ -n "${MODULES:-}" ]; then
    module load $MODULES
fi
source "$VENV/bin/activate"
if [ -n "${EBROOTCUDA:-}" ]; then
    export CUDA_HOME=$EBROOTCUDA
fi
export ATTN_BACKEND=flash_attn
export SPCONV_ALGO=native

python "scripts/$SCRIPT" "$@"
