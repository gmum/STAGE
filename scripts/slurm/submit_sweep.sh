#!/bin/bash
# Submits compare.py for every concept of the evaluation with one method, one SLURM job per
# concept. Runs land in <out>/compare_<METHOD>[_<tag>]_<axis>_<concept>. Concepts whose run
# already has a three_d_us.json are skipped, so resubmitting only reruns failed jobs.
#
#   scripts/slurm/submit_sweep.sh METHOD [--axis AXIS] [--tag TAG] [--env VAR=V[,VAR=V...]]
#                                 [--mode MODE] [--out DIR] [--save-glb]
#
#   --axis      only this axis (shape, material or object)
#   --tag       suffix of the run names, to keep configurations apart
#   --env       hyperparameter overrides exported to the jobs, named as in
#               unlearning_methods/<METHOD>/parameters.py
#   --mode      edit these stages (S, L or SL) on every axis instead of the axis's own mode
#   --out       results directory relative to the repository root (default: results)
#   --save-glb  also export every generated asset as a textured .glb
#
# Example, STAGE at alpha = 1 with r = 32:
#   scripts/slurm/submit_sweep.sh STAGE --tag alpha1_rank32 --env STAGE_ALPHA=1.0,STAGE_RANK=32 \
#       --out results/ablation_hparams/STAGE

set -euo pipefail

[ $# -ge 1 ] || { sed -n '2,19p' "$0" >&2; exit 1; }
METHOD="$1"
shift

AXES=(shape material object)
TAG=""
ENV_ARGS=()
EXTRA_ARGS=()
OUT="results"
while [ $# -gt 0 ]; do
    case "$1" in
        --axis) AXES=("$2"); shift 2 ;;
        --tag) TAG="_$2"; shift 2 ;;
        --env) ENV_ARGS=(--export="$2"); shift 2 ;;
        --mode) EXTRA_ARGS+=(--mode "$2"); shift 2 ;;
        --out) OUT="$2"; shift 2 ;;
        --save-glb) EXTRA_ARGS+=(--save-glb); shift ;;
        *) echo "unknown option: $1" >&2; exit 1 ;;
    esac
done

declare -A CONCEPTS=(
    [shape]="round_circular cubic_square cylinder cellular_latice ring_torus"
    [material]="wood metal glass stone ceramic"
    [object]="car cat chair table teddy_bear"
)

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source scripts/slurm/cluster.env
mkdir -p logs

for axis in "${AXES[@]}"; do
    for concept in ${CONCEPTS[$axis]}; do
        out_dir="$OUT/compare_${METHOD}${TAG}_${axis}_${concept}"
        if [ -f "$out_dir/three_d_us.json" ]; then
            echo "skip (complete): $out_dir"
            continue
        fi
        sbatch --time=04:00:00 --job-name="${METHOD}${TAG}-${concept}" \
            --output="logs/%x_%j.out" --error="logs/%x_%j.err" "${ENV_ARGS[@]}" \
            scripts/slurm/job.slurm compare.py \
            --method "$METHOD" --axis "$axis" --concept "$concept" --output-dir "$out_dir" "${EXTRA_ARGS[@]}"
    done
done
