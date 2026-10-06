#!/bin/bash
# Evaluate one checkpoint on VSI-Bench / ReVSI, sharded over GPUs.
#
# Run from qwen-vl-finetune/:
#   CKPT=checkpoints/qwen3vl-4b-geometry GPUS="0 1 2 3" bash scripts/eval.sh
#
# Env:
#   CKPT        (required) local checkpoint directory
#   BENCHMARKS  subset of {vsibench,revsi}            (default "vsibench revsi")
#   GPUS        GPU ids to shard across               (default "0")
#   MAX_FRAMES  frames per clip                       (default 32)
#   OUTPUT_DIR  results directory                     (default eval_results/<ckpt name>)
#   LIMIT       cap #samples per benchmark (smoke test)
set -o pipefail

CKPT="${CKPT:?set CKPT to a local checkpoint directory}"
BENCHMARKS="${BENCHMARKS:-vsibench revsi}"
GPUS="${GPUS:-0}"
MAX_FRAMES="${MAX_FRAMES:-32}"
OUTPUT_DIR="${OUTPUT_DIR:-eval_results/$(basename "$CKPT")}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-VL-4B-Instruct}"

export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export VGGT_MULTI_LAYER_INDICES="${VGGT_MULTI_LAYER_INDICES:-23}"
export VGGT_REPO_PATH="${VGGT_REPO_PATH:-$(pwd)/../third_party/vggt}"

ENTRY=qwenvl/eval/evaluate_geometry_bench.py
read -r -a GPU_ARR <<< "$GPUS"
NGPU=${#GPU_ARR[@]}
LIMIT_ARG=""
[ -n "${LIMIT:-}" ] && LIMIT_ARG="--limit ${LIMIT}"
mkdir -p "$OUTPUT_DIR"

for bench in $BENCHMARKS; do
    echo "[$bench] launching $NGPU shard(s)"
    pids=()
    for i in "${!GPU_ARR[@]}"; do
        CUDA_VISIBLE_DEVICES="${GPU_ARR[$i]}" python "$ENTRY" \
            --checkpoint "$CKPT" \
            --benchmark "$bench" \
            --max-num-frame "$MAX_FRAMES" \
            --num-shards "$NGPU" \
            --shard-index "$i" \
            --output-dir "$OUTPUT_DIR" \
            --base-model "$BASE_MODEL" \
            $LIMIT_ARG \
            > "${OUTPUT_DIR}/${bench}_${MAX_FRAMES}f_shard${i}.log" 2>&1 &
        pids+=("$!")
    done
    fail=0
    for pid in "${pids[@]}"; do wait "$pid" || fail=1; done
    if [ "$fail" = 1 ]; then
        echo "[$bench] a shard failed; see ${OUTPUT_DIR}/${bench}_*.log" >&2
        continue
    fi
    python "$ENTRY" \
        --checkpoint "$CKPT" \
        --benchmark "$bench" \
        --max-num-frame "$MAX_FRAMES" \
        --output-dir "$OUTPUT_DIR" \
        --merge-metrics-only
done
