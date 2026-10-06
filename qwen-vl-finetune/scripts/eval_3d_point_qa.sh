#!/bin/bash
# Evaluate one checkpoint on the 3D-Point-QA (ScanNet++) val set, sharded over GPUs.
#
# Run from qwen-vl-finetune/:
#   CKPT=checkpoints/qwen3vl-4b-geometry GPUS="0 1 2 3" bash scripts/eval_3d_point_qa.sh
set -o pipefail

CKPT="${CKPT:?set CKPT to a local checkpoint directory}"
GPUS="${GPUS:-0}"
DATA_ROOT="${DATA_ROOT:-data/3d_point_qa}"
ANNOTATION="${ANNOTATION:-$DATA_ROOT/val.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-eval_results/$(basename "$CKPT")/3d_point_qa}"

export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export VGGT_REPO_PATH="${VGGT_REPO_PATH:-$(pwd)/../third_party/vggt}"

ENTRY=qwenvl/eval/evaluate_3d_point_qa.py
read -r -a GPU_ARR <<< "$GPUS"
NGPU=${#GPU_ARR[@]}
LIMIT_ARG=""
[ -n "${LIMIT:-}" ] && LIMIT_ARG="--limit ${LIMIT}"
mkdir -p "$OUTPUT_DIR"

pids=()
for i in "${!GPU_ARR[@]}"; do
    CUDA_VISIBLE_DEVICES="${GPU_ARR[$i]}" python "$ENTRY" \
        --checkpoint "$CKPT" \
        --annotation-file "$ANNOTATION" \
        --data-root "$DATA_ROOT" \
        --output-dir "$OUTPUT_DIR" \
        --num-shards "$NGPU" \
        --shard-index "$i" \
        $LIMIT_ARG \
        > "${OUTPUT_DIR}/shard${i}.log" 2>&1 &
    pids+=("$!")
done
fail=0
for pid in "${pids[@]}"; do wait "$pid" || fail=1; done
[ "$fail" = 1 ] && { echo "a shard failed; see ${OUTPUT_DIR}/shard*.log" >&2; exit 1; }

python "$ENTRY" --checkpoint "$CKPT" --annotation-file "$ANNOTATION" \
    --output-dir "$OUTPUT_DIR" --merge-metrics-only
