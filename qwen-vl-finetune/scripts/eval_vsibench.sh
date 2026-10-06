#!/bin/bash
# Evaluate on VSI-Bench. Run from qwen-vl-finetune/:  bash scripts/eval_vsibench.sh
set -o pipefail

# ---- settings ----
CKPT=checkpoints/R2R-Qwen3-VL-8B           # local checkpoint directory
BASE_MODEL=Qwen/Qwen3-VL-8B-Instruct       # base model of the checkpoint (for the processor)
NUM_FRAMES=128                             # paper: 128 for the 8B model, 16 for the 4B model
GPUS="0 1 2 3"                             # questions are split across these GPUs by scene
OUTPUT_DIR=eval_results/$(basename $CKPT)
# ------------------

export PYTHONPATH=$PYTHONPATH:$(pwd)
export OMP_NUM_THREADS=8
export VGGT_REPO_PATH=$(pwd)/../third_party/vggt
ENTRY=qwenvl/eval/evaluate_geometry_bench.py
mkdir -p $OUTPUT_DIR

read -r -a GPU_ARR <<< "$GPUS"
pids=()
for i in "${!GPU_ARR[@]}"; do
    CUDA_VISIBLE_DEVICES=${GPU_ARR[$i]} python $ENTRY \
        --checkpoint $CKPT \
        --base-model $BASE_MODEL \
        --benchmark vsibench \
        --max-num-frame $NUM_FRAMES \
        --num-shards ${#GPU_ARR[@]} \
        --shard-index $i \
        --output-dir $OUTPUT_DIR \
        > $OUTPUT_DIR/vsibench_${NUM_FRAMES}f_shard$i.log 2>&1 &
    pids+=($!)
done
for pid in "${pids[@]}"; do
    wait $pid || { echo "a shard failed; see $OUTPUT_DIR/vsibench_*.log" >&2; exit 1; }
done

python $ENTRY \
    --checkpoint $CKPT \
    --benchmark vsibench \
    --max-num-frame $NUM_FRAMES \
    --output-dir $OUTPUT_DIR \
    --merge-metrics-only
