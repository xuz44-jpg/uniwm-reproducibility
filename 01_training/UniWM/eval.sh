#!/usr/bin/env bash
# Evaluation on V100 (fp16 + sdpa). Pass the checkpoint dir as the first argument.
# Usage: bash eval.sh /path/to/checkpoint [--do_single_step_eval | --do_task_level_eval [--use_memory_bank_inference] | --do_rollout_eval]
set -euo pipefail
CKPT=${1:?checkpoint path required}; shift
MODE=${1:---do_task_level_eval}; shift || true
export TOKENIZERS_PARALLELISM=false
torchrun --nproc_per_node 1 --master_port=20002 train.py \
    --model anole \
    --model_ckpt "$CKPT" \
    --data go_stanford \
    --data_dir ./data \
    --decoder_type anole \
    --image_seq_length 784 \
    --input_format anole \
    --output output \
    --note eval \
    --report_to none \
    --precision fp16 \
    --attn_impl sdpa \
    "$MODE" "$@"
