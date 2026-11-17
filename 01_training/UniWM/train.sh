#!/usr/bin/env bash
# 8x V100-32GB, fp16 AMP + sdpa attention, DDP (per-GPU batch 1 x grad-acc from cfg -> global batch 8, same as the paper)
# Usage: bash train.sh [extra train.py args]        (NGPU=6 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 bash train.sh ... to use 6 GPUs)
NGPU=${NGPU:-8}
set -euo pipefail
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # less fragmentation; peak is ~31.5 GB of 32 GB
torchrun --nproc_per_node "$NGPU" --master_port=${MASTER_PORT:-20001} train.py \
    --model anole \
    --data go_stanford \
    --data_dir ./data \
    --decoder_type anole \
    --image_seq_length 784 \
    --input_format anole \
    --output output \
    --note train \
    --report_to none \
    --do_train \
    --precision fp16 \
    --attn_impl sdpa \
    --lora_r 16 --lora_alpha 32 \
    --lm_head_mode full \
    --grad_acc 1 \
    --logging_steps 10 --save_steps 500 --save_total_limit 3 \
    "$@"
