#!/usr/bin/env bash
# Runs after the option-order control finishes; one job on the GPU at a time.
set -uo pipefail
cd external/vlm-direction/package
while pgrep -f "run_qwen.py --model" >/dev/null; do sleep 15; done
for v in 4way 4way_rev; do
  nvidia-smi --query-gpu=memory.free --format=csv,noheader
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True nice -n 10 ../venv/bin/python -I run_qwen.py \
    --model ../models/Qwen3-VL-8B-Instruct --prompt-file prompt_$v.txt --labels LEFT,RIGHT,NONE,UNSURE \
    --out results_$v.jsonl > run_$v.log 2>&1 && \
  ../venv/bin/python -I summarize_4way.py results_$v.jsonl summary_$v.json > /dev/null
done
echo queue-finished
