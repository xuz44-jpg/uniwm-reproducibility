#!/usr/bin/env bash
set -euo pipefail
export PIP_CACHE_DIR=external/vlm-direction/pip-cache MODELSCOPE_CACHE=external/vlm-direction/ms-cache HF_HOME=external/vlm-direction/hf-home
cd external/vlm-direction/package
nvidia-smi --query-gpu=memory.used,memory.free --format=csv
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True nice -n 10 \
  ../venv/bin/python -I run_qwen.py --model ../models/Qwen3-VL-8B-Instruct 2>&1 | tee -a run.log
../venv/bin/python -I summarize.py
