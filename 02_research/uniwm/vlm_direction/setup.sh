#!/usr/bin/env bash
# Isolated environment and weights under external/vlm-direction; touches no existing venv or service.
set -euo pipefail
export PIP_CACHE_DIR=external/vlm-direction/pip-cache MODELSCOPE_CACHE=external/vlm-direction/ms-cache HF_HOME=external/vlm-direction/hf-home
ROOT=external/vlm-direction
cd "$ROOT"
df -h /root
[ -d venv ] || python3 -m venv venv
venv/bin/pip install -q --upgrade pip
venv/bin/pip install -q "torch==2.5.1" "transformers>=4.57,<5" accelerate pillow modelscope
venv/bin/python -c "import torch,transformers;print(torch.__version__,torch.cuda.is_available(),transformers.__version__)"
[ -f models/Qwen3-VL-8B-Instruct/config.json ] || \
  venv/bin/modelscope download --model Qwen/Qwen3-VL-8B-Instruct --local_dir models/Qwen3-VL-8B-Instruct
du -sh models/Qwen3-VL-8B-Instruct
