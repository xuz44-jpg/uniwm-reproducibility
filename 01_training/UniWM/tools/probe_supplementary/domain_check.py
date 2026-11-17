"""Is the flow calibration comparable with the probe's measurements?

The probe measures flow(VQ-reconstructed current frame, VQ-decoded generated frame); the calibration measures
flow(raw frame k, raw frame k+1). This script measures the same real turns in three domains to check that the
VQ round-trip does not change the scale:
  raw   : flow(raw k, raw k+1)                      (the calibration domain)
  mixed : flow(VQ k, raw k+1)                       (what the probe reported as real_hx)
  vq    : flow(VQ k, VQ k+1)                        (the domain the generated frames live in)
"""
import json, os, sys, pickle
import numpy as np, cv2
from PIL import Image
from _paths import REPO, TOOLS, DATA
sys.path.insert(0, TOOLS)
from action_probe_analysis import flow_proxy

import torch
sys.path.insert(0, REPO); os.chdir(REPO)
from types import SimpleNamespace
from scripts.load_model import load_model

args = SimpleNamespace(model='anole', model_ckpt='output/1x200image_seq_len-784-train-anole-hyper-train1val1lr0.0002-1x_nav-prompt_anole-42',
    image_seq_length=784, precision='fp16', attn_impl='sdpa', base_model='leloy/Anole-7b-v0.1-hf',
    use_memory_bank_inference=False, do_task_level_eval=True, do_train=False, do_single_step_eval=False,
    do_rollout_eval=False, do_action_probe=False, lora_r=16, lora_alpha=32, lora_dropout=0.05,
    lora_targets='q_proj,k_proj,v_proj,o_proj', lm_head_mode='full', fp32_mlp_layers='auto', init_adapter=None, lora_fp32=True)
mp = load_model(args, None); model, proc = mp['model'], mp['processor']

def vq(img):
    inputs = proc(text="<image>", images=img, return_tensors="pt").to(model.device)
    pv = inputs['pixel_values'].to(model.dtype)
    with torch.no_grad():
        toks = model.model.model.get_image_tokens(pv)
        px = model.model.decode_image_tokens(toks)
    arr = proc.postprocess_pixel_values(px)[0].cpu().numpy().transpose(1, 2, 0)
    return Image.fromarray(arr.astype(np.uint8))

man = json.load(open(f"{REPO}/output/base_probeimage_seq_len-784-anole-hyper-train1val1lr0.0002-1x_nav-prompt_anole-42/action_probe_results/manifest_rank0.json"))
rows = []
for m in [x for x in man if x["kind"] == "turn"]:
    traj, k = m["traj"], m["k"]; d = f"{DATA}/{traj}"
    raw_a = Image.open(f"{d}/{k}.jpg").convert("RGB"); raw_b = Image.open(f"{d}/{k+1}.jpg").convert("RGB")
    vq_a = Image.open(m["current"].replace("output/", f"{REPO}/output/")).convert("RGB")
    vq_b = vq(raw_b)
    rows.append((m["recorded"][2], flow_proxy(raw_a, raw_b)[0], flow_proxy(vq_a, raw_b)[0], flow_proxy(vq_a, vq_b)[0]))
    print(f"{traj} k={k} dyaw={rows[-1][0]:+.3f} raw={rows[-1][1]:+6.2f} mixed={rows[-1][2]:+6.2f} vq={rows[-1][3]:+6.2f}", flush=True)
a = np.array(rows)
for i, name in ((1, "raw  "), (2, "mixed"), (3, "vq   ")):
    sl = np.polyfit(a[:, 0], a[:, i], 1)[0]
    sign = np.mean(np.sign(a[:, i]) == np.sign(a[:, 0]))
    print(f"{name}: slope {sl:7.1f} px/rad, sign agreement {sign:.2f}, mean |flow| {np.abs(a[:, i]).mean():.2f}")
