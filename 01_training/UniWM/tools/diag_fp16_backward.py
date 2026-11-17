"""Locate fp16 gradient overflow in the backward pass (V100 has no bf16).

Runs one training sample through the model exactly like the trainer (fp16 autocast, LoRA, fp32 last-MLP guard),
scales the loss by --scale, and records per-module max |grad_output| / non-finite counts with backward hooks.
Usage (from the repo root):  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python tools/diag_fp16_backward.py --scale 64
"""
import argparse, math, types, torch
from scripts.load_model import load_model
from scripts.load_data import load_data
from scripts.interleaved_tokenized_dataset import InterleaveAnoleTokenizedDataset
from scripts.action_utils import generate_bin_tokens
from peft.utils.other import ModulesToSaveWrapper

ap = argparse.ArgumentParser()
ap.add_argument("--scale", type=float, default=64.0, help="loss scale to test (GradScaler semantics)")
ap.add_argument("--data", default="1x_nav"); ap.add_argument("--data_dir", default="./data")
ap.add_argument("--samples", type=int, default=4)
ap.add_argument("--lm_head_mode", default="full")
args = ap.parse_args()

margs = types.SimpleNamespace(model='anole', model_ckpt=None, image_seq_length=784, precision='fp16', attn_impl='sdpa',
    base_model='leloy/Anole-7b-v0.1-hf', use_memory_bank_inference=False, do_task_level_eval=False, do_train=True,
    do_single_step_eval=False, do_rollout_eval=False, lora_r=16, lora_alpha=32, lora_dropout=0.05,
    lora_targets='q_proj,k_proj,v_proj,o_proj', lm_head_mode=args.lm_head_mode, fp32_mlp_layers='auto')
mp = load_model(margs, None); model, processor = mp['model'], mp['processor']
bins = generate_bin_tokens("dx", -2.46, 2.46, 0.01) + generate_bin_tokens("dy", -2.46, 2.46, 0.01) + generate_bin_tokens("dyaw", -2.82, 2.82, 0.01)
processor.tokenizer.add_tokens(bins, special_tokens=True)
lm_head = model.model.lm_head
if 'ModulesToSaveWrapper' in type(lm_head).__name__:
    model.model.lm_head = lm_head.original_module
    model.model.resize_token_embeddings(len(processor.tokenizer))
    model.model.lm_head = ModulesToSaveWrapper(model.model.lm_head, "default")
else:
    model.model.resize_token_embeddings(len(processor.tokenizer))
for p in model.parameters():
    if p.requires_grad and p.dtype != torch.float32:
        p.data = p.data.float()
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
model.train()

base = model.base_model.model.model  # ChameleonModel
stats = {}
def hook(name):
    def fn(mod, grad_input, grad_output):
        g = grad_output[0]
        if g is None: return
        finite = torch.isfinite(g)
        stats[name] = (float(g[finite].abs().max()) if finite.any() else float('nan'), int((~finite).sum()), str(g.dtype).replace('torch.', ''))
    return fn
for i, layer in enumerate(base.layers):
    layer.register_full_backward_hook(hook(f"L{i:02d}"))
    layer.self_attn.register_full_backward_hook(hook(f"L{i:02d}.attn"))
    layer.mlp.register_full_backward_hook(hook(f"L{i:02d}.mlp"))
base.norm.register_full_backward_hook(hook("final_norm"))
base.embed_tokens.register_full_backward_hook(hook("embed"))

tr = load_data([args.data], args.data_dir, max_train_trajs=20)['train']
ds = InterleaveAnoleTokenizedDataset(dataset=tr, split='train', model=model, processor=processor, input_max_length=3050, label_max_length=850, input_format='anole', interleave=True)
idxs = [next(i for i in range(len(tr)) if tr[i]['train_task'] == t) for t in ('single_step_visualization', 'action_reasoning')][: args.samples]
for idx in idxs:
    stats.clear(); model.zero_grad(set_to_none=True)
    ex = ds[idx]; b = {k: v.cuda() for k, v in ex.items() if isinstance(v, torch.Tensor)}; labels = b.pop('labels')
    shift = labels[:, 1:]; keep = shift != -100; lp = torch.cat([keep, torch.zeros_like(keep[:, :1])], 1)
    with torch.autocast("cuda", dtype=torch.float16):
        logits = model(**b, loss_positions=lp).logits
    loss = torch.nn.functional.cross_entropy(logits.float(), shift[keep])
    (loss * args.scale).backward()
    bad = [(k, v) for k, v in stats.items() if v[1] > 0 or math.isnan(v[0])]
    top = sorted(stats.items(), key=lambda kv: -(kv[1][0] if not math.isnan(kv[1][0]) else 1e30))[:8]
    gn = math.sqrt(sum(float((p.grad.float() ** 2).sum()) for p in model.parameters() if p.grad is not None))
    print(f"[{tr[idx]['train_task']}] scale={args.scale:g} loss={loss.item():.3f} param grad_norm(scaled)={gn:.3g} | non-finite grad_outputs: {[(k, v[1]) for k, v in bad][:6] or 'none'}")
    print("   largest |grad_output| (scaled):", [(k, f"{v[0]:.3g}", v[2]) for k, v in top])
