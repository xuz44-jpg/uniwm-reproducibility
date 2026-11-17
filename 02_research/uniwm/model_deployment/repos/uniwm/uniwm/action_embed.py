"""Train the INPUT-embedding rows of the action-bin tokens on top of a full-mode adapter.

In the released recipe (lm_head_mode=full) the 1,554 action-bin tokens get trained OUTPUT rows (lm_head in
modules_to_save) but their INPUT embeddings stay at the random initialisation of resize_token_embeddings
(N(0, 0.02^2) per dimension, mutually orthogonal, no order or sign structure). `--train_action_embeddings` wraps
embed_tokens so that only those rows become a trainable fp32 parameter; all other rows stay frozen. The rows are
saved as `action_embed.safetensors` next to the adapter and copied back into the plain embedding table at load time.
"""
import hashlib
import os

import torch
from safetensors.torch import load_file, save_file
from transformers import TrainerCallback

from uniwm.extra_tokens import PartialEmbedding, unwrap

FILE = "action_embed.safetensors"


def action_rows(tokenizer, model):
    """(n_old, n_new): the action-bin tokens must be the last n_new rows of the embedding table."""
    vocab = tokenizer.get_vocab()
    ids = sorted(i for t, i in vocab.items() if "_bin_" in t)
    n_rows = unwrap(model).model.embed_tokens.weight.shape[0]
    n_new = len(ids)
    assert n_new > 0 and ids == list(range(n_rows - n_new, n_rows)), (n_new, ids[:3], ids[-3:], n_rows)
    return n_rows - n_new, n_new


def install(model, n_new):
    hf = unwrap(model)
    if not isinstance(hf.model.embed_tokens, PartialEmbedding):
        hf.model.embed_tokens = PartialEmbedding(hf.model.embed_tokens, n_new)
    return hf.model.embed_tokens


def _digest(t):
    return hashlib.md5(t.detach().float().cpu().numpy().tobytes()).hexdigest()[:12]


def save(model, out_dir):
    emb = unwrap(model).model.embed_tokens
    os.makedirs(out_dir, exist_ok=True)
    w = emb.new_weight.detach().cpu().contiguous()
    save_file({"embed_new": w}, os.path.join(out_dir, FILE))
    return _digest(w)


def load_into_plain(model, ckpt_dir):
    """Copy saved rows into the (unwrapped) embedding table; returns the digest of the loaded rows or None."""
    path = os.path.join(ckpt_dir, FILE)
    if not os.path.exists(path):
        return None
    w = load_file(path)["embed_new"]
    emb = unwrap(model).model.embed_tokens
    tgt = emb.new_weight if isinstance(emb, PartialEmbedding) else emb.weight[-w.shape[0]:]
    with torch.no_grad():
        tgt.copy_(w.to(tgt.device).to(tgt.dtype))
    return _digest(w)


class Callback(TrainerCallback):
    """Writes action_embed.safetensors into every checkpoint-N directory."""

    def __init__(self, model):
        self.model = model

    def on_save(self, args, state, control, **kwargs):
        if not args.should_save:
            return
        ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        if os.path.isdir(ckpt_dir):
            print(f"[action_embed] saved rows to {ckpt_dir} (md5 {save(self.model, ckpt_dir)})", flush=True)
