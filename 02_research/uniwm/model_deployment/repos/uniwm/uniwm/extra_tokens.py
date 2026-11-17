"""Train only the rows of `embed_tokens` / `lm_head` that belong to newly added tokens (the action bins).

Why: the released recipe puts the whole `lm_head` into PEFT `modules_to_save` (275M trainable params, a 1.1 GB fp32
copy plus 2.2 GB of Adam state per GPU) and never trains the *input* embeddings of the new tokens at all. Here the base
rows stay frozen in the base dtype (fp16 on V100) and only the new rows are trainable fp32 parameters (about 13M
params for 1554 action tokens). They are saved next to the LoRA adapter as `extra_tokens.safetensors`.
"""
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from transformers import TrainerCallback

EXTRA_FILE = "extra_tokens.safetensors"


def unwrap(model):
    """AnoleforConditionalGeneration under a PeftModel, or the model itself."""
    if hasattr(model, "base_model") and hasattr(model.base_model, "model") and hasattr(model.base_model.model, "lm_head"):
        return model.base_model.model
    return model


class PartialEmbedding(nn.Module):
    """Frozen base embedding + trainable fp32 rows for ids >= n_old."""

    def __init__(self, base: nn.Embedding, n_new: int):
        super().__init__()
        self.base = base
        self.base.weight.requires_grad_(False)
        self.num_embeddings = base.num_embeddings
        self.embedding_dim = base.embedding_dim
        self.padding_idx = base.padding_idx
        self.n_new = int(n_new)
        self.n_old = base.num_embeddings - self.n_new
        self.new_weight = nn.Parameter(base.weight[self.n_old:].detach().float().clone())

    @property
    def weight(self):  # shape / dtype checks elsewhere look at .weight
        return self.base.weight

    def forward(self, input_ids):
        # branch-free: no host<->device sync per call (a sync per generated token halves decoding speed),
        # and new_weight is always part of the graph (DDP with find_unused_parameters=False stays happy)
        out = self.base(input_ids)
        idx_new = (input_ids - self.n_old).clamp_(min=0, max=self.n_new - 1)
        new = F.embedding(idx_new, self.new_weight).to(out.dtype)
        return torch.where((input_ids >= self.n_old).unsqueeze(-1), new, out)


class PartialLMHead(nn.Module):
    """Frozen base lm_head + trainable fp32 output rows for the last n_new vocabulary entries."""

    def __init__(self, base: nn.Linear, n_new: int):
        super().__init__()
        self.base = base
        self.base.weight.requires_grad_(False)
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.n_new = int(n_new)
        self.n_old = base.out_features - self.n_new
        self.new_weight = nn.Parameter(base.weight[self.n_old:].detach().float().clone())

    @property
    def weight(self):
        return self.base.weight

    def forward(self, hidden_states):
        h = hidden_states.to(self.base.weight.dtype)          # fp16 base rows; under autocast this is a no-op
        old = F.linear(h, self.base.weight[: self.n_old])
        new = F.linear(h, self.new_weight.to(h.dtype))        # fp32 master rows, computed in the base dtype
        return torch.cat([old, new.to(old.dtype)], dim=-1)


def install(model, n_new: int):
    """Wrap embed_tokens / lm_head of the (possibly PEFT-wrapped) model. Idempotent."""
    hf = unwrap(model)
    if isinstance(hf.model.embed_tokens, PartialEmbedding):
        return hf
    hf.model.embed_tokens = PartialEmbedding(hf.model.embed_tokens, n_new)
    hf.lm_head = PartialLMHead(hf.lm_head, n_new)
    return hf


def n_new_in(ckpt_dir: str):
    path = os.path.join(ckpt_dir, EXTRA_FILE)
    if not os.path.exists(path):
        return 0
    return load_file(path)["embed_new"].shape[0]


def save(model, out_dir: str):
    hf = unwrap(model)
    os.makedirs(out_dir, exist_ok=True)
    save_file({
        "embed_new": hf.model.embed_tokens.new_weight.detach().cpu().contiguous(),
        "lm_head_new": hf.lm_head.new_weight.detach().cpu().contiguous(),
    }, os.path.join(out_dir, EXTRA_FILE))


def load(model, ckpt_dir: str) -> bool:
    path = os.path.join(ckpt_dir, EXTRA_FILE)
    if not os.path.exists(path):
        return False
    sd = load_file(path)
    hf = unwrap(model)
    with torch.no_grad():
        hf.model.embed_tokens.new_weight.copy_(sd["embed_new"].to(hf.model.embed_tokens.new_weight.device))
        hf.lm_head.new_weight.copy_(sd["lm_head_new"].to(hf.lm_head.new_weight.device))
    return True


class ExtraTokensCallback(TrainerCallback):
    """Writes extra_tokens.safetensors into every checkpoint-N directory (the PEFT adapter save ignores these params)."""

    def __init__(self, model):
        self.model = model

    def on_save(self, args, state, control, **kwargs):
        if not args.should_save:
            return
        ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        if os.path.isdir(ckpt_dir):
            save(self.model, ckpt_dir)
