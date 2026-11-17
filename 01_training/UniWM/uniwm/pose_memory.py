"""Pose-indexed visual memory for UniWM (innovation I3).

A memory entry is (pose, VQ tokens of a low-resolution frame). Before every action / visualization step the
K entries closest to the current pose are retrieved and appended to the prompt as

    - Memory 1 (dx=+0.21, dy=-0.04, dyaw=+0.12 from current): <mem>
    - Memory 2 (...): <mem>

where ``<mem>`` is a single placeholder token (``<reserved08708>``) that ``splice_memory`` replaces by
``boi + tokens + eoi``. Memory frames are encoded at ``mem_res`` (160 -> 100 tokens, 224 -> 196 tokens) so that
K=2 fits the 4096-token context next to the three 784-token input images.

Poses are (x, y, yaw) in the trajectory's start frame, the same convention as the (dx, dy, dyaw) action labels;
relative poses are therefore plain differences. During training the entries are ground-truth frames of the same
trajectory; in the closed loop they are the start frame plus the model's own imagined observations, with poses
integrated from the predicted actions (no external localisation).
"""
import copy
import math
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch

MEM_PLACEHOLDER = "<reserved08708>"   # single unused special token in the Anole/Chameleon vocabulary (id 8712)

MODES = ("none", "recent", "pose")


def wrap_angle(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def relative_pose(cur: Sequence[float], mem: Sequence[float]) -> Tuple[float, float, float]:
    """(dx, dy, dyaw) of the memory pose relative to the current pose, start-frame axes (action convention)."""
    return float(mem[0] - cur[0]), float(mem[1] - cur[1]), wrap_angle(float(mem[2] - cur[2]))


def retrieve(cur_pose: Sequence[float], poses: Sequence[Sequence[float]], k: int, mode: str = "pose",
             sigma_d: float = 0.5, sigma_yaw: float = 0.5, exclude: Sequence[int] = ()) -> List[int]:
    """Indices of the K memory entries to show for ``cur_pose``.

    ``recent``: the K most recent entries. ``pose``: the K entries with the best
    ``-|Δp|/sigma_d - |Δyaw|/sigma_yaw`` score (ties -> more recent first). Both return the most relevant entry
    first. Entries listed in ``exclude`` (e.g. the current observation itself) are skipped.
    """
    cand = [i for i in range(len(poses)) if i not in set(exclude)]
    if k <= 0 or not cand:
        return []
    if mode == "recent":
        return sorted(cand)[-k:][::-1]
    if mode != "pose":
        raise ValueError(f"unknown memory mode {mode!r}")
    scored = []
    for i in cand:
        dx, dy, dyaw = relative_pose(cur_pose, poses[i])
        scored.append((-(math.hypot(dx, dy) / sigma_d) - abs(dyaw) / sigma_yaw, i))
    scored.sort(key=lambda t: (-t[0], -t[1]))
    return [i for _, i in scored[:k]]


def memory_block(rel_poses: Sequence[Tuple[float, float, float]], text: str = "pose") -> str:
    """Prompt text for the retrieved entries (one placeholder token per frame).

    With no entries the block is empty: the training data (scripts/navigation.py) emits no memory block at all
    for the first frame of a trajectory, and inference must match it. A "none yet" line, never seen in
    training, made the fine-tuned model emit a malformed action at the first closed-loop step of every episode.
    """
    if not rel_poses:
        return ""
    lines = []
    for j, (dx, dy, dyaw) in enumerate(rel_poses, 1):
        if text == "none":   # control: the same frames without their relative poses
            lines.append(f"- Memory {j}: {MEM_PLACEHOLDER} \n")
        else:
            lines.append(f"- Memory {j} (dx={dx:+.2f}, dy={dy:+.2f}, dyaw={dyaw:+.2f} from current): {MEM_PLACEHOLDER} \n")
    return "".join(lines)


class MemoryEncoder:
    """Encodes PIL frames into VQ token ids at a fixed low resolution with the (frozen, fp32) VQ-VAE."""

    def __init__(self, model, processor, res: int = 160):
        self.hf = model.base_model.model if hasattr(model, "base_model") and hasattr(model.base_model, "model") else model
        self.res = int(res)
        self.image_processor = copy.deepcopy(processor.image_processor)
        self.image_processor.size = {"shortest_edge": self.res}
        self.image_processor.crop_size = {"height": self.res, "width": self.res}
        self.tokens_per_frame = (self.res // 16) ** 2

    @torch.no_grad()
    def encode(self, images) -> List[List[int]]:
        if not images:
            return []
        vq = self.hf.model.vqmodel
        dtype = next(vq.parameters()).dtype
        pv = self.image_processor([im.convert("RGB") for im in images], return_tensors="pt")["pixel_values"]
        pv = pv.to(next(vq.parameters()).device, dtype)
        toks = self.hf.model.get_image_tokens(pv)          # [B, tokens_per_frame] BPE ids
        return [t.tolist() for t in toks]


def splice_memory(input_ids: torch.LongTensor, mem_tokens: Sequence[Sequence[int]], marker_id: int,
                  boi: int, eoi: int) -> torch.LongTensor:
    """Replace each placeholder token by ``boi + tokens + eoi``. Works on a [1, L] or [L] tensor."""
    flat = input_ids.reshape(-1).tolist()
    n_marker = flat.count(marker_id)
    if n_marker != len(mem_tokens):
        raise ValueError(f"{n_marker} memory placeholders but {len(mem_tokens)} memory frames")
    out, mi = [], 0
    for t in flat:
        if t == marker_id:
            out.append(boi); out.extend(int(x) for x in mem_tokens[mi]); out.append(eoi); mi += 1
        else:
            out.append(t)
    return torch.tensor(out, dtype=input_ids.dtype, device=input_ids.device).reshape(1, -1)


class PoseMemory:
    """Closed-loop memory bank: (pose, tokens) entries, oldest first."""

    def __init__(self, encoder: MemoryEncoder, k: int = 2, mode: str = "pose", text: str = "pose", ablate_image=None):
        self.encoder, self.k, self.mode, self.text = encoder, int(k), mode, text
        # inference-time ablation: every retrieved frame is replaced by the tokens of one unrelated image
        # (relative-pose text unchanged), to test whether the model uses the memory frames at all
        self._ablate_tokens = encoder.encode([ablate_image])[0] if ablate_image is not None else None
        self.poses: List[Tuple[float, float, float]] = []
        self.tokens: List[List[int]] = []

    def reset(self):
        self.poses, self.tokens = [], []

    def add(self, pose: Sequence[float], image):
        self.poses.append(tuple(float(v) for v in pose))
        self.tokens.append(self.encoder.encode([image])[0])

    def query(self, cur_pose: Sequence[float], exclude_latest: bool = True):
        """Returns (memory_block_text, [token lists]) for the current pose."""
        if self.mode == "none" or self.k <= 0:
            return "", []
        exclude = [len(self.poses) - 1] if (exclude_latest and self.poses) else []
        idx = retrieve(cur_pose, self.poses, self.k, self.mode, exclude=exclude)
        rel = [relative_pose(cur_pose, self.poses[i]) for i in idx]
        toks = [self.tokens[i] for i in idx] if self._ablate_tokens is None else [list(self._ablate_tokens) for _ in idx]
        return memory_block(rel, self.text), toks
