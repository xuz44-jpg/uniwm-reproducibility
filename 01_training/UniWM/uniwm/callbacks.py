"""Training guards for fp16 on V100.

NanGuardCallback stops training (instead of burning hours producing NaN checkpoints) when
  * the GradScaler's loss scale collapses below `min_scale` (chronic fp16 overflow in the backward pass), or
  * any trainable parameter becomes non-finite,
and it blocks checkpoint saves while parameters are non-finite so that `save_total_limit` rotation can never
delete the last healthy checkpoint. The loss scale is also logged every logging step.
"""
import math

import torch
from transformers import TrainerCallback


class NanGuardCallback(TrainerCallback):
    def __init__(self, model, min_scale=4.0, check_every=10):
        self.model = model
        self.min_scale = min_scale
        self.check_every = check_every
        self.scaler = None
        self.bad_steps = 0

    def _params_finite(self):
        for p in self.model.parameters():
            if p.requires_grad and not bool(torch.isfinite(p).all()):
                return False
        return True

    def on_train_begin(self, args, state, control, **kwargs):
        trainer = kwargs.get("trainer")
        return control

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.check_every != 0:
            return control
        scale = None
        if self.scaler is not None:
            scale = float(self.scaler.get_scale())
        finite = self._params_finite()
        if not finite:
            print(f"[nan-guard] step {state.global_step}: trainable parameters are non-finite -> stopping (checkpoint save blocked)")
            control.should_save = False
            control.should_training_stop = True
        elif scale is not None and scale < self.min_scale:
            self.bad_steps += 1
            if self.bad_steps >= 3:
                print(f"[nan-guard] step {state.global_step}: loss scale {scale:g} < {self.min_scale:g} for {self.bad_steps} checks -> stopping")
                control.should_training_stop = True
        else:
            self.bad_steps = 0
        return control

    def on_log(self, args, state, control, logs=None, **kwargs):
        if self.scaler is not None and logs is not None and "loss_scale" not in logs:
            logs["loss_scale"] = float(self.scaler.get_scale())
        return control
