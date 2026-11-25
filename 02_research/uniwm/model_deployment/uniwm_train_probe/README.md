# Local UniWM training-input probe

Capstone Section 7.2.4 (*Training-input probe*). Checks whether the Habitat-adapted local UniWM follows its action even on images it was trained on.

- `uniwm_train_probe.py`: 32 single-action pure-yaw training pairs (16 left, 16 right) plus 8 translation pairs. Image task: the next view under the recorded, zero and sign-reversed commands (same seed). Action task: next-action prediction from the same images.
- `uniwm_multi_action_probe.py`: six training start views that each appear with eight distinct actions.
- `uniwm_train_probe_analyze.py`: CPU analysis of image response (pixel identity, LPIPS, phase-correlation shift) and action accuracy. Outputs are in `results/`.
- `uniwm_train_probe_queue.py`: appends the probe jobs to the GPU queue.

Result used in the capstone: the generated frame shifts in the recorded direction in 25 of 32 yaw pairs, but in 31 of 32 the measured shift is identical under recorded, zero and reversed commands; all eight outputs of each multi-action view are pixel-identical. Paths are relative to the repository root under `external/`.
