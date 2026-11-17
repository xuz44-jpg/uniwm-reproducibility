# Supplementary analyses of the action probe (2026-09-28)

These scripts were written during an independent verification of the action-probe recheck (Section 6.5.1).
They run locally with numpy, scipy and PIL (no OpenCV) on `data/1x_nav` and the saved probe outputs.
Their results are recorded in `output/probe_supplementary/verification_journal.jsonl` (one JSON result per
check), and the cached flow estimates are `output/probe_supplementary/pc_flow_all.pkl` and `pc_div_all.pkl`.
Any number the report cites from these scripts is marked "supplementary" and must be re-run from here.

| Script | Purpose |
|---|---|
| `probe_recheck.py` | The recheck itself, run on the server: synthetic known shifts (A1), temporal alignment at lags −1..+1 (A2), calibration clustering (A3), paired LPIPS with trajectory-clustered bootstrap (B). Output: `output/logs/server/probe_recheck.log`. |
| `calib_check.py`, `domain_check.py` | The two earlier checks that showed the 58 px/rad calibration does not hold on the probe frames and that VQ reconstruction is not the cause. Output: `output/logs/server/calib_domain_checks.txt`. |
| `pc_flow.py`, `pc_validate.py`, `pc_calib.py`, `pc_all.py` | An independent phase-correlation shift estimator, validated on synthetic shifts, applied to calibration, test and all 510 trajectories. |
| `pc_lag.py`, `pc_lag2.py`, `lead_labels.py`, `lead_window.py`, `williams.py` | Lag scans beyond ±1, sub-step scans, partial correlations and distributed-lag regressions for the label lead. |
| `pc_joints.py`, `joint_scan.py`, `hip_checks.py`, `onset_hip.py`, `base_only.py` | Timing of joint channels (neck pitch, hip yaw) against image motion; the sign of the hip-yaw term folded into the heading label; base-yaw-only labels. |
| `pc_div.py`, `pc_div_an.py` | A divergence (forward-translation) proxy against displacement labels. |
| `probe_frames.py`, `pc_sheet.py` | Classification of the 23 probe turning frames (onset or mid-turn) and contact sheets. |

## Paths (fixed 2026-09-30)

All paths resolve from the repository root through `_paths.py` (`REPO` = two levels above this folder):
data from `data/1x_nav`, caches read from `output/probe_supplementary/`, probe outputs from `output/figs/server/`.
Environment overrides: `UNIWM_ROOT` (repository root), `PROBE_SUPP_CACHE` (cache directory the analysis scripts
read), `PROBE_SUPP_OUT` (where `pc_all.py` / `pc_div.py` write; default: the cache directory, so set it when you
only want to check that the caches regenerate). The scripts run from any working directory, e.g.
`python3 tools/probe_supplementary/pc_lag.py`.

- `pc_calib.py` uses the archived estimator `pc_flow.pc_shift`; its output matches the numbers recorded for that helper
  (Spearman 0.79 / 0.855, bin medians 4.91 / 16.08 px).
- `probe_recheck.py`, `calib_check.py` and `domain_check.py` need OpenCV, torch, lpips and the model; they point at
  the repository via `_paths.py` but run only on the GPU server.

Re-run of 2026-09-30 (logs, the regenerated caches, and the extra derivations for numbers no archived script prints):
`output/finalize/rerun/probe-supp/`. Both caches regenerate bit-identically (numpy 2.0.2, scipy 1.13.1, Pillow 11.3.0).
Not archived, so not reproducible from here: the NCC, block-matching, Lucas-Kanade and gradient-SSD estimators and
the power simulations.
