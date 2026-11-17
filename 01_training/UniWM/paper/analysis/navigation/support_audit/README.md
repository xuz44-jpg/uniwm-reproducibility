# Frozen-row support audit — 2026-10-30

This is a local CPU reaggregation of existing results. It does not change frozen
experiments, thresholds, images, or protocols, and adds no world-model prediction.

## Reproduce

From this directory:

```bash
python3 audit_support_from_rows.py --self-test
python3 audit_support_from_rows.py --root ../recompute_inputs --out output/support
```

The Python standard library is sufficient. `--root` points to the input tree
with the recorded JSON and source files; `--out` changes the destination. The script reads JSON and
source-code literals only; it never imports the frozen execution scripts or
loads a model. Source-file SHA256 values and the analyzer's own SHA256 are stored
in `support_audit.json`. `artifact_manifest.json` hashes the output files.

## What was audited

- 216 NWM context conditions: two weights × three histories × 12 positions ×
  three actions (0 and ±5°). There are 144 nonzero queries and 72 zero queries.
- Original confirmation conditions: three models × 252, with 216 nonzero
  queries per model, 72 at each absolute input angle 3°, 5°, or 7°.
- These make 972 **row appearances**, including the 36 adapted/static NWM
  conditions reused in the context experiment. Their stored metrics reproduce
  exactly. This is not a count of new predictions or independent observations.
- There are only two scenes and 12 positions. The context experiment has one
  paired seed; the original confirmation uses three. Seeds are not independent
  scenes.

## Three distinct supports

Let `M` be the fixed per-position mask from the real reference images. The same
seven-angle reference mask is used for all histories and weights at that
position. Let `Va` and `V0` be the forward/backward-valid generated-action and
generated-zero flow masks. Then:

| Stored quantity | Pixel support | Inclusion rule |
|---|---|---|
| `effect` EPE/gain | all of `M` | real-reference gate; direction Unknown remains included |
| `direction` | `M & Va & V0` | requires frozen generated-coverage and pixel-count thresholds |
| `direction.response` EPE/gain | `M & Va & V0` | saved after coverage passes, including mixed-direction Unknown |
| `zero_drift` | all of `M` | no generated-valid drift gate |

The flow effect is `F(I0, generated_action) - F(I0, generated_zero)`, compared
with `F(I0, real_action) - F(I0, real_zero)`. All branches are anchored at the
same real current image. Pixel gain is a projection coefficient, not an estimated
physical yaw angle.

The stored valid-region columns are useful diagnostics, but their query sets
and pixel sets vary across arms. Even `context_query_pairs.csv`'s valid-region
deltas use only the intersection of available **queries**, not a common set of
reliable **pixels**. Do not treat these columns as a repaired primary metric.

## Files

- `support_audit.json`: complete definitions, source SHA256, limitations, grouped
  distributions, paired contrasts, and stored-scalar threshold sensitivity.
- `context_support_table.csv`: compact six-row NWM table with complete A/R/Z/U,
  coverage, raw/valid EPE denominators, gain mean/median, and Unknown contribution.
- `baseline_small_angles_table.csv`: nine rows for the original three models at
  absolute input angles 3°, 5°, and 7°, each with complete denominator 72.
- `context_grouped.csv`: NWM arm/scene/position/signed-angle summaries.
- `baseline_angle_response.csv`: original model, angle, scene, and position
  summaries, including reference-gate counts and valid-response counts.
- `context_query_pairs.csv`: all 324 matched condition pairs across nine
  contrasts, 36 pairs each. Delta means **test minus reference**.
- `context_paired_aggregates.csv`: all, per-scene, per-position, and per-angle
  paired statistics. Both mean and median position changes are retained.
- `stored_rows.csv`: scalar audit inputs for all 972 row appearances.
- `unknown_queries.csv`: all Unknown rows with reason, coverage, and raw/valid
  metrics where available. Unknown is never relabelled measured near-zero.
- `threshold_scalar_sensitivity.csv`: one-factor classifications reconstructed
  from existing scalar summaries. The frozen classifier exactly reproduces all
  792 nonzero row appearances, including reused conditions.

## Supported sensitivity analysis and missing measurements

The scalar sensitivity table changes one of the coverage threshold (0.2, 0.35,
0.5, 0.8), near-zero threshold (0.5, 1, 2 pixels), or direction-consensus threshold
(0.6, 0.7, 0.8), while retaining the original pixel-validity masks. It does not
rerun RAFT or measure estimator accuracy. Coverage is only maintained or
tightened: rows failing the original 0.2 gate do not store the necessary motion
summaries to evaluate a relaxed gate. This analysis must not be called a
pixel-level robustness evaluation.

The following remain **pending**, and cannot be recovered from aggregate rows:

1. Metrics on a common generated-reliable pixel support across compared arms.
2. Zero-action drift evaluated only on validated generated pixels.
3. Sensitivity to forward/backward consistency thresholds, texture selection,
   or other changes to pixel-validity masks.

They require existing per-pixel flow/valid caches, excluded from this retrieved
bundle. No new world-model generation is intrinsically needed if those caches
still exist, but this audit neither retrieves nor recomputes them. Real-reference
calibration alone does not validate flow on distorted generated images.

## Interpretation limits

The raw full-mask EPE/gain/drift columns are **unvalidated raw-flow diagnostics**
where generated correspondence fails. Their means cannot establish accurate
camera motion, physical yaw error, or a unique failure cause. Removing Unknown
does not fix the support mismatch, and pooled Unknown + near-zero is not a
measured-static category.

Counterexamples remain visible: adapted NWM right history improves raw EPE at
7/12 positions and has a negative median paired position change, despite a
positive overall mean change. Raw gain medians are much lower than their means.
The old/new tensor-hash encodings also differ, so paired seeds alone do not
constitute independent proof of actual cross-format noise-tensor equality.
