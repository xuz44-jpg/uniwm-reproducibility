# History intervention at new positions: pre-registered confirmation (v2)

Fixed on 2026-11-18 before any new position is drawn, rendered or scored. Any result is reported as is.

## Question

Does changing only the observation history change NWM's measured small-angle action response at
positions never inspected before? The exploratory result (12 reused positions) was: positive
projections rise from static to rotating histories for both released and adapted NWM.

## Positions (drawn without looking at any image or model output)

- Same two public scans as before (apartment_1, skokloster_castle); the adaptation scene is excluded.
- Selection rule copied from the static test, with a new seed 20261118 (+ scene index):
  first finite navmesh random points, camera 1.5 m above the navmesh, yaw seeded uniform,
  at least 1.25 m (Euclidean, floor points) from every chosen point **and from all 12 old positions**,
  at most 20000 attempts per scene.
- Target 6 per scene. If a scene yields fewer than 6, the shortfall is drawn from skokloster_castle with the
  same rule (the castle is processed second). Recorded in the selection files.
- No replacement after rendering or calibration. Positions whose reference calibration fails stay in the
  denominator; their queries are labeled unknown (U).

## Renders and calibration

- Same camera (640x480, HFOV 90, K, depth, C2W kept), same rendering code.
- References at 0, ±3, ±5, ±7 degrees (±3/±7 are needed only by the unchanged reference gate) and history
  views at −9, −6, −2, +9, +6, +2 degrees, all at the same camera center.
- Reference gate: the unchanged RAFT/analytic/known-shift/texture/coverage/separation criteria
  (`probe_habitat_flow_v1.references`).

## Model runs

- NWM only; released EMA weights and the adapted 400-step generator (same files and hashes as before).
- Histories: static (0, 0, 0, 0), left-turn (−9, −6, −2, 0), right-turn (9, 6, 2, 0); all end in the same current view.
- Queries 0, +5, −5 degrees; seeds 202610270, 202610271, 202610272 (paired across histories and angles).
- 2 weights × 3 histories × 12 positions × 3 queries × 3 seeds = 648 new generations, no reuse.
- Same native continuous action, normalized one-frame time 1/128, 250-step sampler, no training.

## Measurement

Unchanged: paired flow D_a, D̂_a against the zero-action output; fixed reference mask; labels A/R/Z/U with
the unchanged thresholds (`habitat_flow_common.direction_label`).

## Pre-registered analysis

Primary endpoint: number of positive projections (A) among the 72 nonzero queries
(12 positions × ±5° × 3 seeds) for each weight × history.

Primary hypotheses (4): for each weight (released, adapted) and each rotating history (left-turn, right-turn),
A(rotating) > A(static).

Primary test: exact one-sided McNemar test on the 72 paired queries (pair = same position, command sign and seed;
outcome A vs not A), Holm–Bonferroni over the 4 hypotheses, family-wise α = 0.05.

Verdict (stated in advance):
- replicated: all 4 Holm-adjusted p < 0.05;
- partially replicated: 1–3 significant, reported per weight and history;
- not replicated: none significant.

Secondary, reported regardless (no correction): position-level sign test (positions with more A under the
rotating history vs fewer, ties dropped); counts per seed and per scene; A split by command sign (A+/A−);
an 80% generated-coverage gate (A with coverage < 0.8 counted as U); raw E_M, gain g_M and zero-action drift;
outputs identical to the paired zero-action output; reference-gate failures.

The queries at one position are not independent; the position-level test is the conservative one.
