# Qwen3-VL-8B yaw-direction probe (2026-11-18)

Protocol: PROTOCOL.md (fixed before the run). Run on one A30 GPU with its own environment under external/vlm-direction
(own venv, bf16, layers beyond 12 GiB offloaded to CPU; about 11.4 GiB GPU, 1.7 s per query).
Files: queries.json, results.jsonl (252 rows), summary.json, run.log, scripts.

| |yaw ±3°|±5°|±7°|±10°|±15°|all|
|---|---|---|---|---|---|---|
|accuracy (n=48 each)|28|29|30|32|34|153/240|
|right turns answered RIGHT (of 24)|24|24|24|24|24|120/120|
|left turns answered LEFT (of 24)|4|5|6|8|10|33/120|
|pairs correct in both orders (of 24)|4|5|6|8|10|33/120|

- Answers are RIGHT for 86% of scored queries; swapping the two images changes the answer in only 33 of 120 pairs.
- Identical pairs (yaw 0 twice): 6 LEFT, 6 RIGHT.
- Greedy answers and first-token logit comparison agree on every query; no unparsed answers.
- Global phase correlation on the same image pairs gets the sign right for 118/120 angle-position pairs.

## All four runs (same 252 queries; runs 2-4 added after seeing run 1, post hoc)

| run | options in prompt | correct of 240 | ±5° of 48 | pairs right in both orders of 120 | rotated pairs answered NONE/UNSURE | identical pairs (12) |
|---|---|---|---|---|---|---|
| 1 | LEFT or RIGHT | 153 | 29 | 33 | n/a | 6 LEFT, 6 RIGHT |
| 2 | RIGHT or LEFT | 201 | 39 | 81 | n/a | 8 LEFT, 4 RIGHT |
| 3 | LEFT, RIGHT, NONE, UNSURE | 153 | 30 | 33 | 0 | 5 NONE, 1 UNSURE, 5 RIGHT, 1 LEFT |
| 4 | UNSURE, NONE, RIGHT, LEFT | 186 | 38 | 68 | 9 (7 at ±3°) | 8 NONE, 4 UNSURE |

Runs 3-4 say the camera "may have rotated". Per-magnitude counts: summary_*.json.
Reading: direction information is present (pairs right in both orders exceed what a biased guesser would get:
about 14 and 29 expected for runs 1 and 2), but the answer depends strongly on the option order, and the model
rarely abstains on rotated pairs even at ±3°.
