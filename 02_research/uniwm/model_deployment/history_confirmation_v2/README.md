# Pre-registered NWM history replication at new positions

Capstone Section 7.2.6 and Table 7.7. `PROTOCOL.md` was fixed before any new position was drawn, rendered or scored; `results/protocol.json` and `results/execution_freeze.json` record the frozen protocol and the source hashes.

- Twelve new positions in the two evaluation scans (six per scan), at least 1.25 m from the earlier twelve positions and from each other.
- NWM released and adapted weights, static / left-turn / right-turn histories, 0° and ±5° queries, three paired seeds: 648 generations.
- Primary test: exact one-sided McNemar test per rotating-history comparison, Holm correction over the four comparisons.

Scripts (in `model_deployment/`, next to the shared modules they import):

| Script | Role |
|---|---|
| `hc2_common.py` | Protocol constants (angles, seeds, histories) and shared helpers |
| `hc2_render.py` | Draws the new positions and renders references and history views (Habitat environment) |
| `hc2_reference.py` | Reference calibration gate (`probe_habitat_flow_v1`) |
| `hc2_run.py` | NWM generation |
| `hc2_score.py` | Flow scoring and A/R/Z/U labels |
| `hc2_analyze.py` | McNemar/Holm, position sign tests, 80% coverage gate; writes `results/summary.json` |
| `hc2_queue.py` | Freezes the protocol and code hashes, then appends the jobs to the GPU job queue (`--study`, `--root`) |

`results/rows.json` holds the per-query records and `results/summary.json` the numbers reported in the capstone. Server paths are written relative to the repository root under `external/`; the recorded hashes refer to the files as run.
