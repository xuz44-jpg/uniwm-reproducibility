"""Repository paths for the probe_supplementary scripts.

Every path resolves from the repository root, so the scripts run from any working directory.
Overrides: UNIWM_ROOT (repository root), PROBE_SUPP_CACHE (where the analysis scripts read
pc_flow_all.pkl / pc_div_all.pkl), PROBE_SUPP_OUT (where pc_all.py / pc_div.py write them).
"""
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
# this copy lives in <repo>/tools/probe_supplementary/
REPO = os.environ.get("UNIWM_ROOT") or os.path.abspath(os.path.join(_HERE, "..", ".."))
DATA = os.path.join(REPO, "data", "1x_nav")
CACHE = os.environ.get("PROBE_SUPP_CACHE") or os.path.join(REPO, "output", "probe_supplementary")
OUT = os.environ.get("PROBE_SUPP_OUT") or CACHE
FIGS = os.path.join(REPO, "output", "figs", "server")
TOOLS = os.path.join(REPO, "tools")

if not os.path.isdir(DATA):
    raise SystemExit(f"_paths: data directory not found: {DATA} (set UNIWM_ROOT)")
