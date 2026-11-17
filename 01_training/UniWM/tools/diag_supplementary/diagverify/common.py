import json, glob, os
import numpy as np
def _repo_root():
    """UniWM repository root: $UNIWM_ROOT if set, else the nearest ancestor of this file that holds output/diag/raw."""
    env = os.environ.get("UNIWM_ROOT")
    if env:
        return env
    d = os.path.dirname(os.path.abspath(__file__))
    while d != os.path.dirname(d):
        if os.path.isdir(os.path.join(d, "output", "diag", "raw")):
            return d
        d = os.path.dirname(d)
    raise SystemExit("cannot locate the UniWM repository (no output/diag/raw above this script); set UNIWM_ROOT")
REPO = _repo_root()
RAW = os.path.join(REPO, "output", "diag", "raw")
def load(pat):
    R = []
    for f in sorted(glob.glob(os.path.join(RAW, pat))):
        R += json.load(open(f))
    return R
def V():
    return load("base_variance*/variance_decomp_results/variance_rank*.json")
def I(run="base"):
    return load(f"{run}_intervention*/action_intervention_results/intervention_rank*.json")
moves = lambda acts: [a for a in (acts or []) if a not in (None, "stop")]
def cluster_boot(stat_fn, groups_items, n_boot=4000, seed=0):
    """groups_items: dict traj -> list of items; stat_fn(list of items) -> float"""
    keys = list(groups_items); rng = np.random.default_rng(seed); out = []
    for _ in range(n_boot):
        pick = rng.choice(len(keys), len(keys))
        items = [it for p in pick for it in groups_items[keys[p]]]
        out.append(stat_fn(items))
    return np.percentile(out, [2.5, 97.5])
