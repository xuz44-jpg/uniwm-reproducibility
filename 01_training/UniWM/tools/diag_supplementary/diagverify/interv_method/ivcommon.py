import json, glob, math, os, pickle
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
DATA = os.path.join(REPO, "data", "1x_nav")
TOL = (0.1, 0.1, 0.2); TURN = 0.05

def load(run):
    R = []
    for f in sorted(glob.glob(f"{RAW}/{run}_intervention*/action_intervention_results/intervention_rank*.json")):
        R += json.load(open(f))
    return R

def _bin(v, w=0.01):
    return round((1 if v >= 0 else -1) * math.floor(abs(v) / w) * w, 4)

def acc(pred, ref, stop):
    if stop or pred == "stop":
        return bool(stop) and pred == "stop"
    return pred is not None and all(abs(p - _bin(r)) <= t for p, r, t in zip(pred, ref, TOL))

def turn_ok(pred, ref):
    return pred not in (None, "stop") and np.sign(pred[2]) == np.sign(ref[2]) and abs(pred[2]) >= TURN / 2

def cluster_ci(values, groups, n_boot=4000, seed=0):
    v = np.asarray(values, float); g = np.asarray(groups)
    if len(v) == 0: return (float("nan"),) * 3
    uniq = np.unique(g); idx = {u: np.where(g == u)[0] for u in uniq}; rng = np.random.default_rng(seed); boots = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, len(uniq)); boots.append(np.concatenate([v[idx[u]] for u in pick]).mean())
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(v.mean()), float(lo), float(hi)

def f3(t, p=3):
    m, lo, hi = t; return f"{m:+.{p}f} [{lo:+.{p}f}, {hi:+.{p}f}]"

def pkl(traj):
    return pickle.load(open(f"{DATA}/{traj}/traj_data.pkl", "rb"))
