import json, glob, os, numpy as np
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
DATA = os.path.join(REPO, "data", "1x_nav")  # 1x_nav frames <traj>/<k>.jpg (v7_visible)
# phase-correlation shift cache written by v7_visible.py and read by v8_visnll.py (was ./vis_shift.pkl in the cwd)
VIS_SHIFT = os.environ.get("VIS_SHIFT_PKL") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "vis_shift.pkl")
RUNS = ["base", "stopft", "de50", "de100", "de200", "de459"]
def load(run):
    fs = sorted(glob.glob(f"{RAW}/{run}_likelihood*/action_likelihood_results/likelihood_rank*.json"))
    R = []
    for f in fs: R += json.load(open(f))
    return R
def text(r, c):
    s = r["scores"][c]; return s.get("action_text") or r["scores"][s["same_text_as"]]["action_text"]
def nll(r, c): return r["scores"][c]["nll_img"]
def act(r, c): return r["scores"][c]["action"]
def nongoal(R): return [r for r in R if not r["target_is_goal"]]
def cci(values, groups, n_boot=4000, seed=0, stat=np.mean):
    v = np.asarray(values, float); g = np.asarray(groups)
    uniq = np.unique(g); idx = {u: np.where(g == u)[0] for u in uniq}; rng = np.random.default_rng(seed); b = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, len(uniq)); b.append(stat(np.concatenate([v[idx[u]] for u in pick])))
    lo, hi = np.percentile(b, [2.5, 97.5]); return float(stat(v)), float(lo), float(hi)
def f3(t, p=2): return f"{t[0]:+.{p}f} [{t[1]:+.{p}f}, {t[2]:+.{p}f}]"
