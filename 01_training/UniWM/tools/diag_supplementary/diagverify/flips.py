import json, glob, os, numpy as np
from math import comb
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
D = os.path.join(REPO, "output", "diag", "raw") + os.sep
R = []
for f in sorted(glob.glob(D + 'base_variance*/variance_decomp_results/variance_rank*.json')):
    R += json.load(open(f))
R = [r for r in R if not r['stop_next_released']]
moves = lambda acts: [a for a in (acts or []) if a not in (None, 'stop')]

def flips(acts, ci, thr=0.05):
    """samples whose component has sign opposite to the state's median sample, both |.|>=thr"""
    m = [a[ci] for a in moves(acts)]
    if not m: return 0, 0
    med = np.median(m)
    if abs(med) < thr: return 0, len(m)
    return sum(1 for v in m if abs(v) >= thr and np.sign(v) != np.sign(med)), len(m)

for comp, ci in (('dy', 1), ('dyaw', 2)):
    for side in ('real_next', 'real_next_vq'):
        f = [flips(r[side], ci) for r in R]
        print(f"{comp} {side:13s}: opposite-sign samples {sum(x for x, _ in f)} of {sum(n for _, n in f)} in {sum(1 for x, _ in f if x)} states")
    fi = [flips(fr, ci) for r in R for fr in r['imagined'] if fr]
    print(f"{comp} imagined     : opposite-sign samples {sum(x for x, _ in fi)} of {sum(n for _, n in fi)} in {sum(1 for x, _ in fi if x)} frames")
    # variance with flips removed
    def var_noflip(acts):
        m = [a[ci] for a in moves(acts)]
        med = np.median(m)
        keep = [v for v in m if not (abs(med) >= 0.05 and abs(v) >= 0.05 and np.sign(v) != np.sign(med))]
        return np.var(keep, ddof=1) if len(keep) >= 2 else np.nan
    for side in ('real_next', 'real_next_vq'):
        print(f"   {side} mean var with opposite-sign samples removed: {np.nanmean([var_noflip(r[side]) for r in R]):.5f}")
# exact binomial for real vs vq dyaw flip counts
