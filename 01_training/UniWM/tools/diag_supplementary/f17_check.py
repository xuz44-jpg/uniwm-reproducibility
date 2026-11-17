import json, glob, math, os, sys
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
sys.path.insert(0, os.path.join(REPO, "tools"))
from action_diag_analysis import _bin
for run, key in (("base", "stop_label_released"), ("stopft", "stop_label_dist")):
    R = []
    for f in sorted(glob.glob(os.path.join(REPO, "output", "diag", "raw", f"{run}_intervention*", "action_intervention_results", "intervention_rank*.json"))):
        d = json.load(open(f)); R += d if isinstance(d, list) else d.get("results", d)
    rows = [st for r in R for st in r["steps"]]
    mv = [st for st in rows if not st[key] and st["recorded"] is not None and st.get("gen") and st.get("real")]
    ok = lambda p: p not in (None, "stop")
    g = [st["gen"][0] for st in mv]; re = [st["real"][0] for st in mv]
    lab = [[_bin(v) for v in st["recorded"]] for st in mv]
    print(run, "move steps n =", len(mv), "trajs", len(R))
    print("  gen dx %.4f  label dx %.4f  real dx %.4f" % (np.mean([p[0] for p in g if ok(p)]), np.mean([l[0] for l in lab]), np.mean([p[0] for p in re if ok(p)])))
    print("  gen |dy| %.4f  label |dy| %.4f  real |dy| %.4f" % (np.mean([abs(p[1]) for p in g if ok(p)]), np.mean([abs(l[1]) for l in lab]), np.mean([abs(p[1]) for p in re if ok(p)])))
    # all non-Stop predictions on all steps
    allg = [st["gen"][0] for st in rows if st.get("gen") and ok(st["gen"][0])]
    print("  all-steps non-Stop gen preds n=%d: dx %.4f |dy| %.4f" % (len(allg), np.mean([p[0] for p in allg]), np.mean([abs(p[1]) for p in allg])))
