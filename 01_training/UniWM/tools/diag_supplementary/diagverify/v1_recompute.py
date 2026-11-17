import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "lik_confounds"))  # helper module this script was written against
from lc_common import *
TURN=0.05
for run in RUNS:
    R = load(run); N = nongoal(R); G=[r for r in R if r["target_is_goal"]]
    T = [r for r in N if abs(r["dyaw_k"]) >= TURN]
    out=[run, len(R), len(N), len(T)]
    for c in ("negated","zero"):
        d=[r for r in T if text(r,c)!=text(r,"recorded")]
        dv=[nll(r,c)-nll(r,"recorded") for r in d]
        out.append(f"{c} {f3(cci(dv,[r['traj'] for r in d]))} sd={np.std(dv):.2f} better={np.mean(np.array(dv)>0):.2f}")
    per_tok_goal=np.mean([nll(r,'recorded') for r in G])/784; per_tok=np.mean([nll(r,'recorded') for r in N])/784
    out.append(f"goal/tok {per_tok_goal:.3f} other/tok {per_tok:.3f}")
    print(*out, sep=" | ")
