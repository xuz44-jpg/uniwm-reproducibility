import numpy as np, sys, os
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lik_check.py')).read().split("DATA = {run")[0])
DATA = {run: load(run) for run in RUNS}
for run in RUNS:
    R = [r for r in DATA[run] if not r['target_is_goal']]
    S = [r for r in R if abs(r['dyaw_k']) < 0.02 and txt(r,'other_traj') != txt(r,'recorded')]
    out = []
    for lab, cond in (("other |dyaw|<0.02", lambda a: abs(a[2]) < 0.02), ("other |dyaw|>=0.05", lambda a: abs(a[2]) >= 0.05), ("other |dyaw|>=0.10", lambda a: abs(a[2]) >= 0.10)):
        Q = [r for r in S if cond(r['scores']['other_traj']['action'])]
        d = [nll(r,'other_traj') - nll(r,'recorded') for r in Q]
        out.append(f"{lab}: n={len(Q)} {f3(ccI(d, [r['traj'] for r in Q]))}")
    # turning frames: other_traj split by |dyaw_other|
    T = [r for r in R if abs(r['dyaw_k']) >= 0.10]
    for lab, cond in (("turn>=0.10, other |dyaw|<0.02", lambda a: abs(a[2]) < 0.02), ("turn>=0.10, other |dyaw|>=0.10 opposite sign", None)):
        if cond is None:
            Q = [r for r in T if abs(r['scores']['other_traj']['action'][2]) >= 0.10 and np.sign(r['scores']['other_traj']['action'][2]) != np.sign(r['dyaw_k'])]
        else:
            Q = [r for r in T if cond(r['scores']['other_traj']['action'])]
        d = [nll(r,'other_traj') - nll(r,'recorded') for r in Q]
        out.append(f"{lab}: n={len(Q)} {f3(ccI(d, [r['traj'] for r in Q]))}")
    print(run, "straight frames other_traj-rec |", " | ".join(out))
