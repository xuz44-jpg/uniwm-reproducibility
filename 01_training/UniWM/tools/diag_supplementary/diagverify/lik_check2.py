import numpy as np, sys, os
sys.argv=['x']
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lik_check.py')).read().split("DATA = {run")[0])
DATA = {run: load(run) for run in RUNS}

def balanced_ci(T, d, n_boot=4000, seed=1):
    """alignment = (mean over true-left + mean over true-right)/2 and left-text bias = (L-R)/2, cluster bootstrap by trajectory."""
    trajs = sorted(set(r['traj'] for r in T)); rng = np.random.default_rng(seed)
    by = {t: [(np.sign(r['dyaw_k']), x) for r, x in zip(T, d) if r['traj'] == t] for t in trajs}
    def est(items):
        L = [x for s, x in items if s > 0]; R = [x for s, x in items if s < 0]
        return (np.mean(L) + np.mean(R)) / 2, (np.mean(L) - np.mean(R)) / 2
    a0, b0 = est([i for t in trajs for i in by[t]]); A, B = [], []
    for _ in range(n_boot):
        items = [i for t in rng.choice(trajs, len(trajs)) for i in by[t]]
        try:
            a, b = est(items); A.append(a); B.append(b)
        except Exception:
            pass
    return (a0, *np.nanpercentile(A, [2.5, 97.5])), (b0, *np.nanpercentile(B, [2.5, 97.5]))

for run in RUNS:
    R = [r for r in DATA[run] if not r['target_is_goal']]
    for lab, lo in (("turning>=0.05", 0.05), ("turning>=0.15", 0.15)):
        T = [r for r in R if abs(r['dyaw_k']) >= lo]
        d = [nll(r, 'negated') - nll(r, 'recorded') for r in T]
        a, b = balanced_ci(T, d)
        nl = sum(r['dyaw_k'] > 0 for r in T); nr = sum(r['dyaw_k'] < 0 for r in T)
        print(f"{run:7s} {lab}: n={len(T)} (L {nl}/R {nr}, trajs L {len(set(r['traj'] for r in T if r['dyaw_k']>0))} R {len(set(r['traj'] for r in T if r['dyaw_k']<0))}) "
              f"balanced alignment {f3(a)}; left-text preference {f3(b)}")
# lag-candidate distances on mid-turn frames
R = [r for r in DATA['base'] if not r['target_is_goal']]
M = [r for r in R if r["dyaw_km1"] is not None and abs(r["dyaw_k"]) >= 0.05 and abs(r["dyaw_km1"]) >= 0.05 and np.sign(r["dyaw_k"]) == np.sign(r["dyaw_km1"])
     and all(c in r['scores'] for c in ('prev2','prev1','next1'))]
for c in ('prev2','prev1','next1','negated'):
    dd = np.array([np.abs(np.array(r['scores'][c]['action']) - np.array(r['scores']['recorded']['action'])) for r in M])
    same_dyaw_tok = np.mean([txt(r,c).split('dyaw: ')[1] == txt(r,'recorded').split('dyaw: ')[1] for r in M])
    print(f"mid-turn {c}: mean |diff| dx {dd[:,0].mean():.3f} dy {dd[:,1].mean():.3f} dyaw {dd[:,2].mean():.3f}; same dyaw token {same_dyaw_tok:.2f}")
print('mean |dyaw_k| on mid-turn', np.mean([abs(r['dyaw_k']) for r in M]), 'on turning', np.mean([abs(r['dyaw_k']) for r in R if abs(r['dyaw_k'])>=0.05]))
# fraction of goal frames etc
T = [r for r in R if abs(r['dyaw_k']) >= 0.05]
print('turning frames: left share', np.mean([r['dyaw_k']>0 for r in T]))
