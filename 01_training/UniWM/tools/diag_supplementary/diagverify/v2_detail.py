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
D = os.path.join(REPO, "output", "diag", "raw") + os.sep
R = []
for f in sorted(glob.glob(D + 'base_variance*/variance_decomp_results/variance_rank*.json')):
    R += json.load(open(f))
moves = lambda acts: [a for a in (acts or []) if a not in (None, 'stop')]
sel = {('traj_00302', 6), ('traj_00216', 10), ('traj_00308', 6), ('traj_00346', 15), ('traj_00216', 14), ('traj_00308', 18)}
for r in R:
    if (r['traj'], r['k']) in sel:
        print(r['traj'], r['k'], 'n', r['n'], 'rec_k', np.round(r['recorded_k'], 3), 'rec_next', np.round(r['recorded_next'], 3))
        print('   real', r['real_next'])
        print('   vq  ', r['real_next_vq'])
        for fr in r['imagined']:
            print('   img ', fr)

# permutation test: within each state, pool real+vq move samples, reshuffle, recompute mean var diff (states kept as in analysis)
rng = np.random.default_rng(1)
for comp, ci in (('dx', 0), ('dy', 1), ('dyaw', 2)):
    pairs = []
    for r in R:
        frames = [[a[ci] for a in moves(fr)] for fr in r['imagined'] if fr]
        frames = [f for f in frames if len(f) >= 2]
        ref = [a[ci] for a in moves(r['real_next'])]; refq = [a[ci] for a in moves(r['real_next_vq'])]
        if len(frames) < 2 or len(ref) < 2 or len(refq) < 2:
            continue
        pairs.append((np.array(ref), np.array(refq)))
    obs = np.mean([np.var(a, ddof=1) for a, b in pairs]) - np.mean([np.var(b, ddof=1) for a, b in pairs])
    null = []
    for _ in range(20000):
        d = []
        for a, b in pairs:
            p = rng.permutation(np.concatenate([a, b])); d.append(np.var(p[:len(a)], ddof=1) - np.var(p[len(a):], ddof=1))
        null.append(np.mean(d))
    null = np.array(null)
    print(f"{comp}: observed real-vq {obs:+.5f}; permutation two-sided p = {np.mean(np.abs(null) >= abs(obs)):.3f}")
    # sign test over states with a difference
    dv = [np.var(a, ddof=1) - np.var(b, ddof=1) for a, b in pairs]
    npos = sum(x > 0 for x in dv); nneg = sum(x < 0 for x in dv)
    from math import comb
    m = npos + nneg; k = min(npos, nneg)
    p = min(1, 2 * sum(comb(m, i) for i in range(k + 1)) / 2 ** m)
    print(f"   sign test: {npos} real>vq, {nneg} real<vq, two-sided p={p:.2f}")
