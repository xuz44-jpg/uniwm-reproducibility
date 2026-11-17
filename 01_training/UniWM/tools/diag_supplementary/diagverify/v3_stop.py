import json, glob, os, numpy as np, collections
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

def load(pat):
    R = []
    for f in sorted(glob.glob(D + pat)):
        R += json.load(open(f))
    return R

def cluster_ci(vals, groups, nb=4000, seed=0):
    v = np.asarray(vals, float); g = np.asarray(groups); u = np.unique(g); idx = {x: np.where(g == x)[0] for x in u}
    rng = np.random.default_rng(seed); b = []
    for _ in range(nb):
        pick = rng.choice(u, len(u)); b.append(np.concatenate([v[idx[x]] for x in pick]).mean())
    return v.mean(), *np.percentile(b, [2.5, 97.5])

# variance run: Stop at the Stop-labelled (k = n-2) states, per trajectory cluster CI
V = load('base_variance*/variance_decomp_results/variance_rank*.json')
S = [r for r in V if r['stop_next_released']]
vals, grp = [], []
for r in S:
    for fr in r['imagined']:
        for a in fr:
            vals.append(a == 'stop'); grp.append(r['traj'])
print('variance run, imagined 1 step, Stop-labelled: %.3f [%.2f, %.2f] n=%d' % (*cluster_ci(vals, grp), len(vals)))
fr_level = [np.mean([a == 'stop' for a in fr]) >= 0.5 for r in S for fr in r['imagined']]
print('  imagined frames with majority Stop: %d / %d' % (sum(fr_level), len(fr_level)))
# Stop on real/vq at the k = n-2 state? (the current frame, not the next) -- not stored; skip

for model in ('base', 'stopft'):
    I = load(f'{model}_intervention*/action_intervention_results/intervention_rank*.json')
    key = 'stop_label_released' if model == 'base' else 'stop_label_dist'
    print(f'\n{model} intervention ({key})')
    for name in ('real', 'vq', 'gen'):
        for where, cond in (('goal frame s=n-1', lambda st, n: st['s'] == n - 1), ('Stop-labelled s<n-1', lambda st, n: st['s'] < n - 1)):
            g_, a_, grp_ = [], [], []
            for r in I:
                for st in r['steps']:
                    if st[key] and cond(st, r['n']) and st.get(name):
                        g_.append(st[name][0] == 'stop')
                        for a in st[name][1:]:
                            a_.append(a == 'stop'); grp_.append(r['traj'])
            if g_:
                m, lo, hi = cluster_ci(a_, grp_)
                print(f'  {name:4s} {where:22s}: greedy Stop {np.mean(g_):.2f} (n={len(g_)}), sampled Stop {m:.2f} [{lo:.2f},{hi:.2f}] (n={len(a_)})')
    if model == 'stopft':
        # depth of the dist-labelled steps
        print('  dist-labelled step offsets n-1-s:', sorted(collections.Counter(r['n'] - 1 - st['s'] for r in I for st in r['steps'] if st[key]).items()))
