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
R = []
for f in sorted(glob.glob(D + 'base_variance*/variance_decomp_results/variance_rank*.json')):
    R += json.load(open(f))
print('states', len(R), 'trajs', len(set(r['traj'] for r in R)))
key = 'stop_next_released'
moves = lambda acts: [a for a in (acts or []) if a not in (None, 'stop')]

# counts of None / failed generation
nfail = sum(1 for r in R for fr in r['imagined'] if fr is None)
nnone = sum(1 for r in R for fr in r['imagined'] if fr for a in fr if a is None)
print('failed imagined frames', nfail, 'malformed action samples on imagined', nnone,
      'malformed real', sum(1 for r in R for a in r['real_next'] if a is None), 'malformed vq', sum(1 for r in R for a in r['real_next_vq'] if a is None))

def per_state(comp):
    ci = {'dx': 0, 'dy': 1, 'dyaw': 2}[comp]
    rows = []
    for r in R:
        frames = [[a[ci] for a in moves(fr)] for fr in r['imagined'] if fr]
        frames = [f for f in frames if len(f) >= 2]
        ref = [a[ci] for a in moves(r['real_next'])]; refq = [a[ci] for a in moves(r.get('real_next_vq'))]
        if len(frames) < 2 or len(ref) < 2 or len(refq) < 2:
            continue
        vj = [np.var(f, ddof=1) for f in frames]; nj = [len(f) for f in frames]
        w = np.mean(vj); braw = np.var([np.mean(f) for f in frames], ddof=1)
        b = braw - np.mean([v / n for v, n in zip(vj, nj)])
        rows.append(dict(traj=r['traj'], k=r['k'], n=r['n'], w=w, b=b, braw=braw, ref=np.var(ref, ddof=1), refq=np.var(refq, ddof=1),
                         dyaw_next=(r['recorded_next'] or [0, 0, 0])[2], dyaw_k=r['recorded_k'][2]))
    return rows

def boot(rows, fn, nb=4000, seed=0):
    g = collections.defaultdict(list)
    for x in rows: g[x['traj']].append(x)
    keys = list(g); rng = np.random.default_rng(seed); out = []
    for _ in range(nb):
        pick = rng.choice(len(keys), len(keys))
        out.append(fn([x for i in pick for x in g[keys[i]]]))
    return np.percentile(out, [2.5, 97.5])

for comp in ('dx', 'dy', 'dyaw'):
    rows = per_state(comp)
    W = np.mean([x['w'] for x in rows]); B = max(np.mean([x['b'] for x in rows]), 0); Braw = np.mean([x['braw'] for x in rows])
    share = lambda rs: max(np.mean([x['b'] for x in rs]), 0) / (np.mean([x['w'] for x in rs]) + max(np.mean([x['b'] for x in rs]), 0))
    lo, hi = boot(rows, share)
    print(f"\n{comp}: n={len(rows)} W={W:.5f} B={B:.5f} Braw={Braw:.5f} share={B/(W+B):.3f} [{lo:.2f},{hi:.2f}] uncorr={Braw/(W+Braw):.3f}")
    # the one-way ANOVA estimator (balanced n=5): sigma_b^2 = (MSB - MSW)/n
    ws = np.array([x['w'] for x in rows]); bs = np.array([x['b'] for x in rows])
    print(f"   states with zero within-variance: {np.mean(ws==0):.2f}; zero between (raw): {np.mean([x['braw']==0 for x in rows]):.2f}")
    # concentration: share of total W and B contributed by top 10% states
    tot = ws + np.maximum(bs, 0)
    order = np.argsort(-tot)
    top = order[:len(rows)//10]
    print(f"   top 10% states hold {tot[top].sum()/tot.sum():.2f} of W+B; share excluding them {max(bs[order[len(rows)//10:]].mean(),0)/(ws[order[len(rows)//10:]].mean()+max(bs[order[len(rows)//10:]].mean(),0)):.2f}")
    # median of per-state share where defined
    ps = [max(x['b'], 0) / (x['w'] + max(x['b'], 0)) for x in rows if x['w'] + max(x['b'], 0) > 0]
    print(f"   per-state share: median {np.median(ps):.2f}, mean {np.mean(ps):.2f} over {len(ps)} states with nonzero total")
    # V2: real vs vq
    ref = np.array([x['ref'] for x in rows]); refq = np.array([x['refq'] for x in rows])
    d = ref - refq
    lo, hi = boot([dict(x, d=x['ref'] - x['refq']) for x in rows], lambda rs: np.mean([x['d'] for x in rs]))
    print(f"   V2 real {ref.mean():.5f} vq {refq.mean():.5f} imagined-within {W:.5f}; diff {d.mean():+.5f} [{lo:+.5f},{hi:+.5f}]")
    print(f"   median real {np.median(ref):.5f} vq {np.median(refq):.5f}; states with real>vq {np.sum(ref>refq)}, real<vq {np.sum(ref<refq)}, equal {np.sum(ref==refq)}")
    o = np.argsort(-ref)[:5]
    print('   top-5 real-var states:', [(rows[i]['traj'], rows[i]['k'], round(ref[i], 5), round(refq[i], 5)) for i in o])
    print(f"   real mean without top-1 state: {np.delete(ref, o[0]).mean():.5f}; without top-3: {np.delete(ref, o[:3]).mean():.5f}; vq w/o its top-3 {np.sort(refq)[:-3].mean():.5f}")
    print(f"   share of real sum held by top-1/top-3 state: {ref[o[0]]/ref.sum():.2f} / {ref[o[:3]].sum()/ref.sum():.2f}")

# Stop rates
print()
for side, getter in (('imagined', lambda r: [a for fr in r['imagined'] if fr for a in fr]), ('real', lambda r: r['real_next'] or []), ('vq', lambda r: r['real_next_vq'] or [])):
    on = [a == 'stop' for r in R if r[key] for a in getter(r)]
    print(side, 'stop on Stop-labelled', np.mean(on), len(on))
# per imagined frame stop counts at stop states
S = [r for r in R if r[key]]
cnt = collections.Counter(sum(a == 'stop' for a in fr) if fr else -1 for r in S for fr in r['imagined'])
print('per-frame #stop out of 5 at Stop states:', sorted(cnt.items()))
per_state = [np.mean([a == 'stop' for fr in r['imagined'] if fr for a in fr]) for r in S]
print('per-state stop rate', sorted(np.round(per_state, 2)))
print('k, n of stop states', collections.Counter(r['n'] - r['k'] for r in S))
