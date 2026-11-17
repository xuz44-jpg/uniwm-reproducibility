import sys; sys.path.insert(0, __file__.rsplit('/',1)[0])
from common import *
R = V()
def clean(vals, on):
    if not on: return vals
    v = np.array(vals); med = np.median(v)
    if abs(med) < 0.05: return vals
    keep = ~((np.sign(v) != np.sign(med)) & (np.abs(v) >= 0.05))
    return v[keep].tolist()
for drop in (False, True):
    W = []; B = []; RF = []; RQ = []
    for r in R:
        frames = [clean([a[2] for a in moves(fr)], drop) for fr in r["imagined"] if fr]
        frames = [f for f in frames if len(f) >= 2]
        ref = clean([a[2] for a in moves(r["real_next"])], drop); refq = clean([a[2] for a in moves(r.get("real_next_vq"))], drop)
        if len(frames) < 2 or len(ref) < 2 or len(refq) < 2: continue
        vj = [np.var(f, ddof=1) for f in frames]; nj = [len(f) for f in frames]
        W.append(np.mean(vj)); B.append(np.var([np.mean(f) for f in frames], ddof=1) - np.mean([v/n for v, n in zip(vj, nj)]))
        RF.append(np.var(ref, ddof=1)); RQ.append(np.var(refq, ddof=1))
    w, b = np.mean(W), max(np.mean(B), 0)
    print(f"dyaw {'without' if drop else 'with'} opposite-sign samples: n={len(W)} within {w:.5f} between {b:.5f} share {b/(w+b):.2f}; real {np.mean(RF):.5f} vq {np.mean(RQ):.5f}")
# intervention: are gen Stops frame-determined? (greedy + 5 samples on the same gen frame)
Ib = I("base")
cnt = {}
for r in Ib:
    for st in r['steps']:
        acts = st['gen'] or []
        if not acts: continue
        ns = sum(a == 'stop' for a in acts)
        if ns: cnt[ns] = cnt.get(ns, 0) + 1
print("base intervention gen steps with >=1 Stop among greedy+5 samples, by number of Stops (of 6):", dict(sorted(cnt.items())))
# direct pose channel of per-step action-sampling noise (closed loop adds dx,dy to x,y; yaw is not used for position)
w = {'dx': 0.00041, 'dy': 0.00055}; bimg = {'dx': 0.00024, 'dy': 0.00031}; T = 25; N = 24
pe = np.sqrt(T * (w['dx'] + w['dy']) / 2); pi_ = np.sqrt(T * (bimg['dx'] + bimg['dy']) / 2)
print(f"direct channel, independent steps: per-episode SD of final distance from action sampling {pe:.3f}, mean over {N} episodes {pe/np.sqrt(N):.3f}; "
      f"from one-step image variance if independent across steps {pi_:.3f} -> {pi_/np.sqrt(N):.3f}; combined {np.hypot(pe, pi_)/np.sqrt(N):.3f}")
sds = {'base': 0.064, 'stop': 0.053, 'mem-pose': 0.030, 'mem-recent': 0.072}
pooled = np.sqrt(np.mean([s**2 for s in sds.values()]))
print(f"pooled 3-run SD of mean final distance (4 sampled models) {pooled:.3f}; share of variance from direct action channel {(pe/np.sqrt(N))**2/pooled**2:.2f}")
# chi-square 95% CI for a 2-df and 1-df SD estimate (multipliers), using tabulated quantiles
q = {2: (0.0506, 7.378), 1: (0.000982, 5.024)}
for df, (lo, hi) in q.items():
    print(f"  SD from {df+1} runs ({df} df): 95% CI = [{np.sqrt(df/hi):.2f}, {np.sqrt(df/lo):.2f}] x estimate")
print("greedy base SD 0.103 from 2 runs:", [round(0.103*np.sqrt(1/5.024),3), round(0.103*np.sqrt(1/0.000982),2)], "; greedy stop SD 0.006:", [round(0.006*np.sqrt(1/5.024),4), round(0.006*np.sqrt(1/0.000982),3)])
