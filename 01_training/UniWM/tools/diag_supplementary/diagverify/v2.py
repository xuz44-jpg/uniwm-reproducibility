import sys; sys.path.insert(0, __file__.rsplit('/',1)[0])
from common import *
R = V()
def rows(ci):
    out = []
    for r in R:
        frames = [[a[ci] for a in moves(fr)] for fr in r["imagined"] if fr]
        frames = [f for f in frames if len(f) >= 2]
        ref = [a[ci] for a in moves(r["real_next"])]; refq = [a[ci] for a in moves(r.get("real_next_vq"))]
        if len(frames) < 2 or len(ref) < 2 or len(refq) < 2: continue
        out.append(dict(traj=r['traj'], k=r['k'], ref=np.var(ref, ddof=1), refq=np.var(refq, ddof=1), w=np.mean([np.var(f, ddof=1) for f in frames]),
                        refv=ref, refqv=refq, rec=r['recorded_k'], recn=r['recorded_next']))
    return out
for comp, ci in (("dx",0),("dy",1),("dyaw",2)):
    S = rows(ci); G = {}
    for s in S: G.setdefault(s['traj'], []).append(s)
    d = np.array([s['ref']-s['refq'] for s in S])
    lo, hi = cluster_boot(lambda SS: np.mean([s['ref']-s['refq'] for s in SS]), G)
    lo2, hi2 = cluster_boot(lambda SS: np.mean([s['w']-s['ref'] for s in SS]), G)
    lo3, hi3 = cluster_boot(lambda SS: np.mean([s['w']-s['refq'] for s in SS]), G)
    ref = np.array([s['ref'] for s in S]); refq = np.array([s['refq'] for s in S])
    # sign test on states where they differ
    npos = int((d>0).sum()); nneg = int((d<0).sum())
    print(f"{comp}: n={len(S)} real {ref.mean():.5f} vq {refq.mean():.5f} diff {d.mean():+.5f} [{lo:+.5f},{hi:+.5f}]; real>vq on {npos}, real<vq on {nneg}, equal {len(d)-npos-nneg}; "
          f"median real {np.median(ref):.5f} vq {np.median(refq):.5f}")
    print(f"     imagined-within minus real {np.mean([s['w']-s['ref'] for s in S]):+.5f} [{lo2:+.5f},{hi2:+.5f}]; minus vq {np.mean([s['w']-s['refq'] for s in S]):+.5f} [{lo3:+.5f},{hi3:+.5f}]")
    for name, arr in (("real", ref), ("vq", refq)):
        o = np.argsort(-arr)[:3]
        print(f"     {name}: top-3 states carry {arr[o].sum()/arr.sum():.2f} of the sum; top-1 {arr[o[0]]/arr.sum():.2f}")
        for i in o:
            s = S[i]; print(f"        {s['traj']} k={s['k']} var {arr[i]:.5f} real samples {[round(x,2) for x in s['refv']]} vq samples {[round(x,2) for x in s['refqv']]} rec_k {round(s['rec'][ci],3)} rec_next {None if s['recn'] is None else round(s['recn'][ci],3)}")
    # drop top-1 / top-3 of the real ones
    for m in (1, 3):
        keep = np.argsort(-ref)[m:]
        print(f"     drop top-{m} real-variance states: real {ref[keep].mean():.5f} vq {refq[keep].mean():.5f}")
