import sys; sys.path.insert(0, __file__.rsplit('/',1)[0])
from common import *
def flips(vals):
    """samples whose sign is opposite to the median sign, both |.|>=0.05"""
    v = np.array(vals); med = np.median(v)
    if abs(med) < 0.05: return 0
    return int(((np.sign(v) != np.sign(med)) & (np.abs(v) >= 0.05)).sum())
for run in ("base", "stopft"):
    R = I(run)
    for comp, ci in (("dx",0),("dy",1),("dyaw",2)):
        rows = []
        for r in R:
            for st in r["steps"]:
                a = [x[ci] for x in moves((st.get("real") or [])[1:])]; b = [x[ci] for x in moves((st.get("vq") or [])[1:])]
                g = [x[ci] for x in moves((st.get("gen") or [])[1:])]
                if len(a) >= 2 and len(b) >= 2:
                    rows.append(dict(traj=r['traj'], ra=np.var(a, ddof=1), rb=np.var(b, ddof=1), fa=flips(a), fb=flips(b), na=len(a), nb=len(b),
                                     rg=np.var(g, ddof=1) if len(g) >= 2 else np.nan))
        G = {}
        for x in rows: G.setdefault(x['traj'], []).append(x)
        lo, hi = cluster_boot(lambda S: np.mean([x['ra']-x['rb'] for x in S]), G)
        ra = np.array([x['ra'] for x in rows]); rb = np.array([x['rb'] for x in rows]); rg = np.array([x['rg'] for x in rows])
        extra = ""
        if ci == 2:
            extra = f"; sign-flip samples: real {sum(x['fa'] for x in rows)}/{sum(x['na'] for x in rows)}, vq {sum(x['fb'] for x in rows)}/{sum(x['nb'] for x in rows)}"
        print(f"{run} {comp}: steps {len(rows)}; 5-sample var real {ra.mean():.5f} vq {rb.mean():.5f} diff {np.mean(ra-rb):+.5f} [{lo:+.5f},{hi:+.5f}]; "
              f"real>vq {int((ra>rb).sum())} real<vq {int((ra<rb).sum())}; gen {np.nanmean(rg):.5f}{extra}")
# variance run sign flips
R = V()
fa = fb = na = nb = 0; fi = ni = 0
for r in R:
    a = [x[2] for x in moves(r['real_next'])]; b = [x[2] for x in moves(r['real_next_vq'])]
    if len(a) >= 2: fa += flips(a); na += len(a)
    if len(b) >= 2: fb += flips(b); nb += len(b)
    for fr in r['imagined']:
        c = [x[2] for x in moves(fr)]
        if len(c) >= 2: fi += flips(c); ni += len(c)
print(f"variance run dyaw sign-flip samples: real {fa}/{na}, vq {fb}/{nb}, imagined {fi}/{ni}")
# matched replicate: variance-run real_next at (traj,k) vs intervention base real at s=k+1 (same frame, same model)
Ib = {r['traj']: {st['s']: st for st in r['steps']} for r in I("base")}
m = []
for r in R:
    st = Ib.get(r['traj'], {}).get(r['k'] + 1)
    if st is None: continue
    a = [x[2] for x in moves(r['real_next'])]; b = [x[2] for x in moves(r['real_next_vq'])]
    c = [x[2] for x in moves((st['real'] or [])[1:])]; d = [x[2] for x in moves((st['vq'] or [])[1:])]
    if min(len(a), len(b), len(c), len(d)) >= 2:
        m.append((np.var(a, ddof=1), np.var(b, ddof=1), np.var(c, ddof=1), np.var(d, ddof=1)))
m = np.array(m)
print(f"matched frames n={len(m)}: dyaw var real(variance run) {m[:,0].mean():.5f}, vq(variance run) {m[:,1].mean():.5f}, real(intervention) {m[:,2].mean():.5f}, vq(intervention) {m[:,3].mean():.5f}")
print("  corr of per-frame variance between runs: real-real %.2f, vq-vq %.2f, real(var)-vq(int) %.2f" % (np.corrcoef(m[:,0], m[:,2])[0,1], np.corrcoef(m[:,1], m[:,3])[0,1], np.corrcoef(m[:,0], m[:,3])[0,1]))
