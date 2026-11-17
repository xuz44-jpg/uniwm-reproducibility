import sys; sys.path.insert(0, __file__.rsplit('/',1)[0])
from common import *
R = V()
print("states", len(R), "trajs", len(set(r['traj'] for r in R)), "stop-next(released)", sum(r['stop_next_released'] for r in R))
# confirm last state per traj is k=n-2
print("stop states all k=n-2:", all((r['k']==r['n']-2)==r['stop_next_released'] for r in R))
def per_state(r, ci):
    frames = [[a[ci] for a in moves(fr)] for fr in r["imagined"] if fr]
    frames = [f for f in frames if len(f) >= 2]
    ref = [a[ci] for a in moves(r["real_next"])]; refq = [a[ci] for a in moves(r.get("real_next_vq"))]
    if len(frames) < 2 or len(ref) < 2 or len(refq) < 2: return None
    vj = [np.var(f, ddof=1) for f in frames]; nj = [len(f) for f in frames]
    w = np.mean(vj); braw = np.var([np.mean(f) for f in frames], ddof=1)
    b = braw - np.mean([v/n for v, n in zip(vj, nj)])
    return dict(traj=r['traj'], k=r['k'], w=w, b=b, braw=braw, ref=np.var(ref, ddof=1), refq=np.var(refq, ddof=1),
                turn=abs(r['recorded_k'][2]) >= 0.05, nframes=len(frames), nact=sum(nj))
share = lambda S: max(np.mean([s['b'] for s in S]), 0) / (np.mean([s['w'] for s in S]) + max(np.mean([s['b'] for s in S]), 0))
share_raw = lambda S: np.mean([s['braw'] for s in S]) / (np.mean([s['w'] for s in S]) + np.mean([s['braw'] for s in S]))
for comp, ci in (("dx",0),("dy",1),("dyaw",2)):
    S = [x for x in (per_state(r, ci) for r in R) if x]
    G = {}
    for s in S: G.setdefault(s['traj'], []).append(s)
    lo, hi = cluster_boot(share, G); lor, hir = cluster_boot(share_raw, G)
    w = np.array([s['w'] for s in S]); b = np.array([s['b'] for s in S])
    # concentration: share of sum(w) and sum(b) from top 5 states
    tw = np.sort(w)[::-1][:5].sum()/w.sum(); tb = np.sort(b)[::-1][:5].sum()/b.sum()
    # per-state median share among states with nonzero total
    tot = w + np.clip(b, 0, None); ps = np.clip(b,0,None)[tot>0]/tot[tot>0]
    # w - b difference with CI
    dlo, dhi = cluster_boot(lambda SS: np.mean([s['w']-s['b'] for s in SS]), G)
    print(f"{comp}: n={len(S)} states ({len(G)} trajs); within {w.mean():.5f}, between corr {b.mean():.5f} (raw {np.mean([s['braw'] for s in S]):.5f}); "
          f"share {share(S):.3f} [{lo:.2f},{hi:.2f}] raw {share_raw(S):.3f} [{lor:.2f},{hir:.2f}]; W-B {np.mean(w-b):+.5f} [{dlo:+.5f},{dhi:+.5f}]")
    print(f"     top-5 states carry {tw:.2f} of sum(within) and {tb:.2f} of sum(between corr); states with within==0: {(w==0).mean():.2f}; "
          f"between<0: {(b<0).mean():.2f}; per-state share median {np.median(ps):.2f} (n={len(ps)} states with nonzero total, {int((tot==0).sum())} all-identical)")
    for lab, sub in (("turn |dyaw_k|>=0.05", [s for s in S if s['turn']]), ("non-turn", [s for s in S if not s['turn']])):
        if sub:
            print(f"     {lab}: n={len(sub)} within {np.mean([s['w'] for s in sub]):.5f} between {np.mean([s['b'] for s in sub]):.5f} share {share(sub):.2f}")
    # leave-top-state-out
    order = np.argsort(-(w+np.clip(b,0,None)))
    S2 = [S[i] for i in order[3:]]
    print(f"     drop 3 highest-variance states: share {share(S2):.2f}")
    print(f"     sample sizes: frames/state min {min(s['nframes'] for s in S)}, actions/state min {min(s['nact'] for s in S)} mean {np.mean([s['nact'] for s in S]):.1f}")
