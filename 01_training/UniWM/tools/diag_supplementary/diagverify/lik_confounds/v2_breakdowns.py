from lc_common import *
TURN=0.05
D = {run: nongoal(load(run)) for run in RUNS}
key = lambda r: (r["traj"], r["k"])
# A: cross-checkpoint correlation of per-frame ΔNLL(negated) and ΔNLL(zero) on turning frames
print("A. Pearson r of per-frame ΔNLL across checkpoints (turning frames, n=295)")
for c in ("negated","zero","other_traj"):
    M = {}
    for run in RUNS:
        dd = {key(r): nll(r,c)-nll(r,"recorded") for r in D[run] if abs(r["dyaw_k"])>=TURN}
        M[run]=dd
    ks = sorted(M["base"])
    X = np.array([[M[run][k] for k in ks] for run in RUNS])
    C = np.corrcoef(X)
    print(f"  {c}:"); 
    for i,run in enumerate(RUNS): print("   ", run.ljust(7), " ".join(f"{C[i,j]:+.2f}" for j in range(len(RUNS))))
# B: ΔNLL(negated) by |dyaw| bin
print("\nB. ΔNLL(negated-recorded) by |dyaw[k]| bin (nats per frame, cluster CI)")
bins=[(0.05,0.10),(0.10,0.20),(0.20,9)]
for run in RUNS:
    s=[]
    for lo,hi in bins:
        S=[r for r in D[run] if lo<=abs(r["dyaw_k"])<hi]
        dv=[nll(r,"negated")-nll(r,"recorded") for r in S]
        s.append(f"[{lo:.2f},{hi if hi<9 else 'inf'}) n={len(S)} {f3(cci(dv,[r['traj'] for r in S]))}")
    # slope of ΔNLL on |dyaw| over turning frames
    S=[r for r in D[run] if abs(r["dyaw_k"])>=TURN]
    x=np.array([abs(r["dyaw_k"]) for r in S]); y=np.array([nll(r,"negated")-nll(r,"recorded") for r in S])
    sl=cci(np.c_[x,y], [r['traj'] for r in S], stat=lambda a: np.polyfit(a[:,0],a[:,1],1)[0]) if False else None
    print(f"  {run:7s} "+" | ".join(s))
# distribution of |dyaw| on turning frames
S=[r for r in D["base"] if abs(r["dyaw_k"])>=TURN]; a=np.abs([r["dyaw_k"] for r in S])
print(f"  |dyaw| on turning frames: median {np.median(a):.3f}, quartiles {np.percentile(a,25):.3f}/{np.percentile(a,75):.3f}, max {a.max():.3f}")
