from lc_common import *
TURN=0.05
D = {run: nongoal(load(run)) for run in RUNS}
def yaws(r):
    g=lambda c: act(r,c)[2] if c in r["scores"] else None
    return g("prev2"),g("prev1"),g("recorded"),g("next1")
def sustained(r):
    y=yaws(r)
    return all(v is not None and abs(v)>=TURN for v in y) and len({np.sign(v) for v in y})==1
print("H. sustained turns: dyaw[k-2..k+1] all >=0.05 rad, same sign (visible rotation k->k+1 has that sign under any lead of 0-2 steps)")
for run in RUNS:
    S=[r for r in D[run] if sustained(r)]
    dn=cci([nll(r,"negated")-nll(r,"recorded") for r in S],[r["traj"] for r in S])
    dz=cci([nll(r,"zero")-nll(r,"recorded") for r in S],[r["traj"] for r in S])
    print(f"  {run:7s} n={len(S)} traj={len(set(r['traj'] for r in S))}: negated {f3(dn)} | zero {f3(dz)}")
print("\nL3 power check on mid-turn frames: |dyaw(c) - dyaw(rec)| for c in prev2/prev1/next1 vs negated")
M=[r for r in D["base"] if r["dyaw_km1"] is not None and abs(r["dyaw_k"])>=TURN and abs(r["dyaw_km1"])>=TURN and np.sign(r["dyaw_k"])==np.sign(r["dyaw_km1"])]
for c in ("prev2","prev1","next1","negated"):
    v=[abs(act(r,c)[2]-act(r,"recorded")[2]) for r in M if c in r["scores"]]
    s=[np.sign(act(r,c)[2])==np.sign(act(r,"recorded")[2]) and abs(act(r,c)[2])>=TURN for r in M if c in r["scores"]]
    print(f"  {c:8s} median |Δdyaw| {np.median(v):.3f} rad, mean {np.mean(v):.3f}; same-sign turn on {np.mean(s):.2f} of {len(v)}")
print("\nL3' lag test where labels differ: onsets (|dyaw[k-1]|<0.02, |dyaw[k]|>=0.05) and offsets (|dyaw[k-1]|>=0.05, |dyaw[k]|<0.02); ΔNLL(prev1 - recorded)")
print("   a 1-step lead predicts prev1 better (ΔNLL<0) at both")
for run in RUNS:
    on=[r for r in D[run] if r["dyaw_km1"] is not None and abs(r["dyaw_km1"])<0.02 and abs(r["dyaw_k"])>=TURN]
    off=[r for r in D[run] if r["dyaw_km1"] is not None and abs(r["dyaw_km1"])>=TURN and abs(r["dyaw_k"])<0.02]
    big=[r for r in D[run] if "prev1" in r["scores"] and abs(act(r,"prev1")[2]-r["dyaw_k"])>=0.05]
    f=lambda S: f3(cci([nll(r,"prev1")-nll(r,"recorded") for r in S],[r["traj"] for r in S])) if len(set(r['traj'] for r in S))>1 else "n/a"
    print(f"  {run:7s} onsets n={len(on)} {f(on)} | offsets n={len(off)} {f(off)} | any |dyaw[k-1]-dyaw[k]|>=0.05 n={len(big)} {f(big)}")
