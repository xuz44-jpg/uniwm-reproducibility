from lc_common import *
TURN=0.05
D = {run: nongoal(load(run)) for run in RUNS}
C4=("recorded","negated","zero","other_traj")
def split_best(r,C):
    v={c:nll(r,c) for c in C}; m=min(v.values()); wt={text(r,c) for c in C if v[c]==m}; w=[c for c in C if text(r,c) in wt]
    return {c:1/len(w) for c in w}
print("F. first-place share of each of the 4 candidates on turning frames (n=295), cluster CI for recorded and for recorded+negated")
for run in RUNS:
    S=[r for r in D[run] if abs(r["dyaw_k"])>=TURN]
    sh={c:np.mean([split_best(r,C4).get(c,0) for r in S]) for c in C4}
    rec=cci([split_best(r,C4).get("recorded",0) for r in S],[r["traj"] for r in S])
    rn=cci([split_best(r,C4).get("recorded",0)+split_best(r,C4).get("negated",0) for r in S],[r["traj"] for r in S])
    # pairwise: recorded beats negated
    pw=cci([float(nll(r,"negated")>nll(r,"recorded")) for r in S],[r["traj"] for r in S])
    print(f"  {run:7s} "+" ".join(f"{c} {sh[c]:.2f}" for c in C4)+f" | recorded {f3(rec)} | rec+neg {f3(rn)} | P(rec beats neg) {f3(pw)}")
print("\nE. zero - recorded on STRAIGHT frames split by recorded dx (forward-motion test)")
for run in RUNS:
    s=[]
    for lo,hi in ((-9,0.03),(0.03,0.08),(0.08,9)):
        S=[r for r in D[run] if abs(r["dyaw_k"])<0.02 and lo<=act(r,"recorded")[0]<hi and text(r,"zero")!=text(r,"recorded")]
        dv=[nll(r,"zero")-nll(r,"recorded") for r in S]
        s.append(f"dx[{lo},{hi}) n={len(S)} {f3(cci(dv,[r['traj'] for r in S]))}")
    print(f"  {run:7s} "+" | ".join(s))
print("\nD. straight frames: other_traj - recorded, by |dyaw| of the other action")
for run in RUNS:
    s=[]
    for lo,hi in ((0,0.05),(0.05,9)):
        S=[r for r in D[run] if abs(r["dyaw_k"])<0.02 and lo<=abs(act(r,"other_traj")[2])<hi and text(r,"other_traj")!=text(r,"recorded")]
        dv=[nll(r,"other_traj")-nll(r,"recorded") for r in S]
        s.append(f"other|dyaw|[{lo},{hi}) n={len(S)} {f3(cci(dv,[r['traj'] for r in S]))}")
    # straight frames right after a turn vs not
    S1=[r for r in D[run] if abs(r["dyaw_k"])<0.02 and r["dyaw_km1"] is not None and abs(r["dyaw_km1"])>=TURN]
    S0=[r for r in D[run] if abs(r["dyaw_k"])<0.02 and r["dyaw_km1"] is not None and abs(r["dyaw_km1"])<0.02]
    z1=cci([nll(r,"zero")-nll(r,"recorded") for r in S1],[r['traj'] for r in S1]); z0=cci([nll(r,"zero")-nll(r,"recorded") for r in S0],[r['traj'] for r in S0])
    print(f"  {run:7s} "+" | ".join(s)+f" || straight after turn (n={len(S1)}): prev1-rec {f3(cci([nll(r,'prev1')-nll(r,'recorded') for r in S1],[r['traj'] for r in S1]))}")
