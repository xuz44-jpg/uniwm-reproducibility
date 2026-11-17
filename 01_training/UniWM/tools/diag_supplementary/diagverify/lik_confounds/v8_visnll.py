from lc_common import *
import pickle
VS=pickle.load(open(VIS_SHIFT,"rb")); SG=+1   # fitted slope positive: +dyaw -> +shift
print("J. Truth = VISIBLE rotation direction (phase-correlation shift k->k+1), turning frames with |shift|>=3 px (~1/3 VQ token at 256 px).")
print("   ΔNLL(vis-mismatched text - vis-matched text) among {recorded, negated}; >0 means the model prefers the text matching the image motion")
for run in RUNS:
    R=[r for r in nongoal(load(run)) if abs(r["dyaw_k"])>=0.05]
    rows=[]
    for r in R:
        s=VS[(r["traj"],r["k"])]
        if abs(s)<3: continue
        match="recorded" if np.sign(s)==SG*np.sign(r["dyaw_k"]) else "negated"
        other="negated" if match=="recorded" else "recorded"
        rows.append((r["traj"], nll(r,other)-nll(r,match), match=="recorded", abs(s)))
    d=[x[1] for x in rows]; g=[x[0] for x in rows]
    dis=[x for x in rows if not x[2]]
    big=[x for x in rows if x[3]>=9.14]
    print(f"  {run:7s} n={len(rows)} ({len(dis)} with label sign opposite to image): {f3(cci(d,g))}; frames with |shift|>=1 token (n={len(big)}): {f3(cci([x[1] for x in big],[x[0] for x in big]))}; P(matched wins) {np.mean(np.array(d)>0):.2f}")
# correlation of per-frame ΔNLL(negated) with signed visible agreement
print("\n   Pearson r between per-frame ΔNLL(negated-recorded) and shift*sign(dyaw[k]) on turning frames")
for run in RUNS:
    R=[r for r in nongoal(load(run)) if abs(r["dyaw_k"])>=0.05]
    a=np.array([VS[(r['traj'],r['k'])]*np.sign(r['dyaw_k']) for r in R]); b=np.array([nll(r,'negated')-nll(r,'recorded') for r in R])
    print(f"  {run:7s} r={np.corrcoef(a,b)[0,1]:+.3f}")
