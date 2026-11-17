from lc_common import *
TURN=0.05
D = {run: nongoal(load(run)) for run in RUNS}
# I: dy-dyaw relation on recorded actions
A = np.array([act(r,"recorded") for r in D["base"]])
print("I. recorded actions (546 frames): corr(dx,dyaw)=%.2f corr(dy,dyaw)=%.2f corr(|dyaw|,dx)=%.2f" % (np.corrcoef(A[:,0],A[:,2])[0,1], np.corrcoef(A[:,1],A[:,2])[0,1], np.corrcoef(np.abs(A[:,2]),A[:,0])[0,1]))
st=[r for r in D["base"] if abs(r["dyaw_k"])<0.02]; tu=[r for r in D["base"] if abs(r["dyaw_k"])>=TURN]
for nm,S in (("straight",st),("turning",tu)):
    a=np.array([act(r,"recorded") for r in S]); print(f"   {nm}: n={len(S)} dx median {np.median(a[:,0]):.3f} (q10 {np.percentile(a[:,0],10):.3f}), |dy| median {np.median(np.abs(a[:,1])):.3f}, |dyaw| median {np.median(np.abs(a[:,2])):.3f}")
# C: pairwise regression across all distinct candidates: ΔNLL ~ b0 + b_mag*| |yaw_c|-|yaw_r| | + b_sign*[opposite sign & both>=0.02] + b_dx*|Δdx| + b_dy*|Δdy|
print("\nC. OLS across all distinct (frame, candidate) pairs, ΔNLL = NLL(c)-NLL(recorded); coefficients per unit; cluster-bootstrap CIs")
def rows(R, cands=("negated","zero","other_traj","prev1","prev2","next1")):
    X=[];y=[];g=[]
    for r in R:
        ar=act(r,"recorded")
        for c in cands:
            if c not in r["scores"] or text(r,c)==text(r,"recorded"): continue
            ac=act(r,c)
            mag=abs(abs(ac[2])-abs(ar[2]))
            sgn=float(np.sign(ac[2])!=np.sign(ar[2]) and abs(ac[2])>=0.02 and abs(ar[2])>=0.02)*min(abs(ac[2]),abs(ar[2]))
            X.append([1,mag,sgn,abs(ac[0]-ar[0]),abs(ac[1]-ar[1])]); y.append(nll(r,c)-nll(r,"recorded")); g.append(r["traj"])
    return np.array(X),np.array(y),np.array(g)
names=["const","||yaw_c|-|yaw_r|| (per rad)","sign flip x min|yaw| (per rad)","|Δdx| (per m)","|Δdy| (per m)"]
for run in RUNS:
    X,y,g=rows(D[run]); beta=np.linalg.lstsq(X,y,rcond=None)[0]
    uniq=np.unique(g); idx={u:np.where(g==u)[0] for u in uniq}; rng=np.random.default_rng(0); B=[]
    for _ in range(2000):
        p=rng.choice(uniq,len(uniq)); ii=np.concatenate([idx[u] for u in p]); B.append(np.linalg.lstsq(X[ii],y[ii],rcond=None)[0])
    B=np.array(B); lo,hi=np.percentile(B,[2.5,97.5],axis=0)
    print(f"  {run:7s} n={len(y)}: "+"; ".join(f"{nm} {b:+.2f} [{l:+.2f},{h:+.2f}]" for nm,b,l,h in zip(names[1:],beta[1:],lo[1:],hi[1:])))
