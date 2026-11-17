from lc_common import *
D = {run: nongoal(load(run)) for run in RUNS}
print("G. ΔNLL = NLL(c)-NLL(rec) ~ b_more*(|yaw_c|-|yaw_r|)_+ + b_less*(|yaw_r|-|yaw_c|)_+ + b_flip*[sign flip]*min|yaw| + b_dx*|Δdx| + b_dy*|Δdy|  (per rad / per m)")
print("   b_more<0 => candidates that turn MORE than recorded make the real frame MORE likely (generic 'turn text' preference)")
def rows(R):
    X=[];y=[];g=[]
    for r in R:
        ar=act(r,"recorded")
        for c in ("negated","zero","other_traj","prev1","prev2","next1"):
            if c not in r["scores"] or text(r,c)==text(r,"recorded"): continue
            ac=act(r,c); d=abs(ac[2])-abs(ar[2])
            flip=float(np.sign(ac[2])!=np.sign(ar[2]) and abs(ac[2])>=0.02 and abs(ar[2])>=0.02)*min(abs(ac[2]),abs(ar[2]))
            X.append([1,max(d,0),max(-d,0),flip,abs(ac[0]-ar[0]),abs(ac[1]-ar[1])]); y.append(nll(r,c)-nll(r,"recorded")); g.append(r["traj"])
    return np.array(X),np.array(y),np.array(g)
nm=["more","less","flip","dx","dy"]
for run in RUNS:
    X,y,g=rows(D[run]); beta=np.linalg.lstsq(X,y,rcond=None)[0]
    uniq=np.unique(g); idx={u:np.where(g==u)[0] for u in uniq}; rng=np.random.default_rng(0); B=[]
    for _ in range(2000):
        p=rng.choice(uniq,len(uniq)); ii=np.concatenate([idx[u] for u in p]); B.append(np.linalg.lstsq(X[ii],y[ii],rcond=None)[0])
    lo,hi=np.percentile(np.array(B),[2.5,97.5],axis=0)
    print(f"  {run:7s} "+"; ".join(f"{n} {b:+.2f} [{l:+.2f},{h:+.2f}]" for n,b,l,h in zip(nm,beta[1:],lo[1:],hi[1:])))
