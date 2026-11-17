from lc_common import *
print("L. ΔNLL ~ b1*(|yaw_c|-|yaw_r|) + b2*(dx_c-dx_r) + b3*(|dy_c|-|dy_r|) + b4*signflip*min|yaw|  [signed differences; per rad / per m], all distinct candidates")
for run in RUNS:
    X=[];y=[];g=[]
    for r in nongoal(load(run)):
        ar=act(r,"recorded")
        for c in ("negated","zero","other_traj","prev1","prev2","next1"):
            if c not in r["scores"] or text(r,c)==text(r,"recorded"): continue
            ac=act(r,c); flip=float(np.sign(ac[2])!=np.sign(ar[2]) and abs(ac[2])>=0.02 and abs(ar[2])>=0.02)*min(abs(ac[2]),abs(ar[2]))
            X.append([1,abs(ac[2])-abs(ar[2]),ac[0]-ar[0],abs(ac[1])-abs(ar[1]),flip]); y.append(nll(r,c)-nll(r,"recorded")); g.append(r["traj"])
    X=np.array(X);y=np.array(y);g=np.array(g); beta=np.linalg.lstsq(X,y,rcond=None)[0]
    uniq=np.unique(g); idx={u:np.where(g==u)[0] for u in uniq}; rng=np.random.default_rng(0); B=[]
    for _ in range(2000):
        p=rng.choice(uniq,len(uniq)); ii=np.concatenate([idx[u] for u in p]); B.append(np.linalg.lstsq(X[ii],y[ii],rcond=None)[0])
    lo,hi=np.percentile(np.array(B),[2.5,97.5],axis=0)
    print(f"  {run:7s} "+"; ".join(f"{n} {b:+.2f} [{l:+.2f},{h:+.2f}]" for n,b,l,h in zip(["Δ|yaw|","Δdx","Δ|dy|","flip"],beta[1:],lo[1:],hi[1:])))
