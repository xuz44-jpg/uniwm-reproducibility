from v_common import *
R=load_var()
S=[r for r in R if r['stop_next_released']]
W=[];B=[]
for r in S:
    fr=[[1.0 if a=='stop' else 0.0 for a in f] for f in r['imagined'] if f]
    vj=[np.var(f,ddof=1) for f in fr]; nj=[len(f) for f in fr]
    W.append(np.mean(vj)); B.append(np.var([np.mean(f) for f in fr],ddof=1)-np.mean([v/n for v,n in zip(vj,nj)]))
W=np.array(W);B=np.array(B)
print(f"Stop indicator on the 26 goal-adjacent states: within-image (action sampling) {W.mean():.4f}, between-image corrected {B.mean():.4f}, image share {B.mean()/(W.mean()+B.mean()):.3f}")
lo,hi=cboot([r['traj'] for r in S],lambda ii:B[ii].mean()/(W[ii].mean()+B[ii].mean()))
print(f"  cluster CI [{lo:.2f},{hi:.2f}]")
# states with between-image disagreement (some images Stop, some not)
mix=[r['traj'] for r,b in zip(S,B) if b>0.01]; print('states with mixed images:',len(mix))
# False stops on non-stop states: are they image-clustered?
from collections import Counter
fs=[(r['traj'],r['k'],j,sum(a=='stop' for a in f)) for r in R if not r['stop_next_released'] for j,f in enumerate(r['imagined']) if f and any(a=='stop' for a in f)]
print('false-stop images (traj,k,img,#stop of 5):',fs)
g=np.array([r['traj'] for r in S]); rng=np.random.default_rng(0); bs=[]
for _ in range(4000):
    ii=rng.choice(len(S),len(S)); den=W[ii].mean()+B[ii].mean()
    if den>0: bs.append(B[ii].mean()/den)
print('share CI (resamples with zero total dropped: %d)'%(4000-len(bs)), np.round(np.percentile(bs,[2.5,97.5]),3))
