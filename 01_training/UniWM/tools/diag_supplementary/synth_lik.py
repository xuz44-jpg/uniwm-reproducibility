import json,glob,os,numpy as np
def _repo_root():
    """UniWM repository root: $UNIWM_ROOT if set, else the nearest ancestor of this file that holds output/diag/raw."""
    env = os.environ.get("UNIWM_ROOT")
    if env:
        return env
    d = os.path.dirname(os.path.abspath(__file__))
    while d != os.path.dirname(d):
        if os.path.isdir(os.path.join(d, "output", "diag", "raw")):
            return d
        d = os.path.dirname(d)
    raise SystemExit("cannot locate the UniWM repository (no output/diag/raw above this script); set UNIWM_ROOT")


REPO = _repo_root()
def boot(vals,trajs,B=4000):
    by={}
    for v,t in zip(vals,trajs): by.setdefault(t,[]).append(v)
    ts=list(by); rng=np.random.default_rng(0); bs=[]
    for _ in range(B):
        s=rng.choice(ts,len(ts)); bs.append(np.mean(np.concatenate([by[t] for t in s])))
    return np.mean(vals),np.percentile(bs,2.5),np.percentile(bs,97.5)
for run in ('base','stopft','de50','de100','de200','de459'):
    d=json.load(open(glob.glob(os.path.join(REPO, 'output', 'diag', 'raw', f'{run}_likelihood*', 'action_likelihood_results', 'likelihood_rank0.json'))[0]))
    ng=[r for r in d if not r['target_is_goal']]
    tr=[r for r in ng if abs(r['dyaw_k'])>=0.05]
    L=[r for r in tr if r['dyaw_k']>0]; Rr=[r for r in tr if r['dyaw_k']<0]
    dn=lambda r:r['scores']['negated']['nll_img']-r['scores']['recorded']['nll_img']
    dz=lambda r:r['scores']['zero']['nll_img']-r['scores']['recorded']['nll_img']
    neg=boot([dn(r) for r in tr],[r['traj'] for r in tr])
    # balanced alignment
    rng=np.random.default_rng(1)
    aL=np.mean([dn(r) for r in L]); aR=np.mean([dn(r) for r in Rr])
    zt=boot([dz(r) for r in tr],[r['traj'] for r in tr])
    st=[r for r in ng if abs(r['dyaw_k'])<0.02 and r['scores']['recorded']['action'][0]>=0.05]
    zs=boot([dz(r) for r in st],[r['traj'] for r in st])
    pw=np.mean([dn(r)>0 for r in tr])
    print(run,'nTurn',len(tr),'L/R',len(L),len(Rr),'neg %.3f [%.3f,%.3f]'%neg,'align %.3f pref %.3f'%((aL+aR)/2,(aL-aR)/2),'zeroTurn %.2f [%.2f,%.2f]'%zt,'zeroStraightFwd(n=%d) %.2f [%.2f,%.2f]'%((len(st),)+zs),'pairwise %.2f'%pw)
