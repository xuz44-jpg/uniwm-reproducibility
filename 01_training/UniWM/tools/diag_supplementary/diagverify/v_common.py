import json, glob, os, numpy as np
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
ROOT = os.path.join(REPO, "output", "diag", "raw") + os.sep
def load_var():
    d=glob.glob(ROOT+'base_variance*/variance_decomp_results/')[0]
    R=[]
    for f in sorted(glob.glob(d+'variance_rank*.json')): R+=json.load(open(f))
    return R
def load_int(run):
    d=glob.glob(ROOT+run+'_intervention*/action_intervention_results/')[0]
    R=[]
    for f in sorted(glob.glob(d+'intervention_rank*.json')): R+=json.load(open(f))
    return R
moves=lambda acts:[a for a in (acts or []) if a not in (None,'stop')]
def per_state(R, ci):
    out=[]
    for r in R:
        frames=[[a[ci] for a in moves(fr)] for fr in r['imagined'] if fr]
        frames=[f for f in frames if len(f)>=2]
        ref=[a[ci] for a in moves(r['real_next'])]; refq=[a[ci] for a in moves(r.get('real_next_vq'))]
        if len(frames)<2 or len(ref)<2 or len(refq)<2: continue
        vj=[np.var(f,ddof=1) for f in frames]; nj=[len(f) for f in frames]
        w=np.mean(vj); braw=np.var([np.mean(f) for f in frames],ddof=1)
        b=braw-np.mean([v/n for v,n in zip(vj,nj)])
        out.append(dict(traj=r['traj'],k=r['k'],n=r['n'],w=w,b=b,braw=braw,ref=np.var(ref,ddof=1),refq=np.var(refq,ddof=1),
                        nframes=len(frames),nj=nj,rec=r['recorded_k'],recn=r['recorded_next'],
                        mimag=np.mean([np.mean(f) for f in frames]),mref=np.mean(ref),mrefq=np.mean(refq),
                        frames=frames,refs=ref,refqs=refq))
    return out
def cboot(groups, stat, n_boot=4000, seed=0):
    """stat: function(list of indices) -> value; resample trajectories."""
    g=np.asarray(groups); uniq=np.unique(g); idx={u:np.where(g==u)[0] for u in uniq}
    rng=np.random.default_rng(seed); bs=[]
    for _ in range(n_boot):
        pick=rng.choice(uniq,len(uniq)); ii=np.concatenate([idx[u] for u in pick]); bs.append(stat(ii))
    return np.percentile(bs,[2.5,97.5])
