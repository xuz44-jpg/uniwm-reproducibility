import json,glob,os,sys,numpy as np
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
sys.path.insert(0, os.path.join(REPO, 'tools'))
import action_diag_analysis as A
for run,key in (('base','stop_label_released'),('stopft','stop_label_dist')):
    R=[]
    for f in sorted(glob.glob(os.path.join(REPO, 'output', 'diag', 'raw', f'{run}_intervention*', 'action_intervention_results', 'intervention_rank*.json'))):
        R+=json.load(open(f))
    rows=[(r['traj'],st) for r in R for st in r['steps']]
    def acc(st,c,tol=None):
        p=st[c][0]
        if tol is None: return A._acc(p,st['recorded'],st[key])
        if st[key] or p=='stop': return bool(st[key]) and p=='stop'
        return p is not None and all(abs(a-A._bin(b))<=t for a,b,t in zip(p,st['recorded'],tol))
    trajs=sorted({t for t,_ in rows})
    def boot(sel,tol=None,B=4000):
        # paired real-gen, clustered
        by={t:[] for t in trajs}
        for t,st in rows:
            if sel(st): by[t].append(acc(st,'real',tol)-acc(st,'gen',tol))
        ts=[t for t in trajs if by[t]]
        rng=np.random.default_rng(0); est=np.mean(sum(by.values(),[]))
        bs=[]
        for _ in range(B):
            s=rng.choice(ts,len(ts)); v=sum((by[t] for t in s),[]); bs.append(np.mean(v))
        return est,np.percentile(bs,2.5),np.percentile(bs,97.5),sum(len(v) for v in by.values())
    print(run,'n',len(rows),'traj',len(trajs))
    print(' all', ['%.3f'%x for x in boot(lambda st:True)[:3]])
    print(' move',['%.3f'%x for x in boot(lambda st:not st[key])[:3]], boot(lambda st:not st[key])[3])
    print(' move strict .05',['%.3f'%x for x in boot(lambda st:not st[key],(0.05,0.05,0.05))[:3]])
    # zero-action accuracy on move steps
    mv=[st for _,st in rows if not st[key]]
    print(' zero-action acc on move', np.mean([all(abs(0-A._bin(b))<=t for b,t in zip(st['recorded'],A.TOL)) for st in mv]))
    for lo,hi in ((0,5),(5,10),(10,15),(15,20),(20,25)):
        sub=[st for _,st in rows if lo<=st['s']<hi and not st[key]]
        nstop=sum(1 for _,st in rows if lo<=st['s']<hi and st[key])
        print('  bin',lo,hi,'nmove',len(sub),'nstop',nstop,'real %.2f gen %.2f'%(np.mean([acc(s,'real') for s in sub]),np.mean([acc(s,'gen') for s in sub])))
    # opposite sign counts on turning steps
    tr=[st for _,st in rows if not st[key] and abs(st['recorded'][2])>=0.05]
    for c in ('real','gen'):
        g=[st[c][0] for st in tr]
        opp=sum(1 for p,st in zip(g,tr) if p not in (None,'stop') and np.sign(p[2])==-np.sign(st['recorded'][2]) and p[2]!=0)
        opp25=sum(1 for p,st in zip(g,tr) if p not in (None,'stop') and np.sign(p[2])==-np.sign(st['recorded'][2]) and abs(p[2])>=0.025)
        small=sum(1 for p in g if p not in (None,'stop') and abs(p[2])<0.025)
        print('  turn',c,'n',len(tr),'ok %.3f'%np.mean([A._turn_ok(p,st['recorded']) for p,st in zip(g,tr)]),'opp(any)',opp,'opp(>=.025)',opp25,'|pred|<.025',small)
