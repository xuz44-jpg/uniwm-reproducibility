from v_common import *
# Replication of V2 in the intervention runs: 5 sampled actions per step on real vs vq (and gen), all steps
for run,key in (('base','stop_label_released'),('stopft','stop_label_dist')):
    R=load_int(run)
    print('==',run, 'trajs',len(R))
    for comp,ci in (('dx',0),('dy',1),('dyaw',2)):
        rows=[]
        for r in R:
            for st in r['steps']:
                if st[key] or st['recorded'] is None: continue
                vals={}
                for nm in ('real','vq','gen'):
                    s=moves((st.get(nm) or [])[1:])  # sampled only, drop greedy
                    vals[nm]=np.var([a[ci] for a in s],ddof=1) if len(s)>=2 else np.nan
                rows.append((r['traj'],st['s'],vals['real'],vals['vq'],vals['gen']))
        g=[x[0] for x in rows]; real=np.array([x[2] for x in rows]); vq=np.array([x[3] for x in rows]); gen=np.array([x[4] for x in rows])
        m=~np.isnan(real)&~np.isnan(vq)
        g2=[gg for gg,mm in zip(g,m) if mm]; d=(real-vq)[m]
        lo,hi=cboot(g2,lambda ii:d[ii].mean())
        r_=real[m]; q_=vq[m]
        rlo,rhi=cboot(g2,lambda ii:r_[ii].mean()/q_[ii].mean())
        mg=m&~np.isnan(gen)
        print(f" {comp}: n={m.sum()} steps; sampled var real {np.nanmean(real[m]):.5f} vq {np.nanmean(vq[m]):.5f} gen {np.nanmean(gen[mg]):.5f}; real-vq {d.mean():+.5f} [{lo:+.5f},{hi:+.5f}]; ratio {r_.mean()/q_.mean():.2f} [{rlo:.2f},{rhi:.2f}]; real>vq {np.mean(d>0):.2f} vq>real {np.mean(d<0):.2f}")
