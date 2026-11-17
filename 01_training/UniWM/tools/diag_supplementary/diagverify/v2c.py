from v_common import *
V=load_var(); I=load_int('base')
idx={(r['traj'],st['s']):st for r in I for st in r['steps']}
print('variance trajs not in intervention:',sorted(set(r['traj'] for r in V)-set(r['traj'] for r in I)))
for comp,ci in (('dx',0),('dy',1),('dyaw',2)):
    S=per_state(V,ci)
    rows=[]
    for s in S:
        st=idx.get((s['traj'],s['k']+1))
        if st is None: continue
        ir=moves(st['real'][1:]); iq=moves(st['vq'][1:])
        if len(ir)<2 or len(iq)<2: continue
        rows.append((s['traj'],s['k'],s['ref'],s['refq'],np.var([a[ci] for a in ir],ddof=1),np.var([a[ci] for a in iq],ddof=1)))
    A=np.array([x[2:] for x in rows]); g=[x[0] for x in rows]
    print(f"{comp}: matched frames {len(rows)}; variance-run real {A[:,0].mean():.5f} vq {A[:,1].mean():.5f} | intervention-run same frames real {A[:,2].mean():.5f} vq {A[:,3].mean():.5f}")
    pooled_r=(A[:,0]+A[:,2])/2; pooled_q=(A[:,1]+A[:,3])/2; d=pooled_r-pooled_q
    lo,hi=cboot(g,lambda ii:d[ii].mean())
    print(f"   pooled over both runs (10 samples/frame): real {pooled_r.mean():.5f} vq {pooled_q.mean():.5f}; diff {d.mean():+.5f} [{lo:+.5f},{hi:+.5f}]")
    if comp=='dyaw':
        top=np.argsort(-(A[:,0]-A[:,1]))[:5]
        for i in top: print('   ',rows[i][0],rows[i][1],np.round(A[i],4), 'real samples var-run', [round(a[2],2) for a in moves(next(r for r in V if r['traj']==rows[i][0] and r['k']==rows[i][1])['real_next'])], 'int-run', [round(a[2],2) for a in moves(idx[(rows[i][0],rows[i][1]+1)]['real'][1:])], 'vq var-run',[round(a[2],2) for a in moves(next(r for r in V if r['traj']==rows[i][0] and r['k']==rows[i][1])['real_next_vq'])],'vq int-run',[round(a[2],2) for a in moves(idx[(rows[i][0],rows[i][1]+1)]['vq'][1:])])
