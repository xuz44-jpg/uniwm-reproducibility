from v_common import *
I=load_int('stopft')
rows=[]
for r in I:
    for st in r['steps']:
        if st['stop_label_dist']:
            rows.append((r['traj'],st['s'],r['n'],st['s']==r['n']-1, np.mean([a=='stop' for a in st['gen'][1:]]), np.mean([a=='stop' for a in st['real'][1:]]), st['gen'][0]=='stop', st['real'][0]=='stop'))
rows=np.array(rows,dtype=object)
print('stopft dist-labelled steps',len(rows))
for lo,hi in ((0,12),(12,18),(18,22),(22,25)):
    m=[(x[1]>=lo and x[1]<hi) for x in rows]
    sub=rows[np.array(m)]
    if len(sub): print(f" depth {lo}-{hi-1}: n={len(sub)} trajs={len(set(sub[:,0]))} gen sampled Stop {np.mean(sub[:,4].astype(float)):.2f} real sampled {np.mean(sub[:,5].astype(float)):.2f}")
last=rows[rows[:,3].astype(bool)]; nl=rows[~rows[:,3].astype(bool)]
print(f" goal frame: n={len(last)} gen {np.mean(last[:,4].astype(float)):.2f} real {np.mean(last[:,5].astype(float)):.2f}; pre-goal dist-labelled: n={len(nl)} gen {np.mean(nl[:,4].astype(float)):.2f} real {np.mean(nl[:,5].astype(float)):.2f}")
# F-interval for greedy vs sampled run-to-run variance (Table 6.3), by simulation
rng=np.random.default_rng(0)
samp=[0.064,0.053,0.030,0.072]; greedy=[0.103,0.006]
vs=np.mean(np.square(samp)); vg=np.mean(np.square(greedy))
print(f"pooled sampled SD {np.sqrt(vs):.3f} (8 df), pooled greedy SD {np.sqrt(vg):.3f} (2 df), ratio of variances {vg/vs:.2f}")
F=(rng.chisquare(2,10**6)/2)/(rng.chisquare(8,10**6)/8)
q=np.percentile(F,[2.5,97.5]); print(f" 95% CI of variance ratio greedy/sampled: [{vg/vs/q[1]:.2f}, {vg/vs/q[0]:.1f}]")
# base only: sampled 0.064 (2df) vs greedy 0.103 (1 df)
F2=(rng.chisquare(1,10**6)/1)/(rng.chisquare(2,10**6)/2); q2=np.percentile(F2,[2.5,97.5]); r=0.103**2/0.064**2
print(f" base only ratio {r:.2f} CI [{r/q2[1]:.3f},{r/q2[0]:.0f}]; stop only ratio {(0.006/0.053)**2:.3f} CI [{(0.006/0.053)**2/q2[1]:.5f},{(0.006/0.053)**2/q2[0]:.1f}]")
