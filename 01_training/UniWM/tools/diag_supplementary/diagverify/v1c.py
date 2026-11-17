from v_common import *
R=load_var()
for comp,ci in (('dx',0),('dy',1),('dyaw',2)):
    S=per_state(R,ci)
    g=[s['traj'] for s in S]
    # squared gap between imagined-frame grand mean and real-next mean, minus noise of both means
    gap=[]; 
    for s in S:
        F=s['frames']; J=len(F)
        m_imag=np.mean([np.mean(f) for f in F])
        var_m_imag=np.var([np.mean(f) for f in F],ddof=1)/J       # unbiased for Var(grand mean of frame means)
        var_m_ref=np.var(s['refs'],ddof=1)/len(s['refs'])
        gap.append((m_imag-np.mean(s['refs']))**2 - var_m_imag - var_m_ref)
    gap=np.array(gap); w=np.array([s['w'] for s in S]); b=np.array([s['b'] for s in S])
    lo,hi=cboot(g,lambda ii:gap[ii].mean())
    tot=w.mean()+max(b.mean(),0)+max(gap.mean(),0)
    print(f"{comp}: bias^2 imagined-vs-real (noise-corrected) {gap.mean():.5f} [{lo:.5f},{hi:.5f}]; with it the three-way split is action-sampling {w.mean()/tot:.2f}, image-sampling {max(b.mean(),0)/tot:.2f}, imagined-vs-real shift {max(gap.mean(),0)/tot:.2f}")
    # share CI via per-trajectory bootstrap already; also compute trajectory-level share median
