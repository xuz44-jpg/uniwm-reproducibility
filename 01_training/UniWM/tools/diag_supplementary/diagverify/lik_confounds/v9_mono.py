from lc_common import *
print("K. nonparametric check of the monotone 'more turn text is better' effect: other_traj - recorded")
for run in RUNS:
    R=nongoal(load(run)); s=[]
    for nm,cond in (("turning, |yaw_o|>|yaw_r|+0.03",lambda r: abs(r["dyaw_k"])>=0.05 and abs(act(r,'other_traj')[2])>abs(r['dyaw_k'])+0.03),
                    ("turning, |yaw_o|<|yaw_r|-0.03",lambda r: abs(r["dyaw_k"])>=0.05 and abs(act(r,'other_traj')[2])<abs(r['dyaw_k'])-0.03),
                    ("turning, same sign, |Δ|<=0.03",lambda r: abs(r["dyaw_k"])>=0.05 and np.sign(act(r,'other_traj')[2])==np.sign(r['dyaw_k']) and abs(abs(act(r,'other_traj')[2])-abs(r['dyaw_k']))<=0.03),
                    ("turning, opp sign, |Δ|<=0.03",lambda r: abs(r["dyaw_k"])>=0.05 and np.sign(act(r,'other_traj')[2])!=np.sign(r['dyaw_k']) and abs(abs(act(r,'other_traj')[2])-abs(r['dyaw_k']))<=0.03)):
        S=[r for r in R if cond(r) and text(r,'other_traj')!=text(r,'recorded')]
        s.append(f"{nm}: n={len(S)} {f3(cci([nll(r,'other_traj')-nll(r,'recorded') for r in S],[r['traj'] for r in S]))}")
    print(f"  {run}\n    "+"\n    ".join(s))
