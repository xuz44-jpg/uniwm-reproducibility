from lc_common import *
print("M. Does the viz likelihood side with the direction the ACTION HEAD infers from the same (start, goal, current) images?")
print("   ΔNLL(negated-recorded) on turning frames, split by the action head's greedy dyaw sign on the real frame (intervention run)")
for run,irun in (("base","base"),("stopft","stopft")):
    I={}
    for f in sorted(glob.glob(f"{RAW}/{irun}_intervention*/action_intervention_results/intervention_rank*.json")):
        for r in json.load(open(f)):
            for st in r["steps"]:
                p=st["real"][0] if st.get("real") else None
                if p not in (None,"stop"): I[(r["traj"],st["s"])]=p[2]
    R=[r for r in nongoal(load(run)) if abs(r["dyaw_k"])>=0.05 and (r["traj"],r["k"]) in I and abs(I[(r["traj"],r["k"])])>=0.025]
    ok=[r for r in R if np.sign(I[(r["traj"],r["k"])])==np.sign(r["dyaw_k"])]; bad=[r for r in R if np.sign(I[(r["traj"],r["k"])])!=np.sign(r["dyaw_k"])]
    f=lambda S: f3(cci([nll(r,"negated")-nll(r,"recorded") for r in S],[r["traj"] for r in S]))
    print(f"  {run:7s} head sign = label (n={len(ok)}): {f(ok)} | head sign opposite (n={len(bad)}): {f(bad)}")
