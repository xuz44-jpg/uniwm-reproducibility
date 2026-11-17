"""L3b: a label-timing test with power: onset/offset frames where delta[k-1] and delta[k] differ in |dyaw| by >= 0.05."""
import sys; sys.path.insert(0, __file__.rsplit("/", 1)[0])
from likcommon import *

for run in RUNS:
    R = [r for r in load_run(run) if not is_goal(r) and "prev1" in r["scores"]]
    res = []
    for lab, cond in (("onset |dyaw_k|-|dyaw_k-1|>=0.05", lambda r: abs(r["dyaw_k"]) - abs(r["dyaw_km1"]) >= 0.05),
                      ("offset |dyaw_k-1|-|dyaw_k|>=0.05", lambda r: abs(r["dyaw_km1"]) - abs(r["dyaw_k"]) >= 0.05)):
        S = [r for r in R if cond(r) and text(r, "prev1") != text(r, "recorded")]
        dv = [nll(r, "prev1") - nll(r, "recorded") for r in S]
        m, lo, hi, _ = cluster_boot(dv, [r["traj"] for r in S], n_boot=2000)
        win = np.mean([d > 0 for d in dv])
        res.append(f"{lab}: n={len(S)} G={len(set(r['traj'] for r in S))} ΔNLL(prev1-rec) {m:+.3f} [{lo:+.3f},{hi:+.3f}] rec better {win:.2f}")
        # same for next1 at the mirrored situation
    for lab, cond in (("pre-onset |dyaw_k+1|-|dyaw_k|>=0.05", lambda r: "next1" in r["scores"] and abs(act(r, "next1")[2]) - abs(r["dyaw_k"]) >= 0.05),
                      ("pre-offset |dyaw_k|-|dyaw_k+1|>=0.05", lambda r: "next1" in r["scores"] and abs(r["dyaw_k"]) - abs(act(r, "next1")[2]) >= 0.05)):
        S = [r for r in load_run(run) if not is_goal(r) and cond(r) and text(r, "next1") != text(r, "recorded")]
        dv = [nll(r, "next1") - nll(r, "recorded") for r in S]
        m, lo, hi, _ = cluster_boot(dv, [r["traj"] for r in S], n_boot=2000)
        res.append(f"{lab}: n={len(S)} ΔNLL(next1-rec) {m:+.3f} [{lo:+.3f},{hi:+.3f}]")
    print(run, "\n   " + "\n   ".join(res))
