"""L4: goal-copy control; per-token NLL by motion; negated ΔNLL on goal frames."""
import sys; sys.path.insert(0, __file__.rsplit("/", 1)[0])
from likcommon import *

for run in RUNS:
    R_all = load_run(run)
    G = [r for r in R_all if is_goal(r)]; R = [r for r in R_all if not is_goal(r)]
    pt = lambda S: np.array([nll(r, "recorded") / N_IMG for r in S])
    still = [r for r in R if abs(act(r, "recorded")[0]) < 0.01 and abs(act(r, "recorded")[1]) < 0.01 and abs(r["dyaw_k"]) < 0.02]
    ptR, ptG, ptS = pt(R), pt(G), pt(still)
    dvg = np.array([nll(r, "negated") - nll(r, "recorded") for r in G if text(r, "negated") != text(r, "recorded")])
    m, lo, hi, _ = cluster_boot(dvg, [r["traj"] for r in G])
    gy = np.array([abs(r["dyaw_k"]) for r in G])
    print(f"{run:7s} per-token NLL: goal {ptG.mean():.3f} (range {ptG.min():.3f}-{ptG.max():.3f}); non-goal {ptR.mean():.3f} (min {ptR.min():.3f}, 1st pct {np.percentile(ptR,1):.3f}); "
          f"standing-still frames (n={len(still)}) {ptS.mean() if len(still) else float('nan'):.3f} | goal frames: turning {np.sum(gy>=0.05)}/{len(G)}, "
          f"negated ΔNLL {m:+.3f} [{lo:+.3f},{hi:+.3f}] mean|Δ| {np.abs(dvg).mean():.3f}; zero ΔNLL {np.mean([nll(r,'zero')-nll(r,'recorded') for r in G]):+.3f}")
