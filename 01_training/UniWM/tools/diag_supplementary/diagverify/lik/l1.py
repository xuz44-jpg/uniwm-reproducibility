"""L1: negated-heading ΔNLL on turning frames; per-token NLL; rank shares."""
import sys; sys.path.insert(0, __file__.rsplit("/", 1)[0])
from likcommon import *

C4 = ("recorded", "negated", "zero", "other_traj")




per_run_neg = {}
for run in RUNS:
    R_all = load_run(run)
    R = [r for r in R_all if not is_goal(r)]
    G = [r for r in R_all if is_goal(r)]
    trajs = sorted(set(r["traj"] for r in R_all))
    print(f"==== {run}: {len(R_all)} frames, {len(trajs)} traj, sum(n-1)={sum(r['n']-1 for r in {r['traj']:r for r in R_all}.values())}, non-goal {len(R)}, goal {len(G)}")
    ptok = np.mean([nll(r, 'recorded') for r in R]) / N_IMG
    ptok_g = np.mean([nll(r, 'recorded') for r in G]) / N_IMG
    print(f"  per-token NLL non-goal {ptok:.3f} goal {ptok_g:.3f}; per-frame NLL mean {np.mean([nll(r,'recorded') for r in R]):.1f} sd {np.std([nll(r,'recorded') for r in R]):.1f}")
    for sname, cond in (("turning", turning), ("straight", straight), ("midturn", midturn), ("all", lambda r: True)):
        S = [r for r in R if cond(r) and text(r, "negated") != text(r, "recorded")]
        dv = np.array([nll(r, "negated") - nll(r, "recorded") for r in S]); g = [r["traj"] for r in S]
        m, lo, hi, _ = cluster_boot(dv, g)
        mt, tlo, thi, se = cluster_t(dv, g)
        # trajectory-weighted mean (each trajectory equal weight)
        tm = np.mean([dv[np.array(g) == u].mean() for u in np.unique(g)])
        win = np.mean(dv > 0)
        wm, wlo, whi, _ = cluster_boot((dv > 0).astype(float), g)
        lr = 1 / (1 + np.exp(-dv))  # posterior P(recorded | real next frame) under equal priors, recorded vs negated
        print(f"  [{sname:8s}] n={len(S)} G={len(set(g))}  neg ΔNLL {m:+.3f} boot[{lo:+.3f},{hi:+.3f}] clusterT[{tlo:+.3f},{thi:+.3f}] "
              f"trajW {tm:+.3f} median {np.median(dv):+.3f} mean|Δ| {np.mean(np.abs(dv)):.3f} sd {dv.std():.3f} deff {icc_deff(dv,g):.2f} "
              f"win {win:.3f} [{wlo:.2f},{whi:.2f}]  mean P(rec|x) {lr.mean():.4f}")
        if sname == "turning":
            per_run_neg[run] = {(r["traj"], r["k"]): d for r, d in zip(S, dv)}
    # rank shares on turning frames, with CI, and alternative nulls
    S = [r for r in R if turning(r) and all(c in r["scores"] for c in C4)]
    sh = np.array([split_best(r, C4).get("recorded", 0.0) for r in S]); g = [r["traj"] for r in S]
    m, lo, hi, _ = cluster_boot(sh, g)
    # which candidate is first
    firsts = {c: np.mean([split_best(r, C4).get(c, 0.0) for r in S]) for c in C4}
    lasts = {c: np.mean([1.0 if nll(r, c) == max(nll(r, x) for x in C4) else 0.0 for r in S]) for c in C4}
    # share of first among recorded/negated/other_traj only (drop zero)
    C3 = ("recorded", "negated", "other_traj")
    sh3 = np.array([split_best(r, C3).get("recorded", 0.0) for r in S]); m3, lo3, hi3, _ = cluster_boot(sh3, g)
    print(f"  turning first-place share recorded {m:.3f} [{lo:.3f},{hi:.3f}] n={len(S)}; firsts {{{', '.join(f'{c}:{v:.2f}' for c,v in firsts.items())}}}; "
          f"lasts {{{', '.join(f'{c}:{v:.2f}' for c,v in lasts.items())}}}; share among 3 (no zero) {m3:.3f} [{lo3:.3f},{hi3:.3f}] (chance .333)")

print("\n==== pooled across checkpoints (turning, negated) ====")
keys = sorted(set.intersection(*[set(v) for v in per_run_neg.values()]))
M = np.array([[per_run_neg[run][k] for run in RUNS] for k in keys])
g = [k[0] for k in keys]
avg = M.mean(1)
m, lo, hi, _ = cluster_boot(avg, g)
print(f"  mean over six checkpoints per frame: {m:+.3f} [{lo:+.3f},{hi:+.3f}] n={len(keys)}")
C = np.corrcoef(M.T)
print("  frame-level correlation of ΔNLL_neg between checkpoints:")
for i, run in enumerate(RUNS):
    print("   ", run.ljust(7), " ".join(f"{C[i,j]:+.2f}" for j in range(len(RUNS))))
