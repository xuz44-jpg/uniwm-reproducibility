"""L3: label-timing test on mid-turn frames: shares with CIs, candidate separation (power), magnitude confound."""
import sys; sys.path.insert(0, __file__.rsplit("/", 1)[0])
from likcommon import *

CL = ("prev2", "prev1", "recorded", "next1")
R0 = [r for r in load_run("base") if not is_goal(r) and midturn(r) and all(c in r["scores"] for c in CL)]
# separation of candidates in dyaw (radians) vs the sign flip
sep = {c: np.array([abs(act(r, c)[2] - r["dyaw_k"]) for r in R0]) for c in ("prev2", "prev1", "next1")}
flip = np.array([2 * abs(r["dyaw_k"]) for r in R0])
print(f"mid-turn frames with all four labels: {len(R0)} ({len(set(r['traj'] for r in R0))} traj)")
for c, v in sep.items():
    print(f"  |dyaw_{c} - dyaw_k|: median {np.median(v):.3f} mean {v.mean():.3f} rad; share <0.02 rad {np.mean(v < 0.02):.2f}; same sign as dyaw_k {np.mean([np.sign(act(r, c)[2]) == np.sign(r['dyaw_k']) for r in R0]):.2f}")
print(f"  negation separation 2|dyaw_k|: median {np.median(flip):.3f} rad")
print(f"  frames with 4 distinct texts: {sum(len({text(r, c) for c in CL}) == 4 for r in R0)}")
# spread of |dyaw| across the four candidates
spread = np.array([max(abs(act(r, c)[2]) for c in CL) - min(abs(act(r, c)[2]) for c in CL) for r in R0])
print(f"  within-frame range of |dyaw| across the four labels: median {np.median(spread):.3f}, mean {spread.mean():.3f} rad")

for run in RUNS:
    R = [r for r in load_run(run) if not is_goal(r) and midturn(r) and all(c in r["scores"] for c in CL)]
    g = [r["traj"] for r in R]
    out = []
    for c in CL:
        v = [split_best(r, CL).get(c, 0.0) for r in R]
        m, lo, hi, _ = cluster_boot(v, g, n_boot=2000)
        out.append(f"{c} {m:.2f} [{lo:.2f},{hi:.2f}]")
    # does the candidate with the largest |dyaw| win more often? (magnitude-only model predicts yes)
    big = []
    for r in R:
        mags = {c: abs(act(r, c)[2]) for c in CL}
        mx = max(mags.values()); top = [c for c in CL if mags[c] == mx]
        sb = split_best(r, CL)
        big.append(sum(sb.get(c, 0.0) for c in top) / 1.0)
    # expected share of the top-|dyaw| label(s) under no information = len(top)/4 on average
    exp_big = np.mean([len([c for c in CL if abs(act(r, c)[2]) == max(abs(act(r, x)[2]) for x in CL)]) / 4 for r in R])
    mb, lb, hb, _ = cluster_boot(big, g, n_boot=2000)
    # correlation between ΔNLL(c) and |dyaw_c|-|dyaw_k| across the three lag candidates
    xs, ys, gg = [], [], []
    for r in R:
        for c in ("prev2", "prev1", "next1"):
            if text(r, c) != text(r, "recorded"):
                xs.append(abs(act(r, c)[2]) - abs(r["dyaw_k"])); ys.append(nll(r, c) - nll(r, "recorded")); gg.append(r["traj"])
    xs = np.array(xs); ys = np.array(ys)
    slope = np.polyfit(xs, ys, 1)[0]
    gg = np.array(gg); uniq = np.unique(gg); idx = {u: np.where(gg == u)[0] for u in uniq}; rng = np.random.default_rng(1); B = []
    for _ in range(1000):
        ii = np.concatenate([idx[u] for u in rng.choice(uniq, len(uniq))]); B.append(np.polyfit(xs[ii], ys[ii], 1)[0])
    sl, sh = np.percentile(B, [2.5, 97.5])
    # pairwise prev1 vs recorded
    dv = [nll(r, "prev1") - nll(r, "recorded") for r in R if text(r, "prev1") != text(r, "recorded")]
    gp = [r["traj"] for r in R if text(r, "prev1") != text(r, "recorded")]
    m1, l1, h1, _ = cluster_boot(dv, gp, n_boot=2000)
    print(f"{run:7s} n={len(R)}: " + "; ".join(out) + f" | largest-|dyaw| label wins {mb:.2f} [{lb:.2f},{hb:.2f}] (null {exp_big:.2f})"
          f" | slope ΔNLL vs (|dyaw_c|-|dyaw_k|) {slope:+.2f} [{sl:+.2f},{sh:+.2f}] nats/rad | prev1-rec {m1:+.3f} [{l1:+.3f},{h1:+.3f}]")
