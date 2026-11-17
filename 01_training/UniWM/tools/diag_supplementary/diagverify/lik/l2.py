"""L2: zero-vs-recorded, what drives it (dx vs |dyaw|); dose-response of negated in |dyaw|; per-candidate regression."""
import sys; sys.path.insert(0, __file__.rsplit("/", 1)[0])
from likcommon import *

R0 = [r for r in load_run("base") if not is_goal(r)]
dx = np.array([act(r, "recorded")[0] for r in R0]); dyaw = np.array([r["dyaw_k"] for r in R0])
print("recorded dx: mean %.3f, share dx<0.01 %.3f; on straight frames mean dx %.3f share<0.03 %.3f; on turning frames mean dx %.3f"
      % (dx.mean(), np.mean(dx < 0.01), dx[np.abs(dyaw) < 0.02].mean(), np.mean(dx[np.abs(dyaw) < 0.02] < 0.03), dx[np.abs(dyaw) >= 0.05].mean()))
print("|dyaw| on turning frames: median %.3f, quartiles %.3f %.3f, max %.3f" % tuple(np.percentile(np.abs(dyaw[np.abs(dyaw) >= 0.05]), [50, 25, 75, 100])))
print("frames: straight %d, 0.02<=|dyaw|<0.05 %d, turning %d" % (np.sum(np.abs(dyaw) < 0.02), np.sum((np.abs(dyaw) >= 0.02) & (np.abs(dyaw) < 0.05)), np.sum(np.abs(dyaw) >= 0.05)))

bins = [(0.0, 0.02), (0.02, 0.05), (0.05, 0.10), (0.10, 0.15), (0.15, 0.18), (0.18, 9)]
for run in RUNS:
    R = [r for r in load_run(run) if not is_goal(r)]
    print(f"\n==== {run}")
    for c in ("zero", "negated", "other_traj"):
        cells = []
        for lo, hi in bins:
            S = [r for r in R if lo <= abs(r["dyaw_k"]) < hi and c in r["scores"] and text(r, c) != text(r, "recorded")]
            dv = [nll(r, c) - nll(r, "recorded") for r in S]
            m, l, h, _ = cluster_boot(dv, [r["traj"] for r in S], n_boot=2000)
            cells.append(f"[{lo:.2f},{hi:.2f}) n={len(S):3d} {m:+.2f} [{l:+.2f},{h:+.2f}]")
        print(f"  {c:10s} " + " | ".join(cells))
    # zero on straight frames with clear forward motion
    S = [r for r in R if straight(r) and act(r, "recorded")[0] >= 0.05]
    dv = [nll(r, "zero") - nll(r, "recorded") for r in S]; m, l, h, _ = cluster_boot(dv, [r["traj"] for r in S])
    print(f"  zero on straight frames with dx>=0.05 (moving vs standing, heading unchanged): n={len(S)} {m:+.3f} [{l:+.3f},{h:+.3f}]")
    # other_traj on turning frames split by the other action being straight vs turning
    for lab, cond in (("other straight |dyaw_o|<0.02", lambda a: abs(a[2]) < 0.02), ("other turning same sign", None), ("other turning opp sign", None)):
        pass
    T = [r for r in R if turning(r)]
    for lab, f in (("other |dyaw|<0.02", lambda r: abs(act(r, 'other_traj')[2]) < 0.02),
                   ("other same-sign turn>=0.05", lambda r: abs(act(r, 'other_traj')[2]) >= 0.05 and np.sign(act(r, 'other_traj')[2]) == np.sign(r['dyaw_k'])),
                   ("other opposite-sign turn>=0.05", lambda r: abs(act(r, 'other_traj')[2]) >= 0.05 and np.sign(act(r, 'other_traj')[2]) != np.sign(r['dyaw_k']))):
        S = [r for r in T if f(r)]
        dv = [nll(r, "other_traj") - nll(r, "recorded") for r in S]
        if len(S) > 3:
            m, l, h, _ = cluster_boot(dv, [r["traj"] for r in S], n_boot=2000)
            print(f"  turning frames, {lab:32s} n={len(S):3d} ΔNLL {m:+.3f} [{l:+.3f},{h:+.3f}]")
    # regression over all distinct candidates on turning frames:
    # ΔNLL ~ b0 + b_mag * (|dyaw_rec|-|dyaw_c|)_+ + b_flip * 1[sign flip, |dyaw_c|>=0.02] + b_dx * |dx_rec-dx_c| + b_dy*|dy_rec-dy_c|
    X, y, g = [], [], []
    for r in T:
        a0 = act(r, "recorded")
        for c in ("negated", "zero", "other_traj", "prev1", "prev2", "next1"):
            if c not in r["scores"] or text(r, c) == text(r, "recorded"):
                continue
            a = act(r, c)
            X.append([1.0, max(abs(a0[2]) - abs(a[2]), 0.0), max(abs(a[2]) - abs(a0[2]), 0.0),
                      float(abs(a[2]) >= 0.02 and np.sign(a[2]) != np.sign(a0[2])), abs(a0[0] - a[0]), abs(a0[1] - a[1])])
            y.append(nll(r, c) - nll(r, "recorded")); g.append(r["traj"])
    X = np.array(X); y = np.array(y); g = np.array(g)
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    uniq = np.unique(g); idx = {u: np.where(g == u)[0] for u in uniq}; rng = np.random.default_rng(0); B = []
    for _ in range(1000):
        pick = rng.choice(uniq, len(uniq)); ii = np.concatenate([idx[u] for u in pick])
        B.append(np.linalg.lstsq(X[ii], y[ii], rcond=None)[0])
    B = np.array(B); lo, hi = np.percentile(B, [2.5, 97.5], axis=0)
    names = ["const", "|dyaw| reduced (per rad)", "|dyaw| increased (per rad)", "sign flip", "|Δdx| (per unit)", "|Δdy| (per unit)"]
    print("  regression on turning frames (n=%d candidate rows): " % len(y) + "; ".join(f"{n} {b:+.2f} [{l:+.2f},{h:+.2f}]" for n, b, l, h in zip(names, beta, lo, hi)))
