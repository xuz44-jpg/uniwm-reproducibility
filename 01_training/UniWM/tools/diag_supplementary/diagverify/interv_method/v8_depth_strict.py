from ivcommon import *
from ivcommon import _bin
def acc_t(p, r, tol):
    return p not in (None, "stop") and all(abs(pp - _bin(rr)) <= tt for pp, rr, tt in zip(p, r, tol))
def slope_ci(s, d, g):
    s = np.asarray(s, float); d = np.asarray(d, float); g = np.asarray(g); uniq = np.unique(g); rng = np.random.default_rng(0); sl = []
    for _ in range(4000):
        pick = rng.choice(uniq, len(uniq)); ix = np.concatenate([np.where(g == u)[0] for u in pick]); sl.append(np.polyfit(s[ix], d[ix], 1)[0])
    return np.polyfit(s, d, 1)[0], np.percentile(sl, 2.5), np.percentile(sl, 97.5)
for run, key in (("base", "stop_label_released"), ("stopft", "stop_label_dist")):
    R = load(run); rows = [(r["traj"], st) for r in R for st in r["steps"]]
    print(f"\n== {run}")
    d_all = [acc(st["real"][0], st["recorded"], st[key]) - acc(st["gen"][0], st["recorded"], st[key]) for _, st in rows]
    m, lo, hi = slope_ci([st["s"] for _, st in rows], d_all, [t for t, _ in rows])
    print(f"  all steps, report metric: slope of paired loss per step {m:+.4f} [{lo:+.4f}, {hi:+.4f}]")
    M = [(t, st) for t, st in rows if not st[key]]
    for tol in ((0.05, 0.05, 0.05), (0.1, 0.1, 0.05)):
        line = []
        for b in range(5):
            B = [st for _, st in M if min(st["s"] // 5, 4) == b]
            line.append(f"{np.mean([acc_t(st['real'][0], st['recorded'], tol) for st in B]):.2f}/{np.mean([acc_t(st['gen'][0], st['recorded'], tol) for st in B]):.2f}")
        d = [int(acc_t(st["real"][0], st["recorded"], tol)) - int(acc_t(st["gen"][0], st["recorded"], tol)) for _, st in M]
        m, lo, hi = slope_ci([st["s"] for _, st in M], d, [t for t, _ in M])
        print(f"  move-only tol {tol}: real/gen by bin " + "  ".join(line) + f"; overall paired {f3(cluster_ci(d, [t for t,_ in M]))}; slope {m:+.4f} [{lo:+.4f}, {hi:+.4f}]")
    # |dyaw err| paired, per bin, move-only
    line = []
    for b in range(5):
        B = [st for _, st in M if min(st["s"] // 5, 4) == b and st["real"][0] not in (None, "stop") and st["gen"][0] not in (None, "stop")]
        line.append(f"{np.mean([abs(st['real'][0][2]-_bin(st['recorded'][2])) for st in B]):.3f}/{np.mean([abs(st['gen'][0][2]-_bin(st['recorded'][2])) for st in B]):.3f}")
    print("  move-only mean |dyaw err| real/gen by bin: " + "  ".join(line))
