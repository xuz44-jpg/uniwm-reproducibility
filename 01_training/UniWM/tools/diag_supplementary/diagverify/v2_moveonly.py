import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "interv_confounds_lens"))  # helper module this script was written against
from ic_common import *
for run, key in (("base", "stop_label_released"), ("stopft", "stop_label_dist")):
    R, rw = rows(run)
    print("=====", run, key)
    A = {nm: np.array([acc(st[nm][0], st["recorded"], st[key]) for _, _, st in rw], float) for nm in ("real", "vq", "gen")}
    T = np.array([t for t, _, _ in rw]); S = np.array([st["s"] for _, _, st in rw]); STOP = np.array([st[key] for _, _, st in rw])
    B = np.minimum(S // 5, 4)
    # reproduce
    for nm in A: print(nm, "acc all %.3f" % A[nm].mean())
    d = A["real"] - A["gen"]
    print("paired real-gen all:", f3(cluster_ci(d, T)), "n", len(d))
    print("  contribution of Stop-labelled steps to the mean loss: %.3f (sum %.1f over %d steps); move steps: %.3f" % (d[STOP].sum() / len(d), d[STOP].sum(), STOP.sum(), d[~STOP].sum() / len(d)))
    mv = ~STOP
    print("paired real-gen move-only:", f3(cluster_ci(d[mv], T[mv])), "n", mv.sum())
    print("paired real-vq move-only:", f3(cluster_ci((A['real'] - A['vq'])[mv], T[mv])))
    print("paired real-gen Stop-only:", f3(cluster_ci(d[STOP], T[STOP])), "n", STOP.sum())
    for nm in A: print("  %s move-only acc %.3f  %s" % (nm, A[nm][mv].mean(), f3(cluster_ci(A[nm][mv], T[mv]), 2)))
    print("  depth bins move-only: real / vq / gen (n) | paired real-gen [CI]")
    for b in range(5):
        m = mv & (B == b)
        print("   bin %d: %.2f / %.2f / %.2f (n=%d, trajs %d) | %s" % (b, A['real'][m].mean(), A['vq'][m].mean(), A['gen'][m].mean(), m.sum(), len(set(T[m])), f3(cluster_ci(d[m], T[m]))))
    print("  depth bins Stop-only: real / gen (n)")
    for b in range(5):
        m = STOP & (B == b)
        if m.sum(): print("   bin %d: %.2f / %.2f (n=%d)" % (b, A['real'][m].mean(), A['gen'][m].mean(), m.sum()))
    # trend test: slope of paired move-only loss vs s, cluster bootstrap
    x = S[mv]; y = d[mv]; tt = T[mv]
    def slope(x, y): return np.polyfit(x, y, 1)[0]
    rng = np.random.default_rng(0); U = np.unique(tt); idx = {u: np.where(tt == u)[0] for u in U}; bs = []
    for _ in range(2000):
        p = rng.choice(U, len(U)); ii = np.concatenate([idx[u] for u in p]); bs.append(slope(x[ii], y[ii]))
    print("  move-only paired loss slope per step %.4f [%.4f, %.4f]" % (slope(x, y), *np.percentile(bs, [2.5, 97.5])))
    # early vs late
    for lo, hi in ((0, 10), (10, 25), (15, 25)):
        m = mv & (S >= lo) & (S < hi)
        print("  move-only s in [%d,%d): paired %s n=%d" % (lo, hi, f3(cluster_ci(d[m], T[m])), m.sum()))
    # common-support: only trajectories with n==25 (fixed composition)
    full = np.array([n == 25 for _, n, _ in rw])
    print("  n==25 trajectories:", len(set(T[full])))
    for b in range(5):
        m = mv & (B == b) & full
        print("   full-length bin %d move-only: %.2f / %.2f (n=%d)" % (b, A['real'][m].mean(), A['gen'][m].mean(), m.sum()))
