from ic_common import *
tol = (0.05, 0.05, 0.05)
def acc2(p, ref): return p not in (None, 'stop') and all(abs(a - bin_(r)) <= t for a, r, t in zip(p, ref, tol))
for run, key in (("base", "stop_label_released"), ("stopft", "stop_label_dist")):
    R, rw = rows(run)
    mv = [(t, st) for t, n, st in rw if not st[key]]
    T = np.array([t for t, _ in mv]); S = np.array([st['s'] for _, st in mv])
    a = {nm: np.array([acc2(st[nm][0], st['recorded']) for _, st in mv], float) for nm in ("real", "gen")}
    d = a['real'] - a['gen']
    print("=====", run, "strict tol", tol)
    for lo, hi in ((1, 5), (5, 10), (10, 15), (15, 20), (20, 25), (1, 10), (10, 25)):
        m = (S >= lo) & (S < hi)
        print("  s[%2d,%2d) real %.2f gen %.2f paired %s n=%d" % (lo, hi, a['real'][m].mean(), a['gen'][m].mean(), f3(cluster_ci(d[m], T[m])), m.sum()))
    # continuous |err| sum over components, paired, s<10 vs >=10
    P = [(t, st) for t, st in mv if st['real'][0] not in (None, 'stop') and st['gen'][0] not in (None, 'stop')]
    Tp = np.array([t for t, _ in P]); Sp = np.array([st['s'] for _, st in P])
    for ci, c in enumerate(("dx", "dy", "dyaw")):
        dd = np.array([abs(st['gen'][0][ci] - bin_(st['recorded'][ci])) - abs(st['real'][0][ci] - bin_(st['recorded'][ci])) for _, st in P])
        print("  paired |err| gen-real %-4s s1-9 %s | s10+ %s" % (c, f3(cluster_ci(dd[(Sp >= 1) & (Sp < 10)], Tp[(Sp >= 1) & (Sp < 10)])), f3(cluster_ci(dd[Sp >= 10], Tp[Sp >= 10]))))
