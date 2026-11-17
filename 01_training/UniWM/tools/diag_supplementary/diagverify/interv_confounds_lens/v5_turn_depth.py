from ic_common import *
for run, key in (("base", "stop_label_released"), ("stopft", "stop_label_dist")):
    R, rw = rows(run)
    tr = [(t, st) for t, n, st in rw if st["recorded"] is not None and not st[key] and abs(st["recorded"][2]) >= TURN]
    T = np.array([t for t, _ in tr]); S = np.array([st["s"] for _, st in tr])
    ok = {nm: np.array([turn_ok(st[nm][0], st["recorded"]) for _, st in tr], float) for nm in ("real", "vq", "gen")}
    d = ok['real'] - ok['gen']
    print("=====", run)
    for lo, hi in ((0, 1), (1, 5), (5, 10), (10, 15), (15, 20), (20, 25), (1, 10), (10, 25)):
        m = (S >= lo) & (S < hi)
        if m.sum(): print("  turn-sign s in [%d,%d): real %.2f vq %.2f gen %.2f paired real-gen %s n=%d" % (lo, hi, ok['real'][m].mean(), ok['vq'][m].mean(), ok['gen'][m].mean(), f3(cluster_ci(d[m], T[m])), m.sum()))
    # move-step accuracy (report tolerance) s=0 excluded (gen==vq there)
    mv = [(t, st) for t, n, st in rw if not st[key]]
    Tm = np.array([t for t, _ in mv]); Sm = np.array([st['s'] for _, st in mv])
    a = {nm: np.array([acc(st[nm][0], st['recorded'], False) for _, st in mv], float) for nm in ("real", "gen")}
    for lo, hi in ((1, 10), (10, 25)):
        m = (Sm >= lo) & (Sm < hi)
        print("  move acc s in [%d,%d): paired %s n=%d" % (lo, hi, f3(cluster_ci((a['real'] - a['gen'])[m], Tm[m])), m.sum()))
