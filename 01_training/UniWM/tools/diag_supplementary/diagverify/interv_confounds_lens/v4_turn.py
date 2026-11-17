from ic_common import *
from ic_common import bin_ as _bin
for run, key in (("base", "stop_label_released"), ("stopft", "stop_label_dist")):
    R, rw = rows(run)
    print("=====", run)
    # cumulative recorded yaw before step s (the rotation the imagination was asked to render)
    cum = {}
    for r in R:
        c = 0.0
        for st in r["steps"]:
            cum[(r["traj"], st["s"])] = c
            if st["recorded"] is not None: c += st["recorded"][2]
    tr = [(t, st) for t, n, st in rw if st["recorded"] is not None and not st[key] and abs(st["recorded"][2]) >= TURN]
    T = np.array([t for t, _ in tr]); S = np.array([st["s"] for _, st in tr]); B = np.minimum(S // 5, 4)
    ok = {nm: np.array([turn_ok(st[nm][0], st["recorded"]) for _, st in tr], float) for nm in ("real", "vq", "gen")}
    print(" turn steps n=%d; real %.3f vq %.3f gen %.3f" % (len(tr), ok['real'].mean(), ok['vq'].mean(), ok['gen'].mean()))
    print(" paired turn-sign real-gen %s; real-vq %s" % (f3(cluster_ci(ok['real'] - ok['gen'], T)), f3(cluster_ci(ok['real'] - ok['vq'], T))))
    print(" by bin real/gen (n):", " ".join("%.2f/%.2f(%d)" % (ok['real'][B == b].mean(), ok['gen'][B == b].mean(), (B == b).sum()) for b in range(5)))
    def kind(p, ref):
        if p is None: return "unparsed"
        if p == "stop": return "stop"
        if abs(p[2]) < TURN / 2: return "no-turn"
        return "ok" if np.sign(p[2]) == np.sign(ref[2]) else "wrong-sign"
    import collections
    for nm in ("real", "gen"):
        print("  %s failure kinds:" % nm, dict(collections.Counter(kind(st[nm][0], st["recorded"]) for _, st in tr)))
    # onset vs mid-turn: previous recorded dyaw same sign and >= TURN
    prevd = {}
    for r in R:
        for i, st in enumerate(r["steps"]):
            prevd[(r["traj"], st["s"])] = r["steps"][i - 1]["recorded"][2] if i >= 1 else None
    mid = np.array([prevd[(t, st["s"])] is not None and abs(prevd[(t, st["s"])]) >= TURN and np.sign(prevd[(t, st["s"])]) == np.sign(st["recorded"][2]) for t, st in tr])
    for lab, m in (("mid-turn", mid), ("onset/other", ~mid)):
        print("  %s n=%d: real %.2f gen %.2f paired %s" % (lab, m.sum(), ok['real'][m].mean(), ok['gen'][m].mean(), f3(cluster_ci((ok['real'] - ok['gen'])[m], T[m]))))
    # does predicted dyaw on gen minus real track cumulative commanded yaw? (move steps, numeric)
    mv = [(t, st) for t, n, st in rw if st["recorded"] is not None and not st[key] and st['real'][0] not in (None, 'stop') and st['gen'][0] not in (None, 'stop') and st['s'] > 0]
    dpy = np.array([st['gen'][0][2] - st['real'][0][2] for _, st in mv]); cy = np.array([cum[(t, st['s'])] for t, st in mv])
    rec = np.array([st['recorded'][2] for _, st in mv]); Tm = np.array([t for t, _ in mv]); Sm = np.array([st['s'] for _, st in mv])
    print("  corr(pred dyaw gen-real, cumulative commanded yaw) = %.3f (n=%d)" % (np.corrcoef(dpy, cy)[0, 1], len(dpy)))
    print("  corr(pred dyaw gen-real, recorded dyaw[s]) = %.3f" % np.corrcoef(dpy, rec)[0, 1])
    # regression dpy ~ cum yaw + rec, cluster bootstrap on coef of cum
    X = np.c_[np.ones_like(cy), cy, rec]
    beta = np.linalg.lstsq(X, dpy, rcond=None)[0]
    U = np.unique(Tm); idx = {u: np.where(Tm == u)[0] for u in U}; rng = np.random.default_rng(0); bs = []
    for _ in range(2000):
        ii = np.concatenate([idx[u] for u in rng.choice(U, len(U))]); bs.append(np.linalg.lstsq(X[ii], dpy[ii], rcond=None)[0])
    bs = np.array(bs)
    print("  OLS dpy = a + b*cumyaw + c*rec: b=%.3f [%.3f,%.3f] c=%.3f [%.3f,%.3f]" % (beta[1], *np.percentile(bs[:, 1], [2.5, 97.5]), beta[2], *np.percentile(bs[:, 2], [2.5, 97.5])))
    # accuracy loss moderated by |cum yaw| within depth (s>=5)
    A = lambda nm: np.array([acc(st[nm][0], st['recorded'], False) for _, st in mv], float)
    d = A('real') - A('gen')
    for lo, hi in ((5, 15), (15, 25)):
        m = (Sm >= lo) & (Sm < hi); med = np.median(abs(cy[m]))
        for lab, mm in (("|cumyaw|<=med", m & (abs(cy) <= med)), ("|cumyaw|>med", m & (abs(cy) > med))):
            print("   s in [%d,%d) %s(%.2f): paired acc loss %s n=%d" % (lo, hi, lab, med, f3(cluster_ci(d[mm], Tm[mm])), mm.sum()))
