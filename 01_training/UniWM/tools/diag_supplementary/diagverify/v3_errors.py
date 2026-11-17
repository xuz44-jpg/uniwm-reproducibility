import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "interv_confounds_lens"))  # helper module this script was written against
from ic_common import *
from ic_common import bin_ as _bin
import collections
def why(pred, ref):
    if pred is None: return "unparsed"
    if pred == "stop": return "falseStop"
    f = [abs(p - _bin(r)) > t for p, r, t in zip(pred, ref, TOL)]
    return "+".join(c for c, x in zip(("dx", "dy", "dyaw"), f) if x) or "ok"
for run, key in (("base", "stop_label_released"), ("stopft", "stop_label_dist")):
    R, rw = rows(run)
    print("=====", run)
    mv = [(t, n, st) for t, n, st in rw if not st[key]]
    for nm in ("real", "vq", "gen"):
        c = collections.Counter(why(st[nm][0], st["recorded"]) for _, _, st in mv)
        print(" ", nm, "move-step outcome:", dict(c))
    # per-component signed bias and abs error on move steps (greedy, numeric preds)
    for nm in ("real", "gen"):
        E = np.array([[p - _bin(r) for p, r in zip(st[nm][0], st["recorded"])] for _, _, st in mv if st[nm][0] not in (None, "stop")])
        print("  %s bias dx %+.3f dy %+.3f dyaw %+.3f | MAE dx %.3f dy %.3f dyaw %.3f" % (nm, *E.mean(0), *abs(E).mean(0)))
    # trivial predictors on move steps
    rec = np.array([st["recorded"] for _, _, st in mv])
    for name, p in (("zero", [0, 0, 0]), ("const median", [float(np.median(rec[:, 0])), 0.0, 0.0])):
        a = np.mean([acc(p, st["recorded"], False) for _, _, st in mv])
        print("  trivial %s %s: move-step accuracy %.3f" % (name, np.round(p, 3), a))
    # accuracy with tighter tolerances
    for tol in ((0.05, 0.05, 0.05), (0.05, 0.05, 0.1)):
        def acc2(pred, ref):
            return pred not in (None, "stop") and all(abs(p - _bin(r)) <= t for p, r, t in zip(pred, ref, tol))
        T = np.array([t for t, _, _ in mv])
        a = {nm: np.array([acc2(st[nm][0], st["recorded"]) for _, _, st in mv], float) for nm in ("real", "vq", "gen")}
        print("  tol %s move-only: real %.3f vq %.3f gen %.3f; paired real-gen %s; real-vq %s" % (tol, a['real'].mean(), a['vq'].mean(), a['gen'].mean(), f3(cluster_ci(a['real'] - a['gen'], T)), f3(cluster_ci(a['real'] - a['vq'], T))))
        S = np.array([st['s'] for _, _, st in mv]); B = np.minimum(S // 5, 4)
        print("    by bin real/gen:", " ".join("%.2f/%.2f" % (a['real'][B == b].mean(), a['gen'][B == b].mean()) for b in range(5)))
    # continuous per-component paired abs error (gen - real), move steps, numeric both
    P = [(t, st) for t, _, st in mv if st['real'][0] not in (None, 'stop') and st['gen'][0] not in (None, 'stop')]
    T = np.array([t for t, _ in P]); S = np.array([st['s'] for _, st in P]); B = np.minimum(S // 5, 4)
    for ci, c in enumerate(("dx", "dy", "dyaw")):
        dd = np.array([abs(st['gen'][0][ci] - _bin(st['recorded'][ci])) - abs(st['real'][0][ci] - _bin(st['recorded'][ci])) for _, st in P])
        print("  paired |err| gen-real %s: %s ; by bin %s" % (c, f3(cluster_ci(dd, T)), " ".join("%+.3f" % dd[B == b].mean() for b in range(5))))
