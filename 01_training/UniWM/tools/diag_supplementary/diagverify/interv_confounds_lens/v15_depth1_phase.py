# depth-1 imagination (variance run, base) vs trajectory phase: separates imagination depth from trajectory phase
from ic_common import *
R = load("base_variance", "variance_decomp_results", "variance_rank*.json")
tolS = (0.05, 0.05, 0.05)
def acc2(p, ref): return p not in (None, 'stop') and all(abs(a - bin_(r)) <= t for a, r, t in zip(p, ref, tolS))
rows_ = []
for r in R:
    if r['stop_next_released'] or r['recorded_next'] is None: continue
    ref = r['recorded_next']
    im = [a for fr in r['imagined'] if fr for a in fr]
    rn = r['real_next']
    rows_.append((r['traj'], r['k'], r['n'], np.mean([acc(a, ref, False) for a in rn]), np.mean([acc(a, ref, False) for a in im]),
                  np.mean([acc2(a, ref) for a in rn]), np.mean([acc2(a, ref) for a in im]),
                  np.mean([turn_ok(a, ref) for a in rn]) if abs(ref[2]) >= TURN else np.nan, np.mean([turn_ok(a, ref) for a in im]) if abs(ref[2]) >= TURN else np.nan))
A = np.array([x[1:] for x in rows_], float); T = np.array([x[0] for x in rows_])
k = A[:, 0]; ph = (k + 1) / (A[:, 1] - 1)
print("depth-1 states (move-labelled next):", len(A), "k range", int(k.min()), int(k.max()))
for lab, m in (("all", np.ones(len(A), bool)), ("next step s=k+1 < 10", k + 1 < 10), ("next step s=k+1 >= 10", k + 1 >= 10), ("next step s>=15", k + 1 >= 15)):
    print("  %-22s n=%3d | report-tol acc real %.2f img %.2f paired %s | strict real %.2f img %.2f paired %s" % (lab, m.sum(), A[m, 2].mean(), A[m, 3].mean(), f3(cluster_ci(A[m, 2] - A[m, 3], T[m])), A[m, 4].mean(), A[m, 5].mean(), f3(cluster_ci(A[m, 4] - A[m, 5], T[m]))))
    mt = m & ~np.isnan(A[:, 6])
    print("  %-22s turn-sign n=%d real %.2f img %.2f paired %s" % ("", mt.sum(), A[mt, 6].mean(), A[mt, 7].mean(), f3(cluster_ci(A[mt, 6] - A[mt, 7], T[mt]))))
# same-phase comparison in the intervention chain (sampled draws, to match): steps s>=10, deep chain
Ri, rw = rows("base")
dd = []; dd2 = []; Tt = []
for t, n, st in rw:
    if st['stop_label_released'] or st['s'] < 10: continue
    ref = st['recorded']
    dd.append(np.mean([acc(a, ref, False) for a in st['real'][1:]]) - np.mean([acc(a, ref, False) for a in st['gen'][1:]]))
    dd2.append(np.mean([acc2(a, ref) for a in st['real'][1:]]) - np.mean([acc2(a, ref) for a in st['gen'][1:]])); Tt.append(t)
print("  intervention chain (depth 10-24, sampled draws), move steps n=%d: paired report-tol %s strict %s" % (len(dd), f3(cluster_ci(dd, Tt)), f3(cluster_ci(dd2, Tt))))
