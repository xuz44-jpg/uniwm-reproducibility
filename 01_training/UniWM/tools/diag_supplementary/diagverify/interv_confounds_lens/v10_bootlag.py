# cluster-bootstrap CI of corr(gen[s], real[s-3]) - corr(gen[s], real[s]) and the same for labels
from ic_common import *
for run, key in (("base", "stop_label_released"), ("stopft", "stop_label_dist")):
    R, rw = rows(run)
    for comp, ci in (("dyaw", 2), ("dx", 0)):
        recs = []
        for r in R:
            S = {st['s']: st for st in r['steps']}
            for s, st in S.items():
                if s < 4: continue
                if not all((s + j) in S and S[s + j]['recorded'] is not None and not S[s + j][key] and S[s + j]['real'][0] not in (None, 'stop') for j in (-3, 0, 1)): continue
                if st['gen'][0] in (None, 'stop'): continue
                recs.append((r['traj'], st['gen'][0][ci], S[s - 3]['real'][0][ci], S[s]['real'][0][ci], bin_(S[s - 3]['recorded'][ci]), bin_(S[s]['recorded'][ci])))
        T = np.array([x[0] for x in recs]); A = np.array([x[1:] for x in recs], float)
        def stat(ii):
            a = A[ii]; c = lambda u, v: np.corrcoef(u, v)[0, 1]
            return c(a[:, 0], a[:, 1]) - c(a[:, 0], a[:, 2]), c(a[:, 0], a[:, 3]) - c(a[:, 0], a[:, 4])
        U = np.unique(T); idx = {u: np.where(T == u)[0] for u in U}; rng = np.random.default_rng(0)
        bs = np.array([stat(np.concatenate([idx[u] for u in rng.choice(U, len(U))])) for _ in range(2000)])
        p = stat(np.arange(len(A)))
        print("%s %s n=%d: corr(gen[s],realpred[s-3]) - corr(gen[s],realpred[s]) = %+.3f [%+.3f,%+.3f]; corr(gen[s],label[s-3]) - corr(gen[s],label[s]) = %+.3f [%+.3f,%+.3f]" % (run, comp, len(A), p[0], *np.percentile(bs[:, 0], [2.5, 97.5]), p[1], *np.percentile(bs[:, 1], [2.5, 97.5])))
