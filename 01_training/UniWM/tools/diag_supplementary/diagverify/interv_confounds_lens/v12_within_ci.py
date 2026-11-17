from ic_common import *
JS = (-6, 0)
for run, key in (("base", "stop_label_released"), ("stopft", "stop_label_dist")):
    R, rw = rows(run)
    for comp, ci in (("dyaw", 2), ("dx", 0)):
        per = {}
        for r in R:
            S = {st['s']: st for st in r['steps']}; loc = []
            for s, st in S.items():
                if s < 8 or s >= 23: continue
                if not all((s + j) in S and S[s + j]['recorded'] is not None and not S[s + j][key] and S[s + j]['real'][0] not in (None, 'stop') for j in (-8, -6, -4, -3, -2, -1, 0, 1, 2)): continue
                if st['gen'][0] in (None, 'stop'): continue
                loc.append((st['gen'][0][ci], st['real'][0][ci], bin_(S[s]['recorded'][ci]), bin_(S[s - 6]['recorded'][ci])))
            if len(loc) >= 3:
                a = np.array(loc); per[r['traj']] = a - a.mean(0)
        keys = list(per)
        def stat(ks):
            a = np.concatenate([per[k] for k in ks]); c = lambda u, v: np.corrcoef(u, v)[0, 1]
            return c(a[:, 0], a[:, 2]), c(a[:, 1], a[:, 2]), c(a[:, 0], a[:, 3])
        rng = np.random.default_rng(0)
        bs = np.array([stat(rng.choice(keys, len(keys))) for _ in range(2000)])
        p = stat(keys)
        print("%s %s (%d trajs): within-traj corr with label[s]: gen %+.2f [%+.2f,%+.2f], real %+.2f [%+.2f,%+.2f]; gen with label[s-6] %+.2f [%+.2f,%+.2f]" % (run, comp, len(keys), p[0], *np.percentile(bs[:, 0], [2.5, 97.5]), p[1], *np.percentile(bs[:, 1], [2.5, 97.5]), p[2], *np.percentile(bs[:, 2], [2.5, 97.5])))
