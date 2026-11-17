from ic_common import *
import collections
for run, key in (("base", "stop_label_released"), ("stopft", "stop_label_dist")):
    R, rw = rows(run)
    print("=====", run, key)
    st_ = [(t, n, st) for t, n, st in rw if st[key]]
    for lab, cond in (("goal frame s=n-1", lambda n, s: s == n - 1), ("non-goal Stop-labelled s<n-1", lambda n, s: s < n - 1)):
        sub = [(t, n, st) for t, n, st in st_ if cond(n, st['s'])]
        if not sub: continue
        for nm in ("real", "vq", "gen"):
            g = [st[nm][0] == "stop" for _, _, st in sub]
            smp = [a == "stop" for _, _, st in sub for a in st[nm][1:]]
            print("  %s n=%d: %s greedy Stop %.2f (%d/%d); sampled Stop rate %.2f" % (lab, len(sub), nm, np.mean(g), sum(g), len(g), np.mean(smp)))
    # any-Stop within the last k steps on gen (would an agent stop "around" the goal?)
    for nm in ("real", "gen"):
        by = collections.defaultdict(list)
        for t, n, st in rw:
            by[t].append((st['s'], n, st[nm][0] == 'stop', any(a == 'stop' for a in st[nm]), st[key]))
        first = []
        for t, L in by.items():
            n = L[0][1]
            fs = [s for s, _, g, _, _ in L if g]
            fa = [s for s, _, _, a, _ in L if a]
            first.append((n - 1 - (fs[0] if fs else np.nan), n - 1 - (fa[0] if fa else np.nan)))
        f = np.array(first)
        print("  %s: trajectories with any greedy Stop %d/24, with any Stop among 6 draws at some step %d/24; steps-before-goal of first greedy Stop: %s" % (nm, np.sum(~np.isnan(f[:, 0])), np.sum(~np.isnan(f[:, 1])), sorted(f[~np.isnan(f[:, 0]), 0].astype(int).tolist())))
    # gen at goal frame: what does it predict? magnitude of move
    g = [st['gen'][0] for t, n, st in rw if st['s'] == n - 1]
    num = np.array([p for p in g if p not in (None, 'stop')])
    print("  gen greedy at goal frame: mean |dx| %.3f |dy| %.3f |dyaw| %.3f (n=%d numeric)" % (*abs(num).mean(0), len(num)))
    r_ = [st['real'][0] for t, n, st in rw if st['s'] == n - 2]
    print("  real greedy at s=n-2 (one step before goal): Stop %d/%d" % (sum(p == 'stop' for p in r_), len(r_)))
