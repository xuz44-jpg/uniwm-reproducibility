# lag profile with per-trajectory demeaning (removes between-trajectory speed/turn differences)
from ic_common import *
JS = (-8, -6, -4, -3, -2, -1, 0, 1, 2)
for run, key in (("base", "stop_label_released"), ("stopft", "stop_label_dist")):
    R, rw = rows(run)
    print("=====", run)
    for comp, ci in (("dyaw", 2), ("dx", 0)):
        for ref in ("label", "realpred"):
            G = {j: [] for j in JS}; Rr = {j: [] for j in JS}
            for r in R:
                S = {st['s']: st for st in r['steps']}
                loc = {j: [] for j in JS}
                for s, st in S.items():
                    if s < 8 or s >= 23: continue
                    if not all((s + j) in S and S[s + j]['recorded'] is not None and not S[s + j][key] and S[s + j]['real'][0] not in (None, 'stop') for j in JS): continue
                    if st['gen'][0] in (None, 'stop'): continue
                    for j in JS:
                        o = S[s + j]; y = bin_(o['recorded'][ci]) if ref == 'label' else o['real'][0][ci]
                        loc[j].append((st['gen'][0][ci], y, st['real'][0][ci]))
                if len(loc[0]) < 3: continue
                for j in JS:
                    a = np.array(loc[j]); a = a - a.mean(0)
                    G[j] += list(zip(a[:, 0], a[:, 1])); Rr[j] += list(zip(a[:, 2], a[:, 1]))
            cg = [np.corrcoef(*np.array(G[j]).T)[0, 1] for j in JS]; cr = [np.corrcoef(*np.array(Rr[j]).T)[0, 1] for j in JS]
            print("  %-4s vs %-8s within-traj n=%d corr gen[s]: %s | real[s]: %s" % (comp, ref, len(G[0]), " ".join("%+.2f" % c for c in cg), " ".join("%+.2f" % c for c in cr)))
    print("   (j = -8,-6,-4,-3,-2,-1,0,1,2)")
