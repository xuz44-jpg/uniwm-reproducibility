from ic_common import *
for run, key in (("base", "stop_label_released"), ("stopft", "stop_label_dist")):
    R, rw = rows(run)
    print("=====", run)
    # shrinkage: slope of predicted dyaw on binned recorded dyaw (move steps, s>=1)
    for nm in ("real", "vq", "gen"):
        P = [(st[nm][0][2], bin_(st['recorded'][2]), st['s']) for t, n, st in rw if st['recorded'] is not None and not st[key] and st[nm][0] not in (None, 'stop') and st['s'] >= 1]
        P = np.array(P)
        sl = np.polyfit(P[:, 1], P[:, 0], 1)[0]
        sl_e = np.polyfit(P[P[:, 2] < 10][:, 1], P[P[:, 2] < 10][:, 0], 1)[0]; sl_l = np.polyfit(P[P[:, 2] >= 10][:, 1], P[P[:, 2] >= 10][:, 0], 1)[0]
        Px = np.array([(st[nm][0][0], bin_(st['recorded'][0])) for t, n, st in rw if st['recorded'] is not None and not st[key] and st[nm][0] not in (None, 'stop') and st['s'] >= 1])
        print("  %s: slope pred dyaw ~ recorded dyaw %.2f (s1-9 %.2f, s10+ %.2f); corr %.2f | slope pred dx ~ recorded dx %.2f corr %.2f" % (nm, sl, sl_e, sl_l, np.corrcoef(P[:, 0], P[:, 1])[0, 1], np.polyfit(Px[:, 1], Px[:, 0], 1)[0], np.corrcoef(Px[:, 0], Px[:, 1])[0, 1]))
    # lag test: compare gen greedy at s with recorded label at s+j
    for comp, ci in (("dyaw", 2), ("dx", 0), ("dy", 1)):
        line = []
        for j in (-3, -2, -1, 0, 1, 2, 3):
            e_gen, e_real = [], []
            for r in R:
                S = {st['s']: st for st in r['steps']}
                for s, st in S.items():
                    if s < 4 or st[key]: continue       # common support: s>=4 so that s-3 exists
                    o = S.get(s + j)
                    if o is None or o['recorded'] is None or o[key]: continue
                    for nm, E in (("gen", e_gen), ("real", e_real)):
                        p = st[nm][0]
                        if p in (None, 'stop'): continue
                        E.append(abs(p[ci] - bin_(o['recorded'][ci])))
            line.append("j=%+d gen %.4f real %.4f" % (j, np.mean(e_gen), np.mean(e_real)))
        print("  %s |pred[s] - label[s+j]|: " % comp + "; ".join(line))
    # dx bias by depth on gen vs real
    for nm in ("real", "gen"):
        E = [(st['s'], st[nm][0][0] - bin_(st['recorded'][0])) for t, n, st in rw if st['recorded'] is not None and not st[key] and st[nm][0] not in (None, 'stop')]
        E = np.array(E); B = np.minimum(E[:, 0] // 5, 4)
        print("  %s dx bias by bin: %s" % (nm, " ".join("%+.3f" % E[B == b, 1].mean() for b in range(5))))
