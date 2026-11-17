from ic_common import *
JS = (-8, -6, -4, -3, -2, -1, 0, 1, 2)
for run, key in (("base", "stop_label_released"), ("stopft", "stop_label_dist")):
    R, rw = rows(run)
    print("=====", run)
    for comp, ci in (("dyaw", 2), ("dx", 0)):
        for ref in ("label", "realpred"):
            for smin, smax in ((8, 23),):
                out = []
                # common support: s in [smin,smax), s-3..s+3 all move steps with numeric preds
                data = {j: ([], [], []) for j in JS}
                for r in R:
                    S = {st['s']: st for st in r['steps']}
                    for s, st in S.items():
                        if s < smin or s >= smax: continue
                        okk = all((s + j) in S and S[s + j]['recorded'] is not None and not S[s + j][key] and S[s + j]['real'][0] not in (None, 'stop') for j in JS)
                        if not okk or st['gen'][0] in (None, 'stop') or st['real'][0] in (None, 'stop'): continue
                        for j in JS:
                            o = S[s + j]
                            y = bin_(o['recorded'][ci]) if ref == "label" else o['real'][0][ci]
                            data[j][0].append(st['gen'][0][ci]); data[j][1].append(y); data[j][2].append(st['real'][0][ci])
                n = len(data[0][0])
                cg = [np.corrcoef(data[j][0], data[j][1])[0, 1] for j in JS]
                cr = [np.corrcoef(data[j][2], data[j][1])[0, 1] if not (ref == "realpred" and j == 0) else 1.0 for j in JS]
                print("  %-4s vs %-8s s[%2d,%2d) n=%3d  corr gen[s]: %s | corr real[s]: %s" % (comp, ref, smin, smax, n, " ".join("%+.2f" % c for c in cg), " ".join("%+.2f" % c for c in cr)))
    print("   (columns j = -8,-6,-4,-3,-2,-1,0,1,2: reference taken at step s+j)")
