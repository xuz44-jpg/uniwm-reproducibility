from ivcommon import *
for run, key in (("base", "stop_label_released"), ("stopft", "stop_label_dist")):
    R = load(run); rows = [(r["traj"], st) for r in R for st in r["steps"]]
    print(f"== {run} ({key}) trajs {len(R)} steps {len(rows)} stop-labelled {sum(st[key] for _,st in rows)}")
    print("   gen unavailable:", sum(not st['gen_available'] for _,st in rows), " n per traj:", sorted(r['n'] for r in R))
    for name in ("real", "vq", "gen"):
        g = [st[name][0] for _, st in rows]
        a = [acc(p, st["recorded"], st[key]) for p, (_, st) in zip(g, rows)]
        malformed = sum(p is None for p in g)
        nstop = sum(p == "stop" for p in g)
        print(f"   {name}: acc {f3(cluster_ci(a,[t for t,_ in rows]),3)} malformed {malformed} stop-preds {nstop}")
    for other in ("gen", "vq"):
        d = [acc(st["real"][0], st["recorded"], st[key]) - acc(st[other][0], st["recorded"], st[key]) for _, st in rows]
        print(f"   paired real-{other}: {f3(cluster_ci(d,[t for t,_ in rows]))}")
