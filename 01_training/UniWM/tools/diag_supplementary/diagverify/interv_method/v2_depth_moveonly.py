from ivcommon import *
from ivcommon import _bin
for run, key in (("base", "stop_label_released"), ("stopft", "stop_label_dist")):
    R = load(run); rows = [(r["traj"], r["n"], st) for r in R for st in r["steps"]]
    N = len(rows)
    print(f"\n== {run} ({key})")
    # decomposition of the paired loss (sum over steps / N)
    for lab, sel in (("Stop-labelled", lambda st: st[key]), ("move-labelled", lambda st: not st[key])):
        S = [(t, st) for t, n, st in rows if sel(st)]
        d = [acc(st["real"][0], st["recorded"], st[key]) - acc(st["gen"][0], st["recorded"], st[key]) for _, st in S]
        print(f"  {lab}: n={len(S)}  contribution to overall paired loss {sum(d)/N:+.3f}  | mean loss within subset {f3(cluster_ci(d,[t for t,_ in S]))}")
        for name in ("real", "vq", "gen"):
            a = [acc(st[name][0], st["recorded"], st[key]) for _, st in S]
            print(f"      {name} acc {np.mean(a):.3f}")
    # stop steps per depth bin
    print("  depth bin | n all | n Stop-lab | acc all real/gen | acc MOVE-only real/vq/gen (n) | paired loss move-only [CI]")
    for b in range(5):
        B = [(t, n, st) for t, n, st in rows if min(st["s"] // 5, 4) == b]
        M = [(t, n, st) for t, n, st in B if not st[key]]
        aa = lambda S, nm: np.mean([acc(st[nm][0], st["recorded"], st[key]) for _, _, st in S])
        d = [acc(st["real"][0], st["recorded"], st[key]) - acc(st["gen"][0], st["recorded"], st[key]) for _, _, st in M]
        print(f"   {5*b:2d}-{5*b+4:2d} | {len(B):3d} | {sum(st[key] for _,_,st in B):2d} | {aa(B,'real'):.2f}/{aa(B,'gen'):.2f} | "
              f"{aa(M,'real'):.2f}/{aa(M,'vq'):.2f}/{aa(M,'gen'):.2f} ({len(M)}) | {f3(cluster_ci(d,[t for t,_,_ in M]),3)}")
    # overall move-only paired loss
    M = [(t, st) for t, n, st in rows if not st[key]]
    d = [acc(st["real"][0], st["recorded"], st[key]) - acc(st["gen"][0], st["recorded"], st[key]) for _, st in M]
    print(f"  move-only overall paired loss real-gen: {f3(cluster_ci(d,[t for t,_ in M]))} n={len(M)}")
    dv = [acc(st["real"][0], st["recorded"], st[key]) - acc(st["vq"][0], st["recorded"], st[key]) for _, st in M]
    print(f"  move-only overall paired loss real-vq : {f3(cluster_ci(dv,[t for t,_ in M]))}")
    # composition held fixed: only full-length (n=25) trajectories
    F = [(t, n, st) for t, n, st in rows if n == 25 and not st[key]]
    print("  full-length trajectories only (n=25), move-labelled: real/gen per bin:",
          [ (round(np.mean([acc(st['real'][0],st['recorded'],st[key]) for _,_,st in F if min(st['s']//5,4)==b]),2),
             round(np.mean([acc(st['gen'][0],st['recorded'],st[key]) for _,_,st in F if min(st['s']//5,4)==b]),2),
             sum(1 for _,_,st in F if min(st['s']//5,4)==b)) for b in range(5)])
    # move-only depth trend: logistic-free check = slope of per-step loss vs s (cluster bootstrap)
    s = np.array([st["s"] for _, st in M], float); dd = np.array(d, float); g = np.array([t for t, _ in M])
    uniq = np.unique(g); rng = np.random.default_rng(0); sl = []
    for _ in range(4000):
        pick = rng.choice(uniq, len(uniq)); ix = np.concatenate([np.where(g == u)[0] for u in pick])
        sl.append(np.polyfit(s[ix], dd[ix], 1)[0])
    print(f"  move-only: OLS slope of paired loss per step index {np.polyfit(s, dd, 1)[0]:+.4f} [{np.percentile(sl,2.5):+.4f}, {np.percentile(sl,97.5):+.4f}]  (x20 steps = {20*np.polyfit(s, dd, 1)[0]:+.3f})")
