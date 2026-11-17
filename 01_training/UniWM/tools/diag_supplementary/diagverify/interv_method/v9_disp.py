from ivcommon import *
for run, key in (("base", "stop_label_released"), ("stopft", "stop_label_dist")):
    R = load(run); rows = [st for r in R for st in r["steps"]]
    out = []
    for name in ("real", "vq", "gen"):
        v = []
        for st in rows:
            S = [a for a in st[name][1:] if a not in (None, "stop")]
            if len(S) >= 2: v.append(np.var(np.array(S), axis=0, ddof=1))
        v = np.array(v); out.append(f"{name}: var dx {v[:,0].mean():.5f} dy {v[:,1].mean():.5f} dyaw {v[:,2].mean():.5f} (n={len(v)})")
    print(run, "| within-step sample variance (5 samples):", "; ".join(out))
