from ivcommon import *
from ivcommon import _bin
for run, key in (("base", "stop_label_released"), ("stopft", "stop_label_dist")):
    R = load(run); rows = [(r["traj"], r["n"], st) for r in R for st in r["steps"]]
    M = [(t, n, st) for t, n, st in rows if not st[key]]
    print(f"\n== {run}: move-labelled steps n={len(M)}")
    lab = np.array([st["recorded"] for _, _, st in M])
    print(f"  recorded labels: |dx| mean {np.abs(lab[:,0]).mean():.3f} p90 {np.percentile(np.abs(lab[:,0]),90):.3f}; dy nonzero {np.mean(lab[:,1]!=0):.3f}; "
          f"|dyaw| mean {np.abs(lab[:,2]).mean():.3f} p90 {np.percentile(np.abs(lab[:,2]),90):.3f}; |dyaw|>0.2: {np.mean(np.abs(lab[:,2])>0.2):.3f}")
    # error anatomy
    for name in ("real", "vq", "gen"):
        cat = {"ok":0, "stop":0, "malformed":0, "dx":0, "dy":0, "dyaw":0}
        for _, _, st in M:
            p = st[name][0]; r = st["recorded"]
            if p == "stop": cat["stop"] += 1; continue
            if p is None: cat["malformed"] += 1; continue
            bad = [c for c, pp, rr, tt in zip(("dx","dy","dyaw"), p, r, TOL) if abs(pp - _bin(rr)) > tt]
            if not bad: cat["ok"] += 1
            for c in bad: cat[c] += 1
        print(f"  {name}: " + ", ".join(f"{k} {v}" for k, v in cat.items()))
    # trivial constant predictors under the same tolerance
    med = [float(np.median([_bin(x) for x in lab[:,0]])), 0.0, 0.0]
    for c in ([0.0,0.0,0.0], [0.10,0.0,0.0], [round(med[0],2),0.0,0.0]):
        a = np.mean([acc(c, st["recorded"], False) for _, _, st in M])
        print(f"  constant prediction {c}: accuracy on move-labelled steps {a:.3f}")
    # stricter metrics: |dyaw err| and |dx err|
    for name in ("real", "vq", "gen"):
        P = [(p, st["recorded"]) for _, _, st in M for p in [st[name][0]] if p not in (None, "stop")]
        ey = np.array([abs(p[2]-_bin(r[2])) for p, r in P]); ex = np.array([abs(p[0]-_bin(r[0])) for p, r in P])
        print(f"  {name}: n={len(P)} mean|dx err| {ex.mean():.3f}  mean|dyaw err| {ey.mean():.3f}  share dyaw within 0.05 {np.mean(ey<=0.05):.3f}  within 0.02 {np.mean(ey<=0.02):.3f}")
    # tighter tolerance (0.05,0.05,0.05) paired loss
    def acc_t(p, r, tol):
        return p not in (None, "stop") and all(abs(pp - _bin(rr)) <= tt for pp, rr, tt in zip(p, r, tol))
    for tol in ((0.05,0.05,0.05),(0.03,0.03,0.03)):
        d = [acc_t(st["real"][0], st["recorded"], tol) - acc_t(st["gen"][0], st["recorded"], tol) for _, _, st in M]
        ar = np.mean([acc_t(st["real"][0], st["recorded"], tol) for _,_,st in M]); ag = np.mean([acc_t(st["gen"][0], st["recorded"], tol) for _,_,st in M])
        dv = [acc_t(st["real"][0], st["recorded"], tol) - acc_t(st["vq"][0], st["recorded"], tol) for _, _, st in M]
        print(f"  tol {tol}: real {ar:.3f} gen {ag:.3f} paired real-gen {f3(cluster_ci(d,[t for t,_,_ in M]))}; real-vq {f3(cluster_ci(dv,[t for t,_,_ in M]))}")
