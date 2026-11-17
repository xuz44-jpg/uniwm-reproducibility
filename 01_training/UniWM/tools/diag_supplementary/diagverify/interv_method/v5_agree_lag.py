from ivcommon import *
from ivcommon import _bin
def wrap(a): return (a + np.pi) % (2 * np.pi) - np.pi
for run, key in (("base", "stop_label_released"), ("stopft", "stop_label_dist")):
    R = load(run); rows = [(r["traj"], r["n"], st) for r in R for st in r["steps"]]
    print(f"\n== {run}")
    # I1: exact agreement of greedy outputs, and mean |diff| of components
    for other in ("vq", "gen"):
        same = np.mean([st["real"][0] == st[other][0] for _, _, st in rows])
        P = [(st["real"][0], st[other][0]) for _, _, st in rows if st["real"][0] not in (None, "stop") and st[other][0] not in (None, "stop")]
        dif = np.abs(np.array([a for a, b in P]) - np.array([b for a, b in P]))
        stopflip = sum((st["real"][0] == "stop") != (st[other][0] == "stop") for _, _, st in rows)
        print(f"  greedy real vs {other}: identical output {same:.3f}; Stop/non-Stop disagreements {stopflip}; mean |diff| dx {dif[:,0].mean():.4f} dy {dif[:,1].mean():.4f} dyaw {dif[:,2].mean():.4f} (n={len(P)})")
    # s>=1 only for vq (s=0 gen==vq by construction)
    # dy predictions
    for name in ("real", "vq", "gen"):
        P = [st[name][0] for _, _, st in rows if st[name][0] not in (None, "stop") and not st[key]]
        L = [st["recorded"] for _, _, st in rows if st[name][0] not in (None, "stop") and not st[key]]
        print(f"  {name}: mean pred |dy| {np.mean([abs(p[1]) for p in P]):.3f} (label {np.mean([abs(_bin(l[1])) for l in L]):.3f}); mean pred dx {np.mean([p[0] for p in P]):.3f} (label {np.mean([_bin(l[0]) for l in L]):.3f}); mean pred |dyaw| {np.mean([abs(p[2]) for p in P]):.3f} (label {np.mean([abs(_bin(l[2])) for l in L]):.3f})")
    # (c) viewpoint divergence: lagged correlation of predicted dyaw with recorded dyaw[s+j]
    for name in ("real", "gen"):
        out = []
        for j in (-3, -2, -1, 0, 1, 2):
            xs, ys = [], []
            for r in R:
                dl = np.asarray(pkl(r["traj"])["delta"])[: r["n"] - 1]
                for st in r["steps"]:
                    s = st["s"]; p = st[name][0]
                    if p in (None, "stop") or st[key] or not (0 <= s + j < len(dl)): continue
                    xs.append(p[2]); ys.append(dl[s + j][2])
            out.append(f"j={j:+d}:{np.corrcoef(xs, ys)[0,1]:.3f}")
        print(f"  corr(pred dyaw[{name}] at s, recorded dyaw[s+j]): " + " ".join(out))
    # predicted dyaw vs remaining heading to goal (true state) vs heading-to-goal from the START (never-rotated imagination)
    for name in ("real", "gen"):
        X, A, B = [], [], []
        for r in R:
            yaw = np.asarray(pkl(r["traj"])["yaw"], float).ravel()[: r["n"]]
            for st in r["steps"]:
                p = st[name][0]; s = st["s"]
                if p in (None, "stop") or st[key]: continue
                X.append(p[2]); A.append(wrap(yaw[-1] - yaw[s])); B.append(wrap(yaw[-1] - yaw[0]))
        X, A, B = map(np.asarray, (X, A, B))
        def pc(x, y, z):
            rx = x - np.polyval(np.polyfit(z, x, 1), z); ry = y - np.polyval(np.polyfit(z, y, 1), z); return np.corrcoef(rx, ry)[0, 1]
        print(f"  {name}: corr(pred dyaw, remaining heading yaw[n-1]-yaw[s]) {np.corrcoef(X,A)[0,1]:.3f}; corr(pred dyaw, yaw[n-1]-yaw[0]) {np.corrcoef(X,B)[0,1]:.3f}; "
              f"partial(remaining | start-gap) {pc(X,A,B):.3f}; partial(start-gap | remaining) {pc(X,B,A):.3f}")
