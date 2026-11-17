import os, pickle, numpy as np
from scipy.stats import spearmanr, pearsonr
from _paths import CACHE as sp, DATA as root
names = sorted(d for d in os.listdir(root) if d.startswith("traj_"))
D = {t: pickle.load(open(f"{root}/{t}/traj_data.pkl", "rb")) for t in names}
F = pickle.load(open(f"{sp}/pc_flow_all.pkl", "rb"))
def scan(series, col, sel=None, lags=range(-6, 19, 2)):
    res = []
    for Lraw in lags:
        L = Lraw / 6; ff, dd = [], []
        for t in names:
            y = series(D[t]); kk = np.arange(len(y)); f = F[t][:, col]
            db = np.diff(D[t]["base_yaw"].astype(float))
            for k in range(4, len(y) - 2):
                if sel is not None and not sel(db, k): continue
                ff.append(f[k]); dd.append(np.interp(k + 1 - L, kk, y) - np.interp(k - L, kk, y))
        res.append((Lraw, pearsonr(ff, dd)[0], spearmanr(ff, dd)[0], len(ff)))
    return res
still = lambda db, k: np.all(np.abs(db[max(0, k - 4):k + 3]) < 1e-7)
for title, series, col, sel in (("NECK_PITCH(20) vs vertical flow, all pairs", lambda d: d["state"][:, 20].astype(float), 1, None),
                                ("HIP_YAW(0) vs horizontal flow, base still", lambda d: d["state"][:, 0].astype(float), 0, still),
                                ("heading label (yaw) vs horizontal flow, all pairs", lambda d: d["yaw"].astype(float), 0, None),
                                ("base_yaw vs horizontal flow, all pairs", lambda d: d["base_yaw"].astype(float), 0, None)):
    res = scan(series, col, sel)
    b = max(res, key=lambda r: abs(r[1]))
    print(f"{title}: n={res[0][3]}; |Pearson| peak at L={b[0]:+d} raw ({b[0]*1000/30:+.0f} ms)")
    print("   " + "  ".join(f"L{r[0]:+d}:{r[1]:+.3f}" for r in res))
