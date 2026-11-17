import os, pickle, numpy as np
from scipy.stats import spearmanr, pearsonr
from _paths import CACHE as sp, DATA as root
names = sorted(d for d in os.listdir(root) if d.startswith("traj_"))
D = {t: pickle.load(open(f"{root}/{t}/traj_data.pkl", "rb")) for t in names}
F = pickle.load(open(f"{sp}/pc_flow_all.pkl", "rb"))
# 1. hip sign test with base commanded still over [k-4, k+2]
fl, dh = [], []
for t in names:
    db = np.diff(D[t]["base_yaw"].astype(float)); h = np.diff(D[t]["hip_yaw"].astype(float)); f = F[t][:, 0]
    for k in range(4, len(db) - 2):
        if np.all(np.abs(db[k-4:k+3]) < 1e-7): fl.append(f[k]); dh.append(h[k])
fl, dh = np.array(fl), np.array(dh)
for thr in (0.01, 0.02, 0.04):
    m = np.abs(dh) >= thr
    print(f"base still, |dhip[k]|>={thr}: n={m.sum()} frac sign(flow)==-sign(dhip) {np.mean(np.sign(fl[m]) == -np.sign(dhip) if False else np.sign(fl[m]) == -np.sign(dh[m])):.2f}; median flow/dhip {np.median(fl[m]/dh[m]):+.0f} px/rad")
# 2. base-yaw lag scan restricted to pairs where hip is still over [k-3,k+2]
def scan(sel, lags=(0, 6, 12)):
    out = []
    for Lraw in lags:
        L = Lraw / 6; ff, dd = [], []
        for t in names:
            y = D[t]["base_yaw"].astype(float); kk = np.arange(len(y)); f = F[t][:, 0]; h = np.diff(D[t]["hip_yaw"].astype(float))
            for k in range(4, len(y) - 2):
                if sel(h, k): ff.append(f[k]); dd.append(np.interp(k + 1 - L, kk, y) - np.interp(k - L, kk, y))
        out.append((Lraw, pearsonr(ff, dd)[0], spearmanr(ff, dd)[0], len(ff)))
    return out
for thr in (0.005, 0.003):
    res = scan(lambda h, k: np.all(np.abs(h[k-3:k+2]) < thr))
    print(f"hip still (|dhip|<{thr} over [k-3,k+2]): " + "  ".join(f"L={r[0]:+d}raw Pearson {r[1]:+.3f} Spearman {r[2]:+.3f}" for r in res) + f"  n={res[0][3]}")
# 3. OLS of flow on hip at lag 0 with + and - sign combined heading: which fused heading fits flow better at best base lag
X, Y = [], []
for t in names:
    y = D[t]["base_yaw"].astype(float); kk = np.arange(len(y)); f = F[t][:, 0]; h = D[t]["hip_yaw"].astype(float)
    for k in range(4, len(y) - 2):
        X.append((np.interp(k + 1 - 2, kk, y) - np.interp(k - 2, kk, y), h[k+1] - h[k], y[k+1] - y[k])); Y.append(f[k])
X, Y = np.array(X), np.array(Y)
M = np.c_[X[:, 0], X[:, 1], np.ones(len(Y))]; c = np.linalg.lstsq(M, Y, rcond=None)[0]
print(f"OLS flow(k,k+1) ~ dbase[k-2] + dhip[k]: {c[0]:+.1f} px/rad(base label), {c[1]:+.1f} px/rad(hip); ratio hip/base {c[1]/c[0]:+.2f}")
print(f"Pearson flow vs dbase[k] {pearsonr(Y, X[:,2]-X[:,1]*0 - 0)[0]:+.3f} (this is base+hip label = yaw diff)")
