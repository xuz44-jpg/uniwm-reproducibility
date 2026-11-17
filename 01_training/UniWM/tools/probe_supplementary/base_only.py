import os, pickle, numpy as np
from scipy.stats import spearmanr, pearsonr
from _paths import CACHE as sp, DATA as root
names = sorted(d for d in os.listdir(root) if d.startswith("traj_"))
D = {t: pickle.load(open(f"{root}/{t}/traj_data.pkl", "rb")) for t in names}
F = pickle.load(open(f"{sp}/pc_flow_all.pkl", "rb"))
rng = np.random.default_rng(5)
# distributed lag with both base and hip terms
Y, X, tr = [], [], []
for t in names:
    db = np.diff(D[t]["base_yaw"].astype(float)); dh = np.diff(D[t]["hip_yaw"].astype(float)); f = F[t][:, 0]
    for k in range(4, len(db) - 1):
        Y.append(f[k]); X.append([db[k+1]] + [db[k-j] for j in range(5)] + [dh[k+1]] + [dh[k-j] for j in range(5)]); tr.append(t)
Y, X, tr = np.array(Y), np.array(X), np.array(tr)
M = np.c_[X, np.ones(len(Y))]; c = np.linalg.lstsq(M, Y, rcond=None)[0]
uniq = np.unique(tr); idx_of = {u: np.where(tr == u)[0] for u in uniq}; Bt = []
for _ in range(500):
    ix = np.concatenate([idx_of[u] for u in rng.choice(uniq, len(uniq))]); Bt.append(np.linalg.lstsq(M[ix], Y[ix], rcond=None)[0])
lo, hi = np.percentile(np.array(Bt), [2.5, 97.5], axis=0)
nm = ["k+1", "k", "k-1", "k-2", "k-3", "k-4"]
print(f"joint distributed-lag OLS, n={len(Y)} (px per rad of label), traj-clustered 95% CI")
for i in range(6): print(f"  dbase[{nm[i]}] {c[i]:+7.1f} [{lo[i]:+7.1f},{hi[i]:+7.1f}]    dhip[{nm[i]}] {c[6+i]:+7.1f} [{lo[6+i]:+7.1f},{hi[6+i]:+7.1f}]")
print(f"  sum base {c[:6].sum():+.1f}; sum hip {c[6:12].sum():+.1f}")
bb = np.array(Bt)[:, :6]; share = bb[:, 3] / bb.sum(1)
print(f"  share of total base response carried by dbase[k-2]: {c[3]/c[:6].sum():.2f} [{np.percentile(share,2.5):.2f},{np.percentile(share,97.5):.2f}]; by dbase[k]: {c[1]/c[:6].sum():.2f}")
# hip-corrected flow lag scan
for kh in (0, -140):
    res = []
    for Lraw in range(-6, 19, 3):
        L = Lraw / 6; ff, dd = [], []
        for t in names:
            y = D[t]["base_yaw"].astype(float); kk = np.arange(len(y)); f = F[t][:, 0]; dh = np.diff(D[t]["hip_yaw"].astype(float))
            for k in range(4, len(y) - 2):
                ff.append(f[k] - kh * dh[k]); dd.append(np.interp(k + 1 - L, kk, y) - np.interp(k - L, kk, y))
        res.append((Lraw, pearsonr(ff, dd)[0], spearmanr(ff, dd)[0]))
    print(f"base_yaw lag scan, flow corrected by hip at {kh} px/rad: " + "  ".join(f"L{r[0]:+d}:{r[1]:+.3f}/{r[2]:+.3f}" for r in res))
