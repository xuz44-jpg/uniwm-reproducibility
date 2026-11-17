import os, pickle, numpy as np
from scipy.stats import spearmanr, pearsonr
from _paths import CACHE as sp, DATA as root
names = sorted(d for d in os.listdir(root) if d.startswith("traj_"))
D = {t: pickle.load(open(f"{root}/{t}/traj_data.pkl", "rb")) for t in names}
F = pickle.load(open(f"{sp}/pc_flow_all.pkl", "rb"))
rng = np.random.default_rng(1)
# fixed sample: pairs with k in [4, N-3] so every lag in the scan is available on the same pairs
def scan(series_fn, flow_col, lags_raw, kmin=4, kback=1):
    out = []
    for Lraw in lags_raw:
        L = Lraw / 6.0; ff, dd = [], []
        for t in names:
            y = series_fn(D[t]); kk = np.arange(len(y)); f = F[t][:, flow_col]
            for k in range(kmin, len(y) - 1 - kback):
                ff.append(f[k]); dd.append(np.interp(k + 1 - L, kk, y) - np.interp(k - L, kk, y))
        out.append((Lraw, spearmanr(ff, dd)[0], pearsonr(ff, dd)[0], len(ff)))
    return out
yaw = lambda d: d["yaw"].astype(float)
print("horizontal flow vs heading label, same pairs (k>=4, k<=N-3), lag L raw frames: label window [k-L, k+1-L]")
for Lraw, s, p, n in scan(yaw, 0, range(-6, 25, 2)):
    print(f"  L={Lraw:+3d} raw ({Lraw*1000/30:+5.0f} ms)  Spearman {s:+.3f}  Pearson {p:+.3f}  n={n}")
# distributed-lag OLS: flow(k,k+1) ~ sum_j b_j dyaw[k-j], j=-1..4
X, yv, tr = [], [], []
for t in names:
    dy = D[t]["delta"][:, 2].astype(float); f = F[t][:, 0]
    for k in range(4, len(dy) - 1):
        X.append([dy[k + 1]] + [dy[k - j] for j in range(0, 5)]); yv.append(f[k]); tr.append(t)
X, yv, tr = np.array(X), np.array(yv), np.array(tr)
M = np.c_[X, np.ones(len(X))]
c = np.linalg.lstsq(M, yv, rcond=None)[0]
uniq = np.unique(tr); idx_of = {u: np.where(tr == u)[0] for u in uniq}; B = []
for _ in range(500):
    ix = np.concatenate([idx_of[u] for u in rng.choice(uniq, len(uniq))]); B.append(np.linalg.lstsq(M[ix], yv[ix], rcond=None)[0])
lo, hi = np.percentile(np.array(B), [2.5, 97.5], axis=0)
print(f"\ndistributed-lag OLS (n={len(yv)}), px/rad with traj-clustered 95% CI:")
for j, nm in enumerate(["dyaw[k+1]", "dyaw[k]", "dyaw[k-1]", "dyaw[k-2]", "dyaw[k-3]", "dyaw[k-4]"]):
    print(f"  {nm:9s} {c[j]:+7.1f}  [{lo[j]:+7.1f}, {hi[j]:+7.1f}]")
# event study: turn onsets = first sampled frame with ang_vel != 0 after >=3 frames of ang_vel == 0; sign-normalised
print("\nevent study at commanded-turn onsets (ang_vel[k0-3..k0-1]==0, |ang_vel[k0]|>0.1; sign by ang_vel[k0]):")
rel = range(-3, 6); acc = {r: ([], [], []) for r in rel}; nev = 0
for t in names:
    d = D[t]; w = d["ang_vel"].astype(float); dy = d["delta"][:, 2].astype(float); f = F[t][:, 0]
    for k0 in range(3, len(w)):
        if np.all(w[k0-3:k0] == 0) and abs(w[k0]) > 0.1:
            s = np.sign(w[k0]); nev += 1
            for r in rel:
                k = k0 + r
                if 0 <= k < len(dy): acc[r][0].append(s * dy[k]); acc[r][1].append(s * f[k]); acc[r][2].append(s * w[k])
            break
print(f"  events: {nev} (one per trajectory, first onset)")
print("  step r (pair k0+r -> k0+r+1): mean signed ang_vel[k0+r] | mean signed dyaw label[k0+r] | mean signed flow(k0+r,k0+r+1) px | median flow")
for r in rel:
    w_, dl, fl = np.array(acc[r][2]), np.array(acc[r][0]), np.array(acc[r][1])
    print(f"  r={r:+d}: w {w_.mean():+.3f}  dyaw {dl.mean():+.4f}  flow {fl.mean():+6.2f}  median {np.median(fl):+6.2f}  n={len(fl)}")
