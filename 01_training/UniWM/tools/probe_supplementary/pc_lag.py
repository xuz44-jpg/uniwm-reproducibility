import os, pickle, random, numpy as np
from scipy.stats import spearmanr, pearsonr, rankdata
from _paths import CACHE as sp, DATA as root
names = sorted(d for d in os.listdir(root) if d.startswith("traj_"))
D = {t: pickle.load(open(f"{root}/{t}/traj_data.pkl", "rb")) for t in names}
F = pickle.load(open(f"{sp}/pc_flow_all.pkl", "rb"))
random.seed(42); sh = names[:]; random.shuffle(sh); n = len(sh); p1 = int(n * 0.9); p2 = p1 + int(n * 0.05); test = sh[p2:]
rng = np.random.default_rng(0)
def rows(trajs, stride=1, kmin=2, kmax_off=2):
    R = []
    for t in trajs:
        d = D[t]; dy = d["delta"][:, 2].astype(float); db = np.diff(d["base_yaw"].astype(float)); dh = np.diff(d["hip_yaw"].astype(float)); f = F[t][:, 0]
        for k in range(kmin, len(dy) - kmax_off, stride):
            R.append((t, k, f[k], dy[k-2], dy[k-1], dy[k], dy[k+1], db[k-1], db[k], dh[k-1], dh[k], dh[k+1] if k+1 < len(dh) else np.nan, db[k+1]))
    return R
def arr(R): return np.array([r[0] for r in R]), np.array([r[2:] for r in R], float)
# 1. replicate A2 samples (k from 1, same strides)
for label, trajs, stride in (("calib trajs 0-59 stride 3", names[:60], 3), ("test trajs stride 2", test, 2)):
    R = []
    for t in trajs:
        dy = D[t]["delta"][:, 2].astype(float); f = F[t][:, 0]
        for k in range(1, len(dy) - 1, stride): R.append((f[k], dy[k-1], dy[k], dy[k+1]))
    R = np.array(R)
    s = [spearmanr(R[:, 0], R[:, j])[0] for j in (1, 2, 3)]
    print(f"{label}: n={len(R)} Spearman(pcflow, dyaw[k-1]/[k]/[k+1]) = {s[0]:+.3f} / {s[1]:+.3f} / {s[2]:+.3f}")
# 2. all pairs pooled
tr, A = arr(rows(names))
f = A[:, 0]; lagcols = {"dyaw[k-2]": 1, "dyaw[k-1]": 2, "dyaw[k]": 3, "dyaw[k+1]": 4}
print(f"\nall trajectories, all k in 2..N-3: n={len(f)} pairs from {len(np.unique(tr))} trajs")
for nm, j in lagcols.items():
    sel = np.abs(A[:, j]) >= 0.05
    print(f"  {nm:9s}: Spearman {spearmanr(f, A[:, j])[0]:+.3f} Pearson {pearsonr(f, A[:, j])[0]:+.3f}; sign agr on |d|>=0.05 {np.mean(np.sign(f[sel]) == np.sign(A[sel, j])):.3f} (n={sel.sum()})")
# clustered bootstrap for difference and for joint regression (rank-based and raw OLS)
uniq = np.unique(tr); idx_of = {u: np.where(tr == u)[0] for u in uniq}
def stats(ix):
    ff, X = f[ix], A[ix]
    r = [spearmanr(ff, X[:, j])[0] for j in (2, 3, 4)]
    # joint OLS of flow on dyaw[k-1], dyaw[k], dyaw[k+1]
    M = np.c_[X[:, 2], X[:, 3], X[:, 4], np.ones(len(ix))]
    c = np.linalg.lstsq(M, ff, rcond=None)[0]
    # partial spearman: rank-residualise
    rf, r1, r0 = rankdata(ff), rankdata(X[:, 2]), rankdata(X[:, 3])
    def resid(y, x): b = np.polyfit(x, y, 1); return y - np.polyval(b, x)
    pc_k_given_km1 = pearsonr(resid(rf, r1), resid(r0, r1))[0]
    pc_km1_given_k = pearsonr(resid(rf, r0), resid(r1, r0))[0]
    return np.array([r[0] - r[1], c[0], c[1], c[2], pc_km1_given_k, pc_k_given_km1])
full = stats(np.arange(len(f)))
B = []
for _ in range(1000):
    pick = rng.choice(uniq, len(uniq)); ix = np.concatenate([idx_of[u] for u in pick]); B.append(stats(ix))
B = np.array(B); lo, hi = np.percentile(B, [2.5, 97.5], axis=0)
labels = ["rho(k-1)-rho(k)", "OLS px/rad dyaw[k-1]", "OLS px/rad dyaw[k]", "OLS px/rad dyaw[k+1]", "partial rho(k-1 | k)", "partial rho(k | k-1)"]
for nm, v, l, h in zip(labels, full, lo, hi): print(f"  {nm:22s} {v:+8.3f}  95% traj-clustered CI [{l:+.3f}, {h:+.3f}]")
# 3. fractional lag scan on the 5 Hz yaw series (linear interpolation of cumulative yaw)
print("\nfractional lag scan: flow(k,k+1) vs yaw(k+1-L)-yaw(k-L), L in raw frames (1 step = 6 raw = 200 ms)")
res = []
for Lraw in range(-6, 13):
    L = Lraw / 6.0; ff, dd = [], []
    for t in names:
        y = D[t]["yaw"].astype(float); kk = np.arange(len(y))
        interp = lambda q: np.interp(q, kk, y, left=np.nan, right=np.nan)
        for k in range(2, len(y) - 3):
            a, b = interp(k - L), interp(k + 1 - L)
            if np.isfinite(a) and np.isfinite(b): ff.append(F[t][k, 0]); dd.append(b - a)
    res.append((Lraw, spearmanr(ff, dd)[0], pearsonr(ff, dd)[0], len(ff)))
for Lraw, s, p, n in res: print(f"  lag {Lraw:+3d} raw ({Lraw*33.3:+6.0f} ms): Spearman {s:+.4f} Pearson {p:+.4f} n={n}")
best = max(res, key=lambda r: r[1]); print("  best Spearman at", best[0], "raw frames")
