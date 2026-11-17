import os, pickle, numpy as np
from scipy.stats import spearmanr, pearsonr
from _paths import DATA as root
names = sorted(d for d in os.listdir(root) if d.startswith("traj_"))
D = {t: pickle.load(open(f"{root}/{t}/traj_data.pkl", "rb")) for t in names}
# window test: dbase[k] = base_yaw[k+1]-base_yaw[k] integrates w over raw idx[k]..idx[k+1]-1 (forward Euler)
X, y, rows = [], [], []
for t in names:
    d = D[t]; b = d["base_yaw"].astype(float); w = d["ang_vel"].astype(float); db = np.diff(b)
    for k in range(1, len(db) - 1):
        X.append([w[k-1], w[k], w[k+1], w[k+2]]); y.append(db[k])
X, y = np.array(X), np.array(y)
coef, *_ = np.linalg.lstsq(np.c_[X, np.ones(len(X))], y, rcond=None)
print("OLS dbase[k] ~ w[k-1], w[k], w[k+1], w[k+2] (x0.2s):", (coef[:4] / 0.2).round(3), "n", len(y))
for j, nm in enumerate(["w[k-1]", "w[k]", "w[k+1]", "w[k+2]"]):
    print(f"  Pearson(dbase[k], 0.2*{nm}) {pearsonr(X[:,j], y)[0]:+.4f}")
# exact test: where w[k]==w[k+1] (piecewise-constant command held), dbase[k] should equal 0.2*w[k] if window=[k,k+1)
held = np.isclose(X[:,1], X[:,2], atol=1e-6) & (np.abs(X[:,1]) > 0.02)
print(f"held w[k]==w[k+1]!=0: n={held.sum()} median |dbase - 0.2 w[k]| {np.median(np.abs(y[held]-0.2*X[held,1])):.5f}  vs median |dbase - 0.2 w[k-1]| {np.median(np.abs(y[held]-0.2*X[held,0])):.5f}")
held2 = np.isclose(X[:,0], X[:,1], atol=1e-6) & (np.abs(X[:,1]) > 0.02) & ~np.isclose(X[:,1], X[:,2], atol=1e-6)
print(f"w[k-1]==w[k]!=w[k+1]: n={held2.sum()} median |dbase[k]-0.2w[k]| {np.median(np.abs(y[held2]-0.2*X[held2,1])):.5f}")
z = (np.abs(X[:,1]) < 1e-9) & (np.abs(X[:,2]) < 1e-9)
print(f"w[k]==w[k+1]==0: n={z.sum()} frac dbase[k]==0 {np.mean(np.abs(y[z])<1e-7):.3f}")
# fraction of exactly zero ang_vel and discreteness
w_all = np.concatenate([D[t]["ang_vel"] for t in names]); v_all = np.concatenate([D[t]["lin_vel"] for t in names])
print(f"ang_vel exactly 0: {np.mean(w_all==0):.3f}; unique values {len(np.unique(w_all))} of {len(w_all)}; range [{w_all.min():.3f},{w_all.max():.3f}]")
print(f"lin_vel exactly 0: {np.mean(v_all==0):.3f}; range [{v_all.min():.3f},{v_all.max():.3f}]")
# run lengths of identical consecutive ang_vel at 5 Hz samples
same = np.concatenate([np.isclose(np.diff(D[t]["ang_vel"]), 0, atol=0) for t in names])
print(f"consecutive 5Hz samples with identical ang_vel: {same.mean():.3f}")
# hip yaw vs base yaw
dh_all, db_all = [], []
for t in names:
    d = D[t]; dh_all.append(np.diff(d["hip_yaw"].astype(float))); db_all.append(np.diff(d["base_yaw"].astype(float)))
dh, db = np.concatenate(dh_all), np.concatenate(db_all)
print(f"std dhip {dh.std():.4f} std dbase {db.std():.4f}; mean|dhip| {np.abs(dh).mean():.4f} mean|dbase| {np.abs(db).mean():.4f}; share of var(dyaw) : var(dbase)={db.var():.5f} var(dhip)={dh.var():.5f} cov={np.cov(db,dh)[0,1]:.5f}")
hy = np.concatenate([D[t]["hip_yaw"] - D[t]["hip_yaw"][0] for t in names]); print(f"|hip_yaw - hip_yaw[0]| p50 {np.median(np.abs(hy)):.4f} p99 {np.percentile(np.abs(hy),99):.4f}")
def xcorr(A, B, lag):  # corr(A[k], B[k+lag]) within trajectories
    a, b = [], []
    for x, z in zip(A, B):
        for k in range(len(x)):
            if 0 <= k + lag < len(z): a.append(x[k]); b.append(z[k + lag])
    return pearsonr(a, b)[0], spearmanr(a, b)[0], len(a)
for lag in (-3, -2, -1, 0, 1, 2, 3):
    p, s, n = xcorr(dh_all, db_all, lag)
    print(f"  corr(dhip[k], dbase[k{lag:+d}]) Pearson {p:+.3f} Spearman {s:+.3f} n={n}")
# hip angle vs ang_vel (does torso yaw anticipate/react?)
H = [D[t]["hip_yaw"].astype(float) - D[t]["hip_yaw"][0] for t in names]; W = [D[t]["ang_vel"].astype(float) for t in names]
for lag in (-2, -1, 0, 1, 2):
    p, s, n = xcorr(W, H, lag)
    print(f"  corr(ang_vel[k], hip_yaw[k{lag:+d}]-hip0) Pearson {p:+.3f}")
