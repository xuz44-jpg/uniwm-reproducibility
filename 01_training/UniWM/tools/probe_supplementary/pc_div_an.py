import os, pickle, numpy as np
from scipy.stats import spearmanr, pearsonr
from _paths import CACHE as sp, DATA as root
names = sorted(d for d in os.listdir(root) if d.startswith("traj_"))
D = {t: pickle.load(open(f"{root}/{t}/traj_data.pkl", "rb")) for t in names}
V = pickle.load(open(f"{sp}/pc_div_all.pkl", "rb"))
rng = np.random.default_rng(3)
def ds(d):  # signed forward displacement per step
    return np.linalg.norm(d["delta"][:, :2].astype(float), axis=1) * np.sign(d["lin_vel"][:-1] + d["lin_vel"][1:])
Y, X, tr = [], [], []
for t in names:
    s = ds(D[t]); div = V[t][:, 1] - V[t][:, 0]
    for k in range(4, len(s) - 1):
        Y.append(div[k]); X.append([s[k + 1]] + [s[k - j] for j in range(5)]); tr.append(t)
Y, X, tr = np.array(Y), np.array(X), np.array(tr)
print(f"divergence proxy (right-third dx minus left-third dx) vs forward displacement label, n={len(Y)}")
for i, nm in enumerate(["ds[k+1]", "ds[k]", "ds[k-1]", "ds[k-2]", "ds[k-3]", "ds[k-4]"]):
    print(f"  {nm}: Spearman {spearmanr(Y, X[:, i])[0]:+.3f} Pearson {pearsonr(Y, X[:, i])[0]:+.3f}")
M = np.c_[X, np.ones(len(Y))]; c = np.linalg.lstsq(M, Y, rcond=None)[0]
uniq = np.unique(tr); idx_of = {u: np.where(tr == u)[0] for u in uniq}; Bt = []
for _ in range(400):
    ix = np.concatenate([idx_of[u] for u in rng.choice(uniq, len(uniq))]); Bt.append(np.linalg.lstsq(M[ix], Y[ix], rcond=None)[0])
lo, hi = np.percentile(np.array(Bt), [2.5, 97.5], axis=0)
print("  distributed-lag OLS px/m:", ", ".join(f"{nm} {c[i]:+.1f} [{lo[i]:+.1f},{hi[i]:+.1f}]" for i, nm in enumerate(["ds[k+1]", "ds[k]", "ds[k-1]", "ds[k-2]", "ds[k-3]", "ds[k-4]"])))
# start-of-trajectory event: most trajectories start from near-standstill (lin_vel[0] small)
print("\nevent study at trajectory start where |lin_vel[0]|<0.1 and |lin_vel[1]|>0.3 (sign by lin_vel[1]):")
acc = {r: ([], [], []) for r in range(0, 6)}; nev = 0
for t in names:
    d = D[t]; v = d["lin_vel"].astype(float)
    if abs(v[0]) < 0.1 and abs(v[1]) > 0.3:
        nev += 1; sg = np.sign(v[1]); s = ds(d); div = V[t][:, 1] - V[t][:, 0]
        for r in range(0, 6):
            if r < len(s): acc[r][0].append(v[r] * sg); acc[r][1].append(s[r] * sg); acc[r][2].append(div[r] * sg)
print(f"  events {nev}")
for r in range(0, 6):
    a = [np.array(x) for x in acc[r]]
    print(f"  pair ({r},{r+1}): lin_vel[{r}] {a[0].mean():+.3f}  label ds {a[1].mean():+.4f} m  divergence proxy mean {a[2].mean():+.3f} median {np.median(a[2]):+.3f} px")
