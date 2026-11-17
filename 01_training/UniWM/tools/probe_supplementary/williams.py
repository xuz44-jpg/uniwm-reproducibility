import os, pickle, numpy as np
from scipy.stats import spearmanr, t as tdist, rankdata, pearsonr
r12, r13, r23, n = 0.844, 0.751, 0.908, 449   # flow~dyaw[k-1], flow~dyaw[k], dyaw[k-1]~dyaw[k] (last from local pkl on same sample)
R = 1 - r12**2 - r13**2 - r23**2 + 2*r12*r13*r23
tw = (r12 - r13) * np.sqrt((n - 1) * (1 + r23)) / np.sqrt(2 * (n - 1) / (n - 3) * R + ((r12 + r13) / 2)**2 * (1 - r23)**3)
print(f"Williams t (published A2 numbers, n={n} treated as independent): t={tw:.2f}, p={2*tdist.sf(abs(tw), n-3):.2g}")
for neff in (60, 120):
    tw2 = (r12 - r13) * np.sqrt((neff - 1) * (1 + r23)) / np.sqrt(2 * (neff - 1) / (neff - 3) * R + ((r12 + r13) / 2)**2 * (1 - r23)**3)
    print(f"  same with n_eff={neff}: t={tw2:.2f}, p={2*tdist.sf(abs(tw2), neff-3):.2g}")
p_k_given_km1 = (r13 - r12*r23) / np.sqrt((1 - r12**2) * (1 - r23**2))
p_km1_given_k = (r12 - r13*r23) / np.sqrt((1 - r13**2) * (1 - r23**2))
print(f"partial rank corr from published numbers: flow~dyaw[k] | dyaw[k-1] = {p_k_given_km1:+.3f}; flow~dyaw[k-1] | dyaw[k] = {p_km1_given_k:+.3f}")
# clustered bootstrap on the local replication of exactly the A2 calibration sample
from _paths import CACHE as sp, DATA as root
names = sorted(d for d in os.listdir(root) if d.startswith("traj_")); F = pickle.load(open(f"{sp}/pc_flow_all.pkl", "rb"))
rows = []
for t in names[:60]:
    dy = pickle.load(open(f"{root}/{t}/traj_data.pkl", "rb"))["delta"][:, 2].astype(float); f = F[t][:, 0]
    for k in range(1, len(dy) - 1, 3): rows.append((t, f[k], dy[k-1], dy[k], dy[k+1], dy[k-2] if k >= 2 else np.nan))
tr = np.array([r[0] for r in rows]); A = np.array([r[1:] for r in rows], float)
rng = np.random.default_rng(7); uniq = np.unique(tr); idx_of = {u: np.where(tr == u)[0] for u in uniq}
def st(ix):
    a = A[ix]; r1 = spearmanr(a[:, 0], a[:, 1])[0]; r0 = spearmanr(a[:, 0], a[:, 2])[0]; r23 = spearmanr(a[:, 1], a[:, 2])[0]
    return r1 - r0, (r0 - r1*r23) / np.sqrt((1 - r1**2) * (1 - r23**2))
full = st(np.arange(len(A))); B = np.array([st(np.concatenate([idx_of[u] for u in rng.choice(uniq, len(uniq))])) for _ in range(2000)])
lo, hi = np.percentile(B, [2.5, 97.5], axis=0)
print(f"local phase-corr replication of the A2 calibration sample (n={len(A)}, 60 trajs): rho(k-1)-rho(k) {full[0]:+.3f} [{lo[0]:+.3f},{hi[0]:+.3f}]; partial rho(k|k-1) {full[1]:+.3f} [{lo[1]:+.3f},{hi[1]:+.3f}]")
m = ~np.isnan(A[:, 4]); print(f"  same sample k>=2: Spearman flow~dyaw[k-2] {spearmanr(A[m,0], A[m,4])[0]:+.3f}, ~dyaw[k-1] {spearmanr(A[m,0], A[m,1])[0]:+.3f}, ~dyaw[k] {spearmanr(A[m,0], A[m,2])[0]:+.3f} (n={m.sum()})")
# how many pairs actually discriminate k-1 vs k: |dyaw[k]-dyaw[k-1]| >= 0.03
d = np.abs(A[:, 2] - A[:, 1]); print(f"  pairs where |dyaw[k]-dyaw[k-1]|>=0.03: {np.sum(d>=0.03)} of {len(A)}")
