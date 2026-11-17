import os, pickle, numpy as np
from scipy.stats import spearmanr, pearsonr
from _paths import CACHE as sp, DATA as root
names = sorted(d for d in os.listdir(root) if d.startswith("traj_"))
D = {t: pickle.load(open(f"{root}/{t}/traj_data.pkl", "rb")) for t in names}
F = pickle.load(open(f"{sp}/pc_flow_all.pkl", "rb"))
rng = np.random.default_rng(2)
LAGS = range(-2, 5)  # j: joint change over [k-j, k+1-j]
def build(flow_col, joint, extra_base=True):
    Y, J, Bz, tr = [], [], [], []
    for t in names:
        d = D[t]; f = F[t][:, flow_col]; dj = np.diff(d["state"][:, joint].astype(float)); db = np.diff(d["base_yaw"].astype(float))
        for k in range(4, len(f) - 2):
            Y.append(f[k]); J.append([dj[k - j] for j in LAGS]); Bz.append([db[k - j] for j in range(-1, 5)]); tr.append(t)
    return np.array(Y), np.array(J), np.array(Bz), np.array(tr)
def report(name, flow_col, joint, sel_fn=None, control_base=True):
    Y, J, Bz, tr = build(flow_col, joint)
    m = np.ones(len(Y), bool) if sel_fn is None else sel_fn(Bz)
    Y, J, Bz, tr = Y[m], J[m], Bz[m], tr[m]
    if control_base:  # residualise flow on base-yaw terms at lags -1..4
        M = np.c_[Bz, np.ones(len(Y))]; Y = Y - M @ np.linalg.lstsq(M, Y, rcond=None)[0]
    r = [pearsonr(Y, J[:, i])[0] for i in range(J.shape[1])]
    rs = [spearmanr(Y, J[:, i])[0] for i in range(J.shape[1])]
    # joint OLS on all joint lags with clustered bootstrap
    M = np.c_[J, np.ones(len(Y))]; c = np.linalg.lstsq(M, Y, rcond=None)[0]
    uniq = np.unique(tr); idx_of = {u: np.where(tr == u)[0] for u in uniq}; Bt = []
    for _ in range(400):
        ix = np.concatenate([idx_of[u] for u in rng.choice(uniq, len(uniq))]); Bt.append(np.linalg.lstsq(M[ix], Y[ix], rcond=None)[0])
    lo, hi = np.percentile(np.array(Bt), [2.5, 97.5], axis=0)
    print(f"{name}: n={len(Y)}")
    for i, j in enumerate(LAGS):
        tag = f"d{joint}[k{-j:+d}]" if j else f"d{joint}[k]"
        print(f"   joint change over [k{-j:+d}, k{1-j:+d}]: Pearson {r[i]:+.3f} Spearman {rs[i]:+.3f} | joint-OLS px/rad {c[i]:+7.1f} [{lo[i]:+7.1f},{hi[i]:+7.1f}]")
report("horizontal flow residual (base yaw lags -1..4 removed) vs HIP_YAW (0)", 0, 0)
report("horizontal flow vs HIP_YAW (0), only pairs with base yaw exactly still over [k-4,k+2]", 0, 0, sel_fn=lambda Bz: np.all(np.abs(Bz) < 1e-7, axis=1), control_base=False)
report("vertical flow vs NECK_PITCH (20)", 1, 20)
report("vertical flow vs HIP_PITCH (2)", 1, 2)
