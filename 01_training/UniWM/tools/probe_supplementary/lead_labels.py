import os, pickle, numpy as np, random
from scipy.stats import spearmanr, pearsonr
from _paths import DATA as root
names = sorted(d for d in os.listdir(root) if d.startswith("traj_"))
D = {t: pickle.load(open(f"{root}/{t}/traj_data.pkl", "rb")) for t in names}
print("n trajs", len(names))
# raw_frame_idx spacing
sp = np.concatenate([np.diff(D[t]["raw_frame_idx"]) for t in names]); print("raw_frame_idx spacing unique:", np.unique(sp))
L = np.array([len(D[t]["yaw"]) for t in names]); print("len min/med/max", L.min(), np.median(L), L.max())
# check yaw = base_yaw + hip - hip0
err = max(np.abs(D[t]["yaw"] - (D[t]["base_yaw"] + D[t]["hip_yaw"] - D[t]["hip_yaw"][0])).max() for t in names); print("max |yaw - (base + hip - hip0)|", err)
err = max(np.abs(np.diff(D[t]["yaw"]) - D[t]["delta"][:, 2]).max() for t in names); print("max |diff(yaw) - delta_yaw|", err)
# 1. lag-1 autocorrelation of dyaw (within trajectory)
def lagpairs(key_fn, lag, trajs=names):
    a, b, tr = [], [], []
    for t in trajs:
        x = key_fn(D[t])
        for k in range(len(x) - lag):
            a.append(x[k]); b.append(x[k + lag]); tr.append(t)
    return np.array(a), np.array(b), np.array(tr)
dy = lambda d: d["delta"][:, 2].astype(float)
for lag in (1, 2, 3):
    a, b, _ = lagpairs(dy, lag)
    print(f"dyaw lag-{lag}: Pearson {pearsonr(a,b)[0]:+.3f} Spearman {spearmanr(a,b)[0]:+.3f} n={len(a)}")
# demeaned per trajectory (within-trajectory autocorrelation)
a, b = [], []
for t in names:
    x = dy(D[t]); x = x - x.mean(); a += list(x[:-1]); b += list(x[1:])
print(f"dyaw lag-1 within-traj demeaned Pearson {pearsonr(a,b)[0]:+.3f}")
# conditional on turning
a, b, _ = lagpairs(dy, 1); sel = np.abs(b) >= 0.05
print(f"given |dyaw[k]|>=0.05 (n={sel.sum()}): sign(dyaw[k-1])==sign(dyaw[k]) {np.mean(np.sign(a[sel])==np.sign(b[sel])):.3f}; |dyaw[k-1]|>=0.05 {np.mean(np.abs(a[sel])>=0.05):.3f}")
# same as used in A2: for pairs k in 1..len-2, Spearman(dyaw[k-1], dyaw[k]) on calibration sample
random.seed(42); sh = names[:]; random.shuffle(sh); n = len(sh); p1 = int(n * 0.9); p2 = p1 + int(n * 0.05); test = sh[p2:]
for label, trajs, stride in (("calib 0-59 stride3", names[:60], 3), ("test stride2", test, 2)):
    rows = []
    for t in trajs:
        dl = D[t]["delta"]
        for k in range(1, len(dl) - 1, stride):
            rows.append((dl[k-1][2], dl[k][2], dl[k+1][2]))
    r = np.array(rows, float)
    print(f"{label}: n={len(r)} Spearman(dyaw[k-1],dyaw[k]) {spearmanr(r[:,0],r[:,1])[0]:+.3f}; (dyaw[k],dyaw[k+1]) {spearmanr(r[:,1],r[:,2])[0]:+.3f}; (k-1,k+1) {spearmanr(r[:,0],r[:,2])[0]:+.3f}")
print("test trajs", len(test))
