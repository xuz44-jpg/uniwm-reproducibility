import os, pickle, numpy as np
from _paths import CACHE as sp, DATA as root
names = sorted(d for d in os.listdir(root) if d.startswith("traj_"))
D = {t: pickle.load(open(f"{root}/{t}/traj_data.pkl", "rb")) for t in names}
F = pickle.load(open(f"{sp}/pc_flow_all.pkl", "rb"))
rel = range(-3, 7)
def study(filter_fn, title):
    acc = {r: [] for r in rel}; nev = 0
    for t in names:
        d = D[t]; w = d["ang_vel"].astype(float); db = np.diff(d["base_yaw"].astype(float)); dh = np.diff(d["hip_yaw"].astype(float)); f = F[t][:, 0]
        for k0 in range(3, len(w)):
            if np.all(w[k0-3:k0] == 0) and abs(w[k0]) > 0.1:
                if not filter_fn(dh, k0): break
                s = np.sign(w[k0]); nev += 1
                for r in rel:
                    k = k0 + r
                    if 0 <= k < len(db): acc[r].append((s * db[k], s * dh[k], s * f[k]))
                break
    print(f"{title}: events {nev}")
    print("   r  base label  dhip   flow(px)  flow median   cumulative: base label  flow")
    cb = cf = 0
    for r in rel:
        a = np.array(acc[r]); cb += a[:, 0].mean(); cf += a[:, 2].mean()
        print(f"  {r:+d}  {a[:,0].mean():+.4f}   {a[:,1].mean():+.4f}  {a[:,2].mean():+6.2f}   {np.median(a[:,2]):+6.2f}        {cb:+.3f}        {cf:+6.2f}   n={len(a)}")
study(lambda dh, k0: True, "all first onsets")
study(lambda dh, k0: np.all(np.abs(dh[max(0,k0-3):k0+4]) < 0.005), "onsets with hip still (|dhip|<0.005 over [k0-3, k0+3])")
