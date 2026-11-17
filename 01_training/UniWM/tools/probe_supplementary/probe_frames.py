import os, pickle, json, numpy as np
from scipy.stats import spearmanr
from _paths import CACHE as sp, DATA as root, FIGS
F = pickle.load(open(f"{sp}/pc_flow_all.pkl", "rb"))
rows = json.load(open(f"{FIGS}/action_probe/probe_rows.json"))
print(len(rows), rows[0].keys())
T = [r for r in rows if r["kind"] == "turn"]
print("turning probe frames:", len(T))
print(" traj        k  dyaw[k-1] dyaw[k] dyaw[k+1] dyaw[k+2] | real_hx(Farneback,VQcur->raw next) | pc flow (k,k+1) (k+1,k+2) (k+2,k+3)")
agg = {0: [], 1: [], 2: []}; lab = {-1: [], 0: [], 1: [], 2: []}
for r in T:
    d = pickle.load(open(f"{root}/{r['traj']}/traj_data.pkl", "rb")); dy = d["delta"][:, 2]; k = r["k"]; f = F[r["traj"]][:, 0]
    s = np.sign(dy[k])
    g = lambda i: dy[i] if 0 <= i < len(dy) else np.nan
    h = lambda i: f[i] if 0 <= i < len(f) else np.nan
    print(f" {r['traj']} {k:3d}  {g(k-1):+.3f}   {g(k):+.3f}  {g(k+1):+.3f}   {g(k+2):+.3f}   | {r['real_hx']:+6.2f} | {h(k):+6.2f} {h(k+1):+6.2f} {h(k+2):+6.2f}")
    for j in (0, 1, 2):
        if not np.isnan(h(k + j)): agg[j].append(s * h(k + j))
    for j in (-1, 0, 1, 2):
        if not np.isnan(g(k + j)): lab[j].append(s * g(k + j))
for j in (-1, 0, 1, 2): print(f"mean sign-normalised label dyaw[k{j:+d}] {np.mean(lab[j]):+.4f}")
for j in (0, 1, 2): print(f"pair (k{j:+d},k{j+1:+d}): mean sign-normalised pc flow {np.mean(agg[j]):+.2f} px, median {np.median(agg[j]):+.2f}, frac with label sign {np.mean(np.array(agg[j])>0):.2f} (n={len(agg[j])})")
