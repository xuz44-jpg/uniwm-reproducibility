"""Re-examine the optical-flow calibration: is 58 px/rad representative, or driven by a few pairs?"""
import os, pickle, sys
import numpy as np, cv2
from PIL import Image
from _paths import TOOLS, DATA
sys.path.insert(0, TOOLS)
from action_probe_analysis import flow_proxy, spearman

root = DATA
def collect(trajs, stride):
    hx, dy, dx = [], [], []
    for t in trajs:
        d = pickle.load(open(os.path.join(root, t, "traj_data.pkl"), "rb")); delta = np.asarray(d["delta"])
        for k in range(0, len(delta), stride):
            a = Image.open(f"{root}/{t}/{k}.jpg").convert("RGB"); b = Image.open(f"{root}/{t}/{k+1}.jpg").convert("RGB")
            h, _ = flow_proxy(a, b); hx.append(h); dy.append(delta[k][2]); dx.append(delta[k][0])
    return np.array(hx), np.array(dy), np.array(dx)

names = sorted(d for d in os.listdir(root) if d.startswith("traj_"))
hx, dy, dx = collect(names[:60], 3)
print(f"calibration sample: {len(hx)} pairs")
print(f"  |dyaw| distribution: p50 {np.percentile(np.abs(dy),50):.3f} p90 {np.percentile(np.abs(dy),90):.3f} p99 {np.percentile(np.abs(dy),99):.3f} max {np.abs(dy).max():.3f}")
print(f"  |flow| distribution: p50 {np.percentile(np.abs(hx),50):.2f} p90 {np.percentile(np.abs(hx),90):.2f} max {np.abs(hx).max():.2f}")
print(f"  Spearman {spearman(dy, hx):.3f}; OLS slope {np.polyfit(dy, hx, 1)[0]:.1f} px/rad")
# robustness: slope without the largest-|dyaw| 5%, and median-of-ratios
keep = np.abs(dy) <= np.percentile(np.abs(dy), 95)
print(f"  slope without the top 5% of |dyaw|: {np.polyfit(dy[keep], hx[keep], 1)[0]:.1f} px/rad")
sel = np.abs(dy) >= 0.05
print(f"  turning pairs (|dyaw|>=0.05): n={sel.sum()}, sign agreement {np.mean(np.sign(hx[sel])==np.sign(dy[sel])):.2f}, median |flow| {np.median(np.abs(hx[sel])):.2f}, median |flow|/|dyaw| {np.median(np.abs(hx[sel])/np.abs(dy[sel])):.1f} px/rad")
for lo, hi in ((0.05,0.10),(0.10,0.20),(0.20,0.40),(0.40,5.0)):
    m = (np.abs(dy)>=lo)&(np.abs(dy)<hi)
    if m.sum(): print(f"    |dyaw| in [{lo},{hi}): n={m.sum():3d} median |flow| {np.median(np.abs(hx[m])):6.2f} sign agr {np.mean(np.sign(hx[m])==np.sign(dy[m])):.2f}")
# same measurement on the 26 test trajectories
random_test = names  # recompute the split
import random
random.seed(42); sh = names[:]; random.shuffle(sh)
n=len(sh); p1=int(n*0.9); p2=p1+int(n*0.05); test = sh[p2:]
hx2, dy2, _ = collect(test, 3)
sel2 = np.abs(dy2) >= 0.05
print(f"test trajectories: {len(hx2)} pairs; Spearman {spearman(dy2,hx2):.3f}; turning n={sel2.sum()} sign agr {np.mean(np.sign(hx2[sel2])==np.sign(dy2[sel2])):.2f} median |flow| {np.median(np.abs(hx2[sel2])):.2f}")
