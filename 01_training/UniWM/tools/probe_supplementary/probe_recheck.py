"""Two low-cost checks requested by the review, before any re-run of the probe.

A. Measurement validity of the flow proxy
   A1 known shifts: translate real frames (raw and VQ-reconstructed) horizontally by s px and check that the proxy
      recovers s.
   A2 temporal alignment: correlation of flow(k, k+1) with dyaw[k-1], dyaw[k], dyaw[k+1].
   A3 clustering: how many trajectories the calibration's |dyaw| bins come from.
B. Paired LPIPS comparison per frame: between-command distance (all 4 cross-sample pairs) minus the mean of the two
   same-command distances, for each command pair, with a trajectory-clustered bootstrap CI, for all six models.
"""
import glob, json, os, pickle, random, sys
import numpy as np, cv2, torch, lpips
import torchvision.transforms as T
from PIL import Image
from _paths import REPO, TOOLS, DATA
sys.path.insert(0, TOOLS)
from action_probe_analysis import flow_proxy, spearman
os.chdir(REPO); root = "data/1x_nav"
rng = np.random.default_rng(0)

def shift(img, s):
    a = np.asarray(img.convert("RGB").resize((256, 256))).copy()
    M = np.float32([[1, 0, s], [0, 1, 0]])
    return Image.fromarray(cv2.warpAffine(a, M, (256, 256), borderMode=cv2.BORDER_REFLECT))

man = json.load(open(glob.glob("output/base_probeimage*/action_probe_results/manifest_rank0.json")[0]))
print("=== A1 known horizontal shifts (proxy should return s) ===")
for dom in ("raw", "vq"):
    for s in (0.5, 1, 2, 4, 8, 16, 32):
        rec = []
        for m in man:
            base = Image.open(f"{root}/{m['traj']}/{m['k']}.jpg") if dom == "raw" else Image.open(m["current"])
            b = base.convert("RGB").resize((256, 256))
            rec.append(flow_proxy(b, shift(b, s))[0])
        rec = np.array(rec)
        print(f"  {dom} s={s:5.1f}px: recovered median {np.median(rec):+7.2f}  p10 {np.percentile(rec,10):+7.2f}  p90 {np.percentile(rec,90):+7.2f}  frames with |rec-s|<0.5*s: {np.mean(np.abs(rec-s)<0.5*s):.2f}")

names = sorted(d for d in os.listdir(root) if d.startswith("traj_"))
random.seed(42); sh = names[:]; random.shuffle(sh); n = len(sh); p1 = int(n * 0.9); p2 = p1 + int(n * 0.05); test = sh[p2:]
def pairs(trajs, stride):
    out = []
    for t in trajs:
        dl = np.asarray(pickle.load(open(f"{root}/{t}/traj_data.pkl", "rb"))["delta"])
        for k in range(1, len(dl) - 1, stride):
            h = flow_proxy(Image.open(f"{root}/{t}/{k}.jpg").convert("RGB"), Image.open(f"{root}/{t}/{k+1}.jpg").convert("RGB"))[0]
            out.append((t, k, h, dl[k - 1][2], dl[k][2], dl[k + 1][2]))
    return out
print("\n=== A2 temporal alignment: which dyaw index matches flow(k,k+1) ===")
for label, trajs, stride in (("calibration trajs 0-59", names[:60], 3), ("test trajs", test, 2)):
    P = pairs(trajs, stride); h = np.array([p[2] for p in P])
    for j, lag in ((3, "dyaw[k-1]"), (4, "dyaw[k]  "), (5, "dyaw[k+1]")):
        d = np.array([p[j] for p in P]); sel = np.abs(d) >= 0.05
        print(f"  {label:22s} {lag}: Spearman {spearman(d, h):+.3f}; turning n={sel.sum()} sign agr {np.mean(np.sign(h[sel]) == np.sign(d[sel])):.2f}")
    if label.startswith("calibration"):
        print("=== A3 clustering of the calibration bins (by trajectory) ===")
        d = np.array([p[4] for p in P]); t = np.array([p[0] for p in P])
        for lo, hi in ((0.05, 0.1), (0.1, 0.2), (0.2, 0.4)):
            m = (np.abs(d) >= lo) & (np.abs(d) < hi)
            if m.sum():
                per = {}
                for tt, hh in zip(t[m], np.abs(h[m])): per.setdefault(tt, []).append(hh)
                top = sorted(per.items(), key=lambda kv: -len(kv[1]))[:3]
                print(f"  |dyaw| [{lo},{hi}): {m.sum()} pairs from {len(per)} trajectories; median |flow| {np.median(np.abs(h[m])):.2f}; "
                      f"trajectory medians p25/p50/p75 {np.percentile([np.median(v) for v in per.values()],[25,50,75]).round(2).tolist()}; largest contributors {[(k, len(v)) for k, v in top]}")

print("\n=== B paired LPIPS: between-command minus same-command, per frame, trajectory-clustered bootstrap ===")
net = lpips.LPIPS(net="alex", verbose=False).cuda().eval()
tf = lambda im: (T.ToTensor()(im.convert("RGB").resize((256, 256))) * 2 - 1)[None].cuda()
def lp(a, b):
    with torch.no_grad(): return float(net(a, b).item())
PAIRS = [("left", "right"), ("recorded", "mirrored"), ("zero", "forward"), ("recorded", "zero"), ("zero", "left"), ("zero", "right")]
for mdir in sorted(glob.glob("output/*_probeimage*/action_probe_results")):
    model = os.path.basename(os.path.dirname(mdir)).split("_probeimage")[0]
    M = json.load(open(f"{mdir}/manifest_rank0.json")); per = {p: [] for p in PAIRS}; traj_of = []
    for m in M:
        X = {c: [tf(Image.open(p)) for p in g["paths"] if p] for c, g in m["generated"].items()}
        if any(len(v) < 2 for v in X.values()): continue
        traj_of.append(m["traj"]); w = {c: lp(v[0], v[1]) for c, v in X.items()}
        for a, b in PAIRS:
            between = np.mean([lp(X[a][i], X[b][j]) for i in range(2) for j in range(2)])
            per[(a, b)].append(between - 0.5 * (w[a] + w[b]))
    trajs = np.array(traj_of); uniq = np.unique(trajs)
    line = [f"{model:7s} frames={len(trajs)}"]
    for p in PAIRS:
        v = np.array(per[p]); boots = []
        for _ in range(4000):
            pick = rng.choice(uniq, len(uniq)); idx = np.concatenate([np.where(trajs == u)[0] for u in pick]); boots.append(v[idx].mean())
        lo, hi = np.percentile(boots, [2.5, 97.5]); line.append(f"{p[0]}-{p[1]} {v.mean():+.4f} [{lo:+.4f},{hi:+.4f}]")
    print("  " + " | ".join(line), flush=True)
print("done")
