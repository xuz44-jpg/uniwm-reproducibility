import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from pc_flow import gray256, pc_shift
from PIL import Image
from _paths import DATA as root
names = sorted(d for d in os.listdir(root) if d.startswith("traj_"))
rng = np.random.default_rng(0)
imgs = [gray256(Image.open(f"{root}/{t}/{rng.integers(0, 15)}.jpg")) for t in rng.choice(names, 40, replace=False)]
def shift_img(a, s):   # subpixel horizontal shift via linear interpolation, reflect border
    x = np.arange(256) - s; x0 = np.floor(x).astype(int); f = x - x0
    x0c = np.clip(x0, 0, 255); x1c = np.clip(x0 + 1, 0, 255)
    return a[:, x0c] * (1 - f) + a[:, x1c] * f
for s in (0.5, 1, 2, 4, 8, 16, 32):
    rec = np.array([pc_shift(a, shift_img(a, s))[0] for a in imgs])
    print(f"s={s:5.1f}: median {np.median(rec):+.2f} p10 {np.percentile(rec,10):+.2f} p90 {np.percentile(rec,90):+.2f} frac within 0.5s {np.mean(np.abs(rec-s)<0.5*s):.2f}")
