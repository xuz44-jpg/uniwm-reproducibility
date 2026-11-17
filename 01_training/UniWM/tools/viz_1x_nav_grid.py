"""Render N trajectories as rows of frames with the (dx, dyaw) action under each frame.
Usage: python tools/viz_1x_nav_grid.py --data_dir data/1x_nav --n 25 --out output/1x_nav_grid25.jpg
Trajectories are picked evenly across the range of total heading change, so straight runs and left/right turns all appear.
"""
import argparse, glob, os, pickle, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ap = argparse.ArgumentParser()
ap.add_argument("--data_dir", default="data/1x_nav")
ap.add_argument("--n", type=int, default=25)
ap.add_argument("--tile", type=int, default=112)
ap.add_argument("--out", default="output/1x_nav_grid25.jpg")
args = ap.parse_args()

def load_font(size):
    for f in ["/System/Library/Fonts/Supplemental/Arial.ttf", "/System/Library/Fonts/Helvetica.ttc",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"]:
        if os.path.exists(f):
            try:
                return ImageFont.truetype(f, size)
            except Exception:
                pass
    return ImageFont.load_default()

trajs = []
for d in sorted(glob.glob(os.path.join(args.data_dir, "traj_*"))):
    p = os.path.join(d, "traj_data.pkl")
    if not os.path.exists(p):
        continue
    md = pickle.load(open(p, "rb"))
    delta = np.asarray(md["delta"], dtype=float)
    trajs.append((d, delta, float(delta[:, 2].sum()), float(np.linalg.norm(delta[:, :2], axis=1).sum()), md))
trajs.sort(key=lambda t: t[2])                      # by total heading change: right turns ... straight ... left turns
idx = np.linspace(0, len(trajs) - 1, args.n).round().astype(int)
picked = [trajs[i] for i in idx]

T = args.tile; LAB = 30; ROW = T + LAB + 6; LEFT = 150; TOP = 34
ncol = max(len(t[1]) + 1 for t in picked)
W = LEFT + ncol * (T + 2) + 8; H = TOP + len(picked) * ROW + 8
canvas = Image.new("RGB", (W, H), (245, 245, 242)); draw = ImageDraw.Draw(canvas)
f_small, f_lab, f_hdr = load_font(12), load_font(13), load_font(15)
draw.text((8, 8), f"{args.data_dir}: {len(picked)} of {len(trajs)} trajectories, sorted by total heading change (top: right turns, bottom: left turns). "
                  f"Label under frame k = action k->k+1: dx (forward, 1X units) / dyaw (rad, +left). 5 Hz.", fill=(40, 40, 40), font=f_hdr)
for r, (d, delta, tot_yaw, tot_dist, md) in enumerate(picked):
    y0 = TOP + r * ROW
    name = os.path.basename(d); n = len(delta) + 1
    draw.text((8, y0 + 4), name, fill=(20, 20, 20), font=f_lab)
    draw.text((8, y0 + 22), f"{n} frames, shard {md.get('shard', '?')}", fill=(90, 90, 90), font=f_small)
    draw.text((8, y0 + 40), f"dist {tot_dist:.2f}", fill=(90, 90, 90), font=f_small)
    draw.text((8, y0 + 56), f"yaw {tot_yaw:+.2f} rad", fill=(90, 90, 90), font=f_small)
    for k in range(n):
        x0 = LEFT + k * (T + 2)
        fp = os.path.join(d, f"{k}.jpg")
        if os.path.exists(fp):
            im = Image.open(fp).convert("RGB").resize((T, T), Image.BILINEAR)
            canvas.paste(im, (x0, y0))
        if k < len(delta):
            dx, dy, dyaw = delta[k]
            c_dx = (200, 90, 0) if dx < -1e-3 else (30, 30, 30)
            c_yaw = (30, 90, 200) if dyaw > 0.02 else ((200, 40, 40) if dyaw < -0.02 else (30, 30, 30))
            draw.text((x0 + 2, y0 + T + 2), f"dx {dx:+.2f}", fill=c_dx, font=f_small)
            draw.text((x0 + 2, y0 + T + 15), f"yaw {dyaw:+.2f}", fill=c_yaw, font=f_small)
        else:
            draw.text((x0 + 2, y0 + T + 2), "stop", fill=(120, 120, 120), font=f_small)
os.makedirs(os.path.dirname(args.out), exist_ok=True)
canvas.save(args.out, quality=88)
print(f"saved {args.out} ({W}x{H}); rows = {[os.path.basename(t[0]) for t in picked]}")
