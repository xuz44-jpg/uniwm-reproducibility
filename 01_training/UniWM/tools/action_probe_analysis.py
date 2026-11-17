"""Analysis of the action-conditioning probe (runs where cv2 is available, e.g. on the server).

1. Calibrates an optical-flow yaw proxy on real consecutive frames: median horizontal flow (px) of the frame pair
   against the recorded dyaw, and mean radial flow (divergence) against the recorded dx.
2. For every probed frame: LPIPS between generations of the same condition (sampling variation) and between
   conditions; the flow proxy of every generated frame; whether the proxy follows the commanded turn.

usage: python tools/action_probe_analysis.py --probe output/<note>/action_probe_results --data data/1x_nav --out output/figs/action_probe
"""
import argparse, glob, json, os, pickle, sys
import numpy as np, cv2
from PIL import Image


def flow_proxy(img_a, img_b, size=256):
    """(median horizontal flow, mean radial flow) from a to b, in pixels at `size`."""
    a = cv2.cvtColor(np.asarray(img_a.resize((size, size))), cv2.COLOR_RGB2GRAY)
    b = cv2.cvtColor(np.asarray(img_b.resize((size, size))), cv2.COLOR_RGB2GRAY)
    flow = cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 4, 21, 5, 7, 1.5, cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
    h, w = flow.shape[:2]; ys, xs = np.mgrid[0:h, 0:w]; xs = xs - w / 2; ys = ys - h / 2
    r = np.sqrt(xs ** 2 + ys ** 2) + 1e-6
    radial = (flow[..., 0] * xs + flow[..., 1] * ys) / r
    c = slice(h // 4, 3 * h // 4)   # central region, less affected by the robot's own body at the bottom
    return float(np.median(flow[c, :, 0])), float(np.mean(radial[c, :]))


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float); ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", required=True); ap.add_argument("--data", default="data/1x_nav"); ap.add_argument("--out", default="output/figs/action_probe")
    ap.add_argument("--calib_trajs", type=int, default=60)
    args = ap.parse_args(); os.makedirs(args.out, exist_ok=True)

    # ---- 1. calibration on real frame pairs ----
    trajs = sorted(d for d in os.listdir(args.data) if d.startswith("traj_"))[: args.calib_trajs]
    hx, rad, dyaw, dx = [], [], [], []
    for t in trajs:
        d = pickle.load(open(os.path.join(args.data, t, "traj_data.pkl"), "rb")); delta = np.asarray(d["delta"])
        for k in range(0, len(delta), 3):
            a = Image.open(os.path.join(args.data, t, f"{k}.jpg")).convert("RGB"); b = Image.open(os.path.join(args.data, t, f"{k + 1}.jpg")).convert("RGB")
            h, r = flow_proxy(a, b); hx.append(h); rad.append(r); dyaw.append(delta[k][2]); dx.append(delta[k][0])
    hx, rad, dyaw, dx = map(np.array, (hx, rad, dyaw, dx))
    slope = float(np.polyfit(dyaw, hx, 1)[0])
    turn = np.abs(dyaw) >= 0.05
    print(f"calibration on {len(hx)} real pairs: Spearman(dyaw, horizontal flow) = {spearman(dyaw, hx):.3f}; slope {slope:.1f} px/rad; "
          f"sign agreement on |dyaw|>=0.05: {np.mean(np.sign(hx[turn]) == np.sign(dyaw[turn])):.2f} (n={turn.sum()}); "
          f"Spearman(dx, radial flow) = {spearman(dx, rad):.3f}")
    thr = float(np.percentile(np.abs(hx[~turn]), 75))   # 'no turn' band from straight pairs
    print(f"straight pairs (|dyaw|<0.05): |horizontal flow| p75 = {thr:.2f} px (used as the no-turn band)")

    # ---- 2. probe ----
    manifest = []
    for f in glob.glob(os.path.join(args.probe, "manifest_rank*.json")):
        manifest += json.load(open(f))
    print(f"\nprobe frames: {len(manifest)} ({sum(m['kind'] == 'turn' for m in manifest)} turn, {sum(m['kind'] == 'straight' for m in manifest)} straight)")
    try:
        import torch, lpips, torchvision.transforms as T
        net = lpips.LPIPS(net="alex", verbose=False).eval()
        def lp(a, b):
            ta, tb = (T.ToTensor()(x.convert("RGB").resize((256, 256))) * 2 - 1 for x in (a, b))
            with torch.no_grad():
                return float(net(ta[None], tb[None]).item())
    except Exception as e:
        print("lpips unavailable:", e); lp = None

    within, between, rows = [], [], []
    conds = ["recorded", "mirrored", "zero", "forward", "left", "right"]
    per_cond = {c: {"hx": [], "rad": [], "cmd": []} for c in conds}
    n_fail = 0
    for m in manifest:
        cur = Image.open(m["current"]).convert("RGB"); real = Image.open(m["real_next"]).convert("RGB")
        imgs = {}
        for c in conds:
            g = m["generated"].get(c, {}); ps = [p for p in g.get("paths", []) if p]; n_fail += len(g.get("paths", [])) - len(ps)
            imgs[c] = [Image.open(p).convert("RGB") for p in ps]
            for im in imgs[c]:
                h, r = flow_proxy(cur, im); per_cond[c]["hx"].append(h); per_cond[c]["rad"].append(r); per_cond[c]["cmd"].append(g["action"][2])
        hreal, _ = flow_proxy(cur, real)
        rows.append(dict(traj=m["traj"], k=m["k"], kind=m["kind"], rec_dyaw=m["recorded"][2], real_hx=hreal,
                         **{f"{c}_hx": float(np.mean(per_cond[c]["hx"][-len(imgs[c]):])) if imgs[c] else np.nan for c in conds}))
        if lp:
            for c in conds:
                if len(imgs[c]) >= 2: within.append(lp(imgs[c][0], imgs[c][1]))
            for c1, c2 in (("recorded", "mirrored"), ("left", "right"), ("zero", "forward"), ("recorded", "zero")):
                if imgs[c1] and imgs[c2]: between.append(lp(imgs[c1][0], imgs[c2][0]))
    print(f"failed generations: {n_fail}")
    if lp:
        print(f"LPIPS within condition (same action, two samples): {np.mean(within):.3f} ± {np.std(within):.3f}; between conditions (different actions): {np.mean(between):.3f} ± {np.std(between):.3f}")
    print("\nhorizontal-flow proxy of the imagined frame, by condition (positive = scene moves right = left turn if calibrated so):")
    for c in conds:
        hx_c = np.array(per_cond[c]["hx"]); cmd = np.array(per_cond[c]["cmd"])
        line = f"  {c:9s} n={len(hx_c):3d} mean flow {hx_c.mean():+6.2f} px, expected {slope * cmd.mean():+6.2f} px from the command"
        if c in ("left", "right", "mirrored", "recorded"):
            sel = np.abs(cmd) >= 0.05
            if sel.any():
                line += f"; sign agreement with the command on |dyaw|>=0.05: {np.mean(np.sign(hx_c[sel]) == np.sign(slope * cmd[sel])):.2f} (n={sel.sum()})"
        if c in ("zero", "forward"):
            line += f"; |flow| <= no-turn band: {np.mean(np.abs(hx_c) <= thr):.2f}"
        print(line)
    # direct comparison: left vs right on the same frame
    lr = [(r["left_hx"], r["right_hx"]) for r in rows if not np.isnan(r["left_hx"]) and not np.isnan(r["right_hx"])]
    if lr:
        l, r_ = np.array(lr).T; print(f"\nsame frame, left vs right command: left-turn flow > right-turn flow in {np.mean(l > r_):.2f} of {len(lr)} frames (mean difference {np.mean(l - r_):+.2f} px; real turn of that size would give about {abs(slope) * 0.4:.1f} px)")
    rm = [(r["recorded_hx"], r["mirrored_hx"], r["rec_dyaw"]) for r in rows if r["kind"] == "turn" and not np.isnan(r["recorded_hx"]) and not np.isnan(r["mirrored_hx"])]
    if rm:
        a, b, d = np.array(rm).T; print(f"turn frames, recorded vs mirrored command: flow follows the command sign in {np.mean(np.sign(a - b) == np.sign(slope * d)):.2f} of {len(rm)} frames")
    # real next frame vs imagined frame under the recorded action
    rr = [(r["real_hx"], r["recorded_hx"]) for r in rows if not np.isnan(r["recorded_hx"])]
    a, b = np.array(rr).T; print(f"recorded action: Spearman(flow of real next frame, flow of imagined frame) = {spearman(a, b):.3f} over {len(rr)} frames")
    json.dump(rows, open(os.path.join(args.out, "probe_rows.json"), "w"), indent=1)
    # contact sheet for the first few turn frames
    sheet = [m for m in manifest if m["kind"] == "turn"][:6]
    if sheet:
        tw = 160; W = tw * (2 + len(conds)); im = Image.new("RGB", (W, tw * len(sheet)), "white")
        from PIL import ImageDraw
        for i, m in enumerate(sheet):
            tiles = [Image.open(m["current"]), Image.open(m["real_next"])] + [Image.open(m["generated"][c]["paths"][0]) if m["generated"][c]["paths"][0] else Image.new("RGB", (tw, tw), "gray") for c in conds]
            for j, t in enumerate(tiles):
                im.paste(t.convert("RGB").resize((tw, tw)), (j * tw, i * tw))
            ImageDraw.Draw(im).text((2, i * tw + 2), f"{m['traj']} k={m['k']} dyaw={m['recorded'][2]:+.2f}", fill="yellow")
        ImageDraw.Draw(im).text((2, 14), "current | real next | " + " | ".join(conds), fill="yellow")
        im.save(os.path.join(args.out, "probe_sheet.jpg"), quality=85); print("saved", os.path.join(args.out, "probe_sheet.jpg"))


if __name__ == "__main__":
    main()
