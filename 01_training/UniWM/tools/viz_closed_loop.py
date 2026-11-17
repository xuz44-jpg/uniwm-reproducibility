"""Render every closed-loop episode of an evaluation directory as one strip image for inspection.

Row 1: reference frames of the test trajectory with the recorded action (dx, dyaw) executed after each frame.
Row 2: imagined frames with the predicted action executed after each; the header carries the episode metrics.
Also writes <out>/episodes.csv (one row per episode, with an empty `failure_mode` column to fill in by hand).

usage: python tools/viz_closed_loop.py --run output/evals/stopft_evaltask25 --out output/figs/episodes/stopft
"""
import argparse, csv, json, os, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from closed_loop_breakdown import load_pkl, parse_action, summarise


def font(size):
    for f in ("/System/Library/Fonts/Helvetica.ttc", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if os.path.exists(f):
            return ImageFont.truetype(f, size)
    return ImageFont.load_default()


def strip(frames, labels, thumb, pad=6, text_h=30):
    w = len(frames) * (thumb + pad) + pad; h = thumb + text_h + pad
    im = Image.new("RGB", (w, h), "white"); d = ImageDraw.Draw(im); f = font(13)
    for i, (fr, lab) in enumerate(zip(frames, labels)):
        x = pad + i * (thumb + pad)
        if fr is not None:
            im.paste(fr.resize((thumb, thumb)), (x, pad))
        else:
            d.rectangle([x, pad, x + thumb, pad + thumb], outline="gray")
        d.text((x, pad + thumb + 2), lab, fill="black", font=f)
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True); ap.add_argument("--data", default="data/1x_nav")
    ap.add_argument("--out", required=True); ap.add_argument("--thumb", type=int, default=150)
    ap.add_argument("--max_steps", type=int, default=25); ap.add_argument("--trajs", nargs="*", default=None)
    args = ap.parse_args(); os.makedirs(args.out, exist_ok=True)
    rows = []
    for traj in sorted(os.listdir(args.run)):
        f = os.path.join(args.run, traj, "eval_result.json")
        if not os.path.isfile(f) or (args.trajs and traj not in args.trajs):
            continue
        res = json.load(open(f)); summ = summarise(None, res)
        d = load_pkl(os.path.join(args.data, traj, "traj_data.pkl")); delta = np.asarray(d["delta"])
        n_ref = len(d["position"]); K = min(args.max_steps, n_ref)
        ref_frames = [Image.open(os.path.join(args.data, traj, f"{k}.jpg")).convert("RGB") for k in range(K)]
        ref_labels = [f"t={k}  dx={delta[k][0]:+.2f} dyaw={delta[k][2]:+.2f}" if k < len(delta) else f"t={k} (goal)" for k in range(K)]
        acts = res["trajectory_actions"]; steps = len(acts)
        img_frames, img_labels = [Image.open(os.path.join(args.data, traj, "0.jpg")).convert("RGB")], ["t=0 (start)"]
        for k in range(1, min(steps, args.max_steps) + 1):
            p = os.path.join(args.run, traj, f"step_{k}_observation.png")
            img_frames.append(Image.open(p).convert("RGB") if os.path.isfile(p) else None)
            a = acts[k - 1]; dx, dy, dyaw = parse_action(a)
            img_labels.append("Stop" if a.strip().lower().startswith("stop") else f"a{k}: dx={dx:+.2f} dyaw={dyaw:+.2f}")
        s1, s2 = strip(ref_frames, ref_labels, args.thumb), strip(img_frames, img_labels, args.thumb)
        head = 26; im = Image.new("RGB", (max(s1.width, s2.width), head + s1.height + s2.height), "white")
        ImageDraw.Draw(im).text((6, 6), f"{traj}  steps={steps}  stop={'yes' if summ['stop'] else 'no'}  start={summ['start_dist']:.2f}  "
                                       f"final={summ['distance']:.2f}  closest={summ['closest']:.2f}  ATE={summ['ate']:.3f}   (row 1: reference, row 2: imagined)",
                                fill="black", font=font(15))
        im.paste(s1, (0, head)); im.paste(s2, (0, head + s1.height))
        im.save(os.path.join(args.out, f"{traj}.jpg"), quality=88)
        rows.append(dict(traj=traj, ref_steps=n_ref - 1, steps=steps, stop=int(summ["stop"]), start=round(summ["start_dist"], 3),
                         final=round(summ["distance"], 3), closest=round(summ["closest"], 3), ate=round(summ["ate"], 3), failure_mode=""))
    with open(os.path.join(args.out, "episodes.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"{len(rows)} episodes rendered to {args.out}")


if __name__ == "__main__":
    main()
