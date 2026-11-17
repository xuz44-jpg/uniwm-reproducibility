"""Does the disagreement between sampled candidate actions predict the error of the executed step?

Reads the per-step logs of a closed-loop evaluation run with --action_samples > 1 (`steps[*].action_std`) and
relates the disagreement d = max per-component std to (a) the error of the executed action against the recorded
action at the same step index and (b) the position error after the step. Prints Spearman correlations, the mean
error by disagreement quartile, the malformed-candidate rate, and a suggested gate threshold (75th percentile).

usage: python tools/disagreement_analysis.py --run output/evals/base_sc5_evaltask25 [--out output/figs/disagreement_base.png]
"""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from closed_loop_breakdown import load_pkl, parse_action


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1]) if len(a) > 2 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True); ap.add_argument("--data", default="data/1x_nav"); ap.add_argument("--out", default=None)
    args = ap.parse_args()
    D, Ea, Ep, n_mal, n_cand, n_gated, n_steps = [], [], [], 0, 0, 0, 0
    for traj in sorted(os.listdir(args.run)):
        f = os.path.join(args.run, traj, "eval_result.json")
        if not os.path.isfile(f):
            continue
        res = json.load(open(f)); steps = res.get("steps", [])
        if not steps or "action_std" not in steps[0]:
            continue
        d = load_pkl(os.path.join(args.data, traj, "traj_data.pkl")); delta = np.asarray(d["delta"]); ref = np.asarray(d["position"])[:, :2]
        pred = np.asarray(res["predicted_coords"])[:, :2] if "predicted_coords" in res else None
        for i, st in enumerate(steps):
            n_steps += 1; n_gated += bool(st.get("gated")); n_mal += st.get("n_malformed", 0); n_cand += len(st.get("candidates", []))
            if st.get("action_std") is None or i >= len(delta):
                continue
            a = parse_action(st["predicted_action"]); D.append(max(st["action_std"]))
            Ea.append(float(np.linalg.norm(np.array(a) - delta[i][:3])))
            Ep.append(float(np.linalg.norm(pred[i + 1] - ref[i + 1])) if pred is not None and i + 1 < min(len(pred), len(ref)) else np.nan)
    D, Ea, Ep = np.array(D), np.array(Ea), np.array(Ep)
    print(f"steps {n_steps}, with disagreement {len(D)}, gated {n_gated}, malformed candidates {n_mal}/{n_cand}")
    if len(D) == 0:
        return
    print(f"disagreement: median {np.median(D):.3f}, p75 {np.percentile(D, 75):.3f}, p90 {np.percentile(D, 90):.3f}")
    print(f"Spearman(d, action error vs recorded) = {spearman(D, Ea):.3f}; Spearman(d, position error after step) = {spearman(D[~np.isnan(Ep)], Ep[~np.isnan(Ep)]):.3f}")
    q = np.percentile(D, [25, 50, 75]); bins = np.digitize(D, q)
    for b in range(4):
        sel = bins == b
        print(f"  disagreement quartile {b + 1}: n={sel.sum():3d}  mean action error {Ea[sel].mean():.3f}  mean position error {np.nanmean(Ep[sel]):.3f}")
    print(f"suggested --gate_std (p75): {np.percentile(D, 75):.3f}")
    if args.out:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
        ax[0].scatter(D, Ea, s=8, alpha=0.6); ax[0].set_xlabel("disagreement (max std)"); ax[0].set_ylabel("action error vs recorded")
        ax[1].scatter(D, Ep, s=8, alpha=0.6); ax[1].set_xlabel("disagreement (max std)"); ax[1].set_ylabel("position error after step")
        fig.tight_layout(); os.makedirs(os.path.dirname(args.out), exist_ok=True); fig.savefig(args.out, dpi=150); print("saved", args.out)


if __name__ == "__main__":
    main()
