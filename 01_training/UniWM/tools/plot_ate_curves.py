"""Position error of the predicted pose against the reference pose as a function of the step index.

For every episode the predicted poses are integrated from the executed actions (or read from `predicted_coords`
when present) and compared with the reference poses at the same step index; the plot shows the mean per-step
error and the cumulative RMSE (ATE up to that step) over the episodes that reach the step.

usage: python tools/plot_ate_curves.py --runs base=output/evals/1x200_evaltask25 stop=output/evals/stopft_evaltask25 \
           --out output/figs/ate_curves.png
"""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from closed_loop_breakdown import load_pkl, parse_action


def episode_errors(res, traj_dir, max_steps):
    if "reference_coords" in res:
        ref = np.asarray(res["reference_coords"])[:, :2]
    else:
        ref = np.asarray(load_pkl(os.path.join(traj_dir, "traj_data.pkl"))["position"])[:, :2]
    if "predicted_coords" in res:
        pred = np.asarray(res["predicted_coords"])[:, :2]
    else:
        acts = np.array([parse_action(a)[:2] for a in res["trajectory_actions"]]).reshape(-1, 2)
        pred = np.vstack([np.zeros((1, 2)), np.cumsum(acts, axis=0)])
    K = min(len(ref), len(pred), max_steps + 1)
    return np.linalg.norm(pred[1:K] - ref[1:K], axis=1)   # error after step 1..K-1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="name=eval_dir"); ap.add_argument("--data", default="data/1x_nav")
    ap.add_argument("--out", default="output/figs/ate_curves.png"); ap.add_argument("--max_steps", type=int, default=25)
    args = ap.parse_args()
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    table = {}
    for run in args.runs:
        name, path = run.split("=", 1); per_step = []
        for traj in sorted(os.listdir(path)):
            f = os.path.join(path, traj, "eval_result.json")
            if os.path.isfile(f):
                per_step.append(episode_errors(json.load(open(f)), os.path.join(args.data, traj), args.max_steps))
        K = max(len(e) for e in per_step); mean_err, rmse, n = [], [], []
        for k in range(K):
            vals = np.array([e[k] for e in per_step if len(e) > k]); n.append(len(vals))
            mean_err.append(vals.mean())
            upto = np.concatenate([e[:k + 1] for e in per_step if len(e) > k]); rmse.append(np.sqrt(np.mean(upto ** 2)))
        steps = np.arange(1, K + 1)
        axes[0].plot(steps, mean_err, marker="o", ms=3, label=name); axes[1].plot(steps, rmse, marker="o", ms=3, label=name)
        table[name] = (mean_err, rmse, n)
    axes[0].set_xlabel("step"); axes[0].set_ylabel("mean position error vs reference (units)"); axes[0].set_title("per-step error")
    axes[1].set_xlabel("step"); axes[1].set_ylabel("cumulative RMSE (units)"); axes[1].set_title("ATE up to step k")
    for ax in axes:
        ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); os.makedirs(os.path.dirname(args.out), exist_ok=True); fig.savefig(args.out, dpi=150)
    for name, (m, r, n) in table.items():
        print(name, "| step 5 / 10 / 15 / 20 / 25 mean error:", " / ".join(f"{m[k-1]:.3f}" if len(m) >= k else "-" for k in (5, 10, 15, 20, 25)),
              "| episodes at step 25:", n[24] if len(n) >= 25 else "-")
    print("saved", args.out)


if __name__ == "__main__":
    main()
