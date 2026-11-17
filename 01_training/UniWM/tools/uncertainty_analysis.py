"""Zero-GPU analyses of the closed-loop logs.

A. Rollout selection by the model's own disagreement: for each episode with several 5-sample runs, pick the run
   with the lowest mean disagreement and compare its final distance with the mean over runs and the best run.
B. Early failure prediction: disagreement over the first k steps vs the final distance of the episode.
C. Task geometry: per-episode outcome (mean over all runs and models) vs total heading change / start distance.
D. Soft stopping: Stop votes per step vs true distance to goal, and offline replay of "stop at first step with
   >= t votes".

usage: python tools/uncertainty_analysis.py --sc5 base=dir1,dir2,dir3 stop=dir1,dir2,dir3 --all dirA dirB ... --data data/1x_nav
"""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from closed_loop_breakdown import load_pkl, parse_action, summarise


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float); m = ~(np.isnan(a) | np.isnan(b)); a, b = a[m], b[m]
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1]) if len(a) > 2 else float("nan")


def load_run(path):
    out = {}
    for n in sorted(os.listdir(path)):
        f = os.path.join(path, n, "eval_result.json")
        if os.path.isfile(f):
            out[n] = json.load(open(f))
    return out


def disagreement(res):
    return [max(st["action_std"]) if st.get("action_std") else np.nan for st in res.get("steps", [])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sc5", nargs="+", default=[], help="name=dir1,dir2,... (5-sample runs)")
    ap.add_argument("--all", nargs="*", default=[], help="all closed-loop run dirs for the geometry analysis")
    ap.add_argument("--data", default="data/1x_nav"); ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    # ---------- A + B + D on the 5-sample runs ----------
    for cfg in args.sc5:
        name, dirs = cfg.split("=", 1); runs = [load_run(d) for d in dirs.split(",")]
        names = sorted(set.intersection(*[set(r) for r in runs]))
        print(f"\n===== {name}: {len(runs)} five-sample runs, {len(names)} episodes =====")
        # A. rollout selection
        sel, mean_d, best_d, worst_d, rand_expect = [], [], [], [], []
        for n in names:
            ds = [r[n]["distance_to_goal"] for r in runs]
            dis = [np.nanmean(disagreement(r[n])) for r in runs]
            i = int(np.nanargmin(dis)); sel.append(ds[i]); mean_d.append(np.mean(ds)); best_d.append(min(ds)); worst_d.append(max(ds))
        print(f"A. select the run with the lowest mean disagreement: selected {np.mean(sel):.3f} | mean over runs {np.mean(mean_d):.3f} | "
              f"best run (oracle) {np.mean(best_d):.3f} | worst {np.mean(worst_d):.3f}; selected beats run-mean in {sum(s < m for s, m in zip(sel, mean_d))}/{len(names)} episodes")
        # early-k selection
        sel_k = []
        for n in names:
            ds = [r[n]["distance_to_goal"] for r in runs]; dis = [np.nanmean(disagreement(r[n])[:args.k]) for r in runs]
            sel_k.append(ds[int(np.nanargmin(dis))])
        print(f"   selection by the first {args.k} steps only: {np.mean(sel_k):.3f}")
        # B. early failure prediction (pooled over runs)
        early, total, final, prog = [], [], [], []
        for r in runs:
            for n in names:
                d = disagreement(r[n]); early.append(np.nanmean(d[:args.k])); total.append(np.nanmean(d)); final.append(r[n]["distance_to_goal"])
                s = summarise(None, r[n]); prog.append(1 - s["distance"] / s["start_dist"])
        print(f"B. Spearman(first-{args.k}-step disagreement, final distance) = {spearman(early, final):.3f}; (mean disagreement, final distance) = {spearman(total, final):.3f}; "
              f"(first-{args.k}, progress) = {spearman(early, prog):.3f}  [n = {len(final)} episodes]")
        q = np.nanpercentile(early, [50]); lo = np.array(early) <= q[0]
        print(f"   final distance, low first-{args.k} disagreement half: {np.mean(np.array(final)[lo]):.3f}; high half: {np.mean(np.array(final)[~lo]):.3f}")
        # D. soft stopping
        by_votes = {v: [] for v in range(6)}; fired = {t: [] for t in (1, 2, 3)}; actual = []
        for r in runs:
            for n in names:
                res = r[n]; goal = np.array(res["goal_position"]); pc = np.array(res["predicted_coords"])[:, :2]
                dist_after = np.linalg.norm(pc[1:] - goal, axis=1)   # distance after step i (i = 0..)
                actual.append(res["distance_to_goal"])
                for i, st in enumerate(res["steps"]):
                    if i < len(dist_after):
                        by_votes[int(st.get("n_stop", 0))].append(dist_after[i] if i > 0 else np.linalg.norm(pc[0] - goal))
                for t in (1, 2, 3):
                    hit = [i for i, st in enumerate(res["steps"]) if st.get("n_stop", 0) >= t]
                    fired[t].append(np.linalg.norm(pc[hit[0]] - goal) if hit else res["distance_to_goal"])   # stop before executing step hit[0]
        print("D. distance to goal at steps with n Stop votes: " + "; ".join(f"{v}: {np.mean(by_votes[v]):.2f} (n={len(by_votes[v])})" for v in range(6) if by_votes[v]))
        print("   offline rule 'stop at the first step with >= t votes': " + "; ".join(f"t={t}: {np.mean(fired[t]):.3f}" for t in (1, 2, 3)) + f" | actual (majority): {np.mean(actual):.3f}")

    # ---------- C. task geometry over all runs ----------
    if args.all:
        per = {}
        for d in args.all:
            for n, res in load_run(d).items():
                s = summarise(None, res); per.setdefault(n, []).append(1 - s["distance"] / s["start_dist"])
        names = sorted(per); geo = {}
        for n in names:
            t = load_pkl(os.path.join(args.data, n, "traj_data.pkl")); yaw = np.asarray(t["yaw"]).ravel(); p = np.asarray(t["position"])[:, :2]
            dyaw = np.diff(yaw); dyaw = (dyaw + np.pi) % (2 * np.pi) - np.pi
            geo[n] = dict(turn=float(np.abs(dyaw).sum()), net_turn=float(abs(dyaw.sum())), start=float(np.linalg.norm(p[-1] - p[0])), path=float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum()), n=len(p))
        prog = np.array([np.mean(per[n]) for n in names]); sd = np.array([np.std(per[n]) for n in names])
        print(f"\n===== C. task geometry: {len(names)} episodes, {len(args.all)} runs; mean progress {prog.mean():.2f} =====")
        for key, lab in (("turn", "total |heading change| (rad)"), ("net_turn", "net heading change"), ("start", "start distance"), ("path", "path length"), ("n", "frames")):
            print(f"   Spearman(progress, {lab}) = {spearman(prog, [geo[n][key] for n in names]):+.3f}")
        order = np.argsort(prog)
        print("   consistently worst (mean progress, sd over runs, turn, start):")
        for i in order[:6]:
            n = names[i]; print(f"     {n}: progress {prog[i]:+.2f} ± {sd[i]:.2f}, turn {geo[n]['turn']:.2f} rad, start {geo[n]['start']:.2f}")
        print("   consistently best:")
        for i in order[-4:]:
            n = names[i]; print(f"     {n}: progress {prog[i]:+.2f} ± {sd[i]:.2f}, turn {geo[n]['turn']:.2f} rad, start {geo[n]['start']:.2f}")
        hi = np.array([geo[n]["turn"] for n in names]) > np.median([geo[n]["turn"] for n in names])
        print(f"   mean progress, low-turn half: {prog[~hi].mean():.2f}; high-turn half: {prog[hi].mean():.2f}")


if __name__ == "__main__":
    main()
