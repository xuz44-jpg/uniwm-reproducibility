"""Per-subset breakdown of closed-loop results (clean vs same-recording test trajectories).

Reproduces the trajectory-level split of scripts/navigation.py (seed 42, 0.9/0.05/0.05, first
`--max_train_trajs` training trajectories), marks each test trajectory as 'clean' (its (shard, segment)
recording has no trajectory in the training subset) or 'same recording', and aggregates the per-trajectory
eval_result.json files of one or more closed-loop evaluation directories.

usage: python tools/closed_loop_breakdown.py --data data/1x_nav --max_train_trajs 200 \
           --runs base=output/evals/1x200_evaltask25 stop=output/evals/stopft_evaltask25
"""
import argparse, json, os, pickle, random, sys, types
import numpy as np

# pickles written by numpy 2 reference numpy._core
if not hasattr(np, "_core"):
    core = types.ModuleType("numpy._core"); core.multiarray = np.core.multiarray; core.numeric = np.core.numeric
    sys.modules["numpy._core"] = core; sys.modules["numpy._core.multiarray"] = np.core.multiarray
    sys.modules["numpy._core.numeric"] = np.core.numeric

STEP_LEN = 0.096  # mean step length of the test trajectories (SR thresholds are multiples of it)


def load_pkl(p):
    with open(p, "rb") as f:
        return pickle.load(f)


def split(data_root, ratios=(0.9, 0.05, 0.05), max_train=200):
    names = sorted(d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d)))
    random.seed(42); random.shuffle(names)
    n = len(names); p1 = int(n * ratios[0]); p2 = p1 + int(n * ratios[1])
    return names[:p1][:max_train], names[p1:p2], names[p2:]


def parse_action(s):
    if s.strip().lower().startswith("stop"):
        return 0.0, 0.0, 0.0
    d = {}
    for part in s.split(","):
        k, v = part.split(":"); d[k.strip()] = float(v)
    return d["dx"], d["dy"], d["dyaw"]


def summarise(traj_dir, res):
    acts = [parse_action(a) for a in res["trajectory_actions"]]
    pos = np.cumsum(np.array([[a[0], a[1]] for a in acts]), axis=0) if acts else np.zeros((0, 2))
    goal = np.array(res["goal_position"])
    d_path = np.linalg.norm(np.vstack([np.zeros((1, 2)), pos]) - goal, axis=1)
    stop = any(t.replace("<reserved08706>", "").strip().lower().startswith("stop") for t in res.get("trajectory_decoded", []))
    return dict(distance=res["distance_to_goal"], closest=float(d_path.min()), ate=res["ate"], rpe=res["rpe_trans"],
                steps=len(acts), stop=stop, start_dist=float(d_path[0]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/1x_nav"); ap.add_argument("--max_train_trajs", type=int, default=200)
    ap.add_argument("--runs", nargs="+", required=True, help="name=eval_dir")
    ap.add_argument("--oracle_steps", type=int, default=10)
    args = ap.parse_args()

    train, dev, test = split(args.data, max_train=args.max_train_trajs)
    held = test + dev   # dev trajectories are also unseen in training (--closed_loop_extra dev)
    rec = {n: (int(d["shard"]), int(d["segment"])) for n in train + held for d in [load_pkl(os.path.join(args.data, n, "traj_data.pkl"))]}
    train_recs = {rec[n] for n in train}
    subset = {n: ("same recording" if rec[n] in train_recs else "clean") for n in held}

    # oracle: ground-truth actions for `oracle_steps` steps
    oracle = {}
    for n in held:
        d = load_pkl(os.path.join(args.data, n, "traj_data.pkl"))
        p = np.asarray(d["position"])[:, :2]; k = min(args.oracle_steps, len(p) - 1)
        oracle[n] = float(np.linalg.norm(p[k] - p[-1]))

    for run in args.runs:
        name, path = run.split("=", 1)
        rows = {}
        for n in sorted(os.listdir(path)):
            f = os.path.join(path, n, "eval_result.json")
            if os.path.isfile(f):
                rows[n] = summarise(os.path.join(path, n), json.load(open(f)))
        print(f"\n== {name}: {len(rows)} trajectories, steps budget {max(r['steps'] for r in rows.values())}")
        for sub in ("clean", "same recording", "all"):
            sel = [r for n, r in rows.items() if sub == "all" or subset.get(n) == sub]
            if not sel:
                continue
            a = lambda key: np.mean([r[key] for r in sel])
            sr = lambda th: np.mean([r["distance"] < th for r in sel])
            osel = [oracle[n] for n in rows if sub == "all" or subset.get(n) == sub]
            print(f"  {sub:15s} n={len(sel):2d} start={a('start_dist'):.3f} dist={a('distance'):.3f} closest={a('closest'):.3f} "
                  f"ATE={a('ate'):.3f} RPE={a('rpe'):.3f} SR@1={sr(STEP_LEN):.2f} SR@3={sr(3*STEP_LEN):.2f} SR@0.5={sr(0.5):.2f} "
                  f"stop={sum(r['stop'] for r in sel)} | oracle@{args.oracle_steps} dist={np.mean(osel):.3f} SR@0.5={np.mean([o < 0.5 for o in osel]):.2f}")
        print("  per-trajectory:", {n: subset.get(n) for n in rows})


if __name__ == "__main__":
    main()
