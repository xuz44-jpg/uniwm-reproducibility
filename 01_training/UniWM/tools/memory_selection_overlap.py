"""How often do the `pose` and `recent` memory rules select the same frames?

Training-time (teacher forcing): for every frame k >= 1 of the given trajectories, entries are frames 0..k-1 with
reference poses. Closed-loop: for every evaluated episode, entries are the start pose plus the poses integrated
from the predicted actions (as in the evaluation loop), queried before each step.

usage: python tools/memory_selection_overlap.py --data data/1x_nav --max_train_trajs 200 [--episodes output/evals/<dir>]
"""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
from closed_loop_breakdown import load_pkl, split, parse_action
try:
    from uniwm.pose_memory import retrieve
except ImportError:  # no torch locally: same logic as uniwm/pose_memory.py
    import math

    def _wrap(a):
        return (a + math.pi) % (2 * math.pi) - math.pi

    def retrieve(cur_pose, poses, k, mode="pose", sigma_d=0.5, sigma_yaw=0.5, exclude=()):
        cand = [i for i in range(len(poses)) if i not in set(exclude)]
        if k <= 0 or not cand:
            return []
        if mode == "recent":
            return sorted(cand)[-k:][::-1]
        scored = []
        for i in cand:
            dx, dy = poses[i][0] - cur_pose[0], poses[i][1] - cur_pose[1]
            dyaw = _wrap(poses[i][2] - cur_pose[2])
            scored.append((-(math.hypot(dx, dy) / sigma_d) - abs(dyaw) / sigma_yaw, i))
        scored.sort(key=lambda t: (-t[0], -t[1]))
        return [i for _, i in scored[:k]]


def overlap(pose_seq, k=2):
    """Fraction of query steps at which both rules return the same set (and the same order)."""
    same_set = same_order = n = 0
    for t in range(1, len(pose_seq)):
        entries = pose_seq[:t]
        a = retrieve(pose_seq[t], entries, k, "pose"); b = retrieve(pose_seq[t], entries, k, "recent")
        n += 1; same_set += set(a) == set(b); same_order += a == b
    return same_set, same_order, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/1x_nav"); ap.add_argument("--max_train_trajs", type=int, default=200)
    ap.add_argument("--episodes", nargs="*", default=[], help="closed-loop eval dirs (per-episode eval_result.json)")
    ap.add_argument("--k", type=int, default=2)
    args = ap.parse_args()
    train, dev, test = split(args.data, max_train=args.max_train_trajs)
    for name, names in (("train200", train), ("test", test)):
        s = o = n = 0
        for tr in names:
            d = load_pkl(os.path.join(args.data, tr, "traj_data.pkl"))
            seq = [(float(p[0]), float(p[1]), float(y)) for p, y in zip(np.asarray(d["position"]), np.asarray(d["yaw"]).ravel())]
            a, b, c = overlap(seq, args.k); s += a; o += b; n += c
        print(f"{name} (reference poses, {len(names)} trajectories): same set {s}/{n} = {s/n:.3f}, same order {o}/{n} = {o/n:.3f}")
    for path in args.episodes:
        s = o = n = 0; m = 0
        for ep in sorted(os.listdir(path)):
            f = os.path.join(path, ep, "eval_result.json")
            if not os.path.isfile(f):
                continue
            r = json.load(open(f)); acts = [parse_action(x) for x in r["trajectory_actions"]]
            seq, pose = [(0.0, 0.0, 0.0)], np.zeros(3)
            for dx, dy, dyaw in acts:
                pose = pose + np.array([dx, dy, dyaw]); seq.append(tuple(pose.tolist()))
            a, b, c = overlap(seq, args.k); s += a; o += b; n += c; m += 1
        print(f"{path} (predicted poses, {m} episodes): same set {s}/{n} = {s/n:.3f}, same order {o}/{n} = {o/n:.3f}")


if __name__ == "__main__":
    main()
