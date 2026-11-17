"""Build a UniWM-format navigation dataset from the raw 1X world_model_raw_data (train_v2.0_raw) shards.

Keeps only frames whose base is moving (|linear velocity| or |angular velocity| above a threshold), cuts the
moving runs of each contiguous segment into trajectories sampled at `--hz`, integrates the velocities into a
planar pose (x, y, yaw) in the trajectory's start frame, and writes `traj_xxxx/{k}.jpg` + `traj_data.pkl` with
`position (N,2)`, `yaw (N,)`, `delta (N-1,3)` (same layout as the wheeled datasets, so scripts/navigation.py
loads it unchanged) plus the 25-dim `state (N,25)` of every kept frame.

Example:
  python tools/build_1x_nav.py --raw_dir external/1x_raw/train_v2.0_raw --out_dir data/1x_nav --shards 0-9 --hz 5
"""
import argparse, glob, json, os, pickle, sys
import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

LIN, ANG = 23, 24  # state indices (README "Index-to-State Mapping")
HIP_YAW = 0        # torso yaw: the head camera turns with it, so it is folded into the observed heading


def parse_shards(spec):
    out = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-"); out += list(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def moving_runs(mask, min_len, max_gap):
    """Contiguous runs of True in `mask`, bridging gaps up to `max_gap` frames, keeping runs >= min_len."""
    runs, start, gap = [], None, 0
    for i, m in enumerate(mask):
        if m:
            if start is None:
                start = i
            gap = 0
        elif start is not None:
            gap += 1
            if gap > max_gap:
                end = i - gap
                if end - start + 1 >= min_len:
                    runs.append((start, end))
                start, gap = None, 0
    if start is not None:
        end = len(mask) - 1 - gap
        if end - start + 1 >= min_len:
            runs.append((start, end))
    return runs


def integrate_pose(states, hz_raw):
    """Integrate planar pose from per-frame (v, w) at the raw rate; returns (N,3) [x, y, yaw]."""
    dt = 1.0 / hz_raw
    v, w = states[:, LIN].astype(np.float64), states[:, ANG].astype(np.float64)
    pose = np.zeros((len(states), 3))
    for t in range(1, len(states)):
        yaw = pose[t - 1, 2] + w[t - 1] * dt
        pose[t, 0] = pose[t - 1, 0] + v[t - 1] * np.cos(pose[t - 1, 2]) * dt
        pose[t, 1] = pose[t - 1, 1] + v[t - 1] * np.sin(pose[t - 1, 2]) * dt
        pose[t, 2] = yaw
    return pose


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--shards", default="0-9")
    ap.add_argument("--hz", type=float, default=5.0, help="sampling rate of the output trajectories")
    ap.add_argument("--min_len", type=int, default=15, help="min frames per trajectory (at --hz)")
    ap.add_argument("--max_len", type=int, default=25, help="max frames per trajectory (long runs are split)")
    ap.add_argument("--vel_thr", type=float, default=0.05, help="|v| or |w| above this counts as moving")
    ap.add_argument("--max_gap", type=int, default=6, help="raw frames of non-motion tolerated inside a run")
    ap.add_argument("--img_size", type=int, default=448)
    ap.add_argument("--jpeg_quality", type=int, default=92)
    ap.add_argument("--max_traj", type=int, default=0, help="stop after this many trajectories (0 = all)")
    ap.add_argument("--no_camera_yaw", action="store_true",
                    help="by default yaw = integrated base yaw + (hip_yaw - hip_yaw[0]); set to use the base yaw only")
    args = ap.parse_args()

    meta = json.load(open(os.path.join(args.raw_dir, "metadata.json")))
    hz_raw = float(meta.get("hz", 30))
    stride = int(round(hz_raw / args.hz))
    os.makedirs(args.out_dir, exist_ok=True)
    n_traj = 0
    summary = []
    for shard in parse_shards(args.shards):
        mp4 = os.path.join(args.raw_dir, "videos", f"video_{shard}.mp4")
        st_path = os.path.join(args.raw_dir, "robot_states", f"states_{shard}.bin")
        seg_path = os.path.join(args.raw_dir, "segment_indices", f"segment_idx_{shard}.bin")
        if not (os.path.exists(mp4) and os.path.exists(st_path) and os.path.exists(seg_path)):
            print(f"[skip] shard {shard}: files missing"); continue
        states = np.fromfile(st_path, dtype=np.float32).reshape(-1, 25)
        seg = np.fromfile(seg_path, dtype=np.int32)
        n = min(len(states), len(seg))
        states, seg = states[:n], seg[:n]
        moving = (np.abs(states[:, LIN]) > args.vel_thr) | (np.abs(states[:, ANG]) > args.vel_thr)

        # runs of motion inside each contiguous segment
        plan = []  # (raw_start, raw_end) at raw rate, already cut to trajectory length
        bounds = np.flatnonzero(np.diff(seg) != 0) + 1
        seg_starts = np.concatenate([[0], bounds]); seg_ends = np.concatenate([bounds, [n]])
        min_raw = (args.min_len - 1) * stride + 1
        for s0, s1 in zip(seg_starts, seg_ends):
            for r0, r1 in moving_runs(moving[s0:s1], min_raw, args.max_gap):
                r0 += s0; r1 += s0
                # split long runs into chunks of at most max_len sampled frames
                total_samples = (r1 - r0) // stride + 1
                pos = r0
                while total_samples >= args.min_len:
                    take = min(args.max_len, total_samples)
                    plan.append((pos, pos + (take - 1) * stride))
                    pos += take * stride
                    total_samples -= take
        if not plan:
            print(f"[shard {shard}] no moving runs >= {args.min_len} samples"); continue
        need = {}
        for ti, (a, b) in enumerate(plan):
            for k, fi in enumerate(range(a, b + 1, stride)):
                need.setdefault(fi, []).append((ti, k))
        print(f"[shard {shard}] frames={n} moving={moving.mean():.1%} trajectories={len(plan)} frames_to_decode={len(need)}", flush=True)

        # sequential decode, write only the needed frames
        cap = cv2.VideoCapture(mp4)
        if not cap.isOpened():
            print(f"[shard {shard}] cannot open video"); continue
        traj_dirs = {}
        fi, last_needed = 0, max(need)
        while fi <= last_needed:
            ok, frame = cap.read()
            if not ok:
                break
            if fi in need:
                if frame.shape[0] != args.img_size:
                    frame = cv2.resize(frame, (args.img_size, args.img_size), interpolation=cv2.INTER_AREA)
                for ti, k in need[fi]:
                    d = traj_dirs.get(ti)
                    if d is None:
                        d = os.path.join(args.out_dir, f"traj_{n_traj + ti:05d}"); os.makedirs(d, exist_ok=True); traj_dirs[ti] = d
                    cv2.imwrite(os.path.join(d, f"{k}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality])
            fi += 1
        cap.release()
        decoded = fi
        for ti, (a, b) in enumerate(plan):
            idx = np.arange(a, b + 1, stride)
            if idx[-1] >= decoded:
                continue
            raw = states[a:b + 1]
            pose_raw = integrate_pose(raw, hz_raw)          # base motion (position follows the base heading)
            pose = pose_raw[idx - a].copy()
            if not args.no_camera_yaw:                       # observed heading = base yaw + torso (hip) yaw change
                pose[:, 2] += raw[idx - a, HIP_YAW] - raw[0, HIP_YAW]
            delta = np.diff(pose, axis=0)
            d = traj_dirs[ti]
            with open(os.path.join(d, "traj_data.pkl"), "wb") as f:
                pickle.dump({
                    "position": pose[:, :2].astype(np.float32),
                    "yaw": pose[:, 2].astype(np.float32),
                    "delta": delta.astype(np.float32),
                    "state": states[idx].astype(np.float32),
                    "lin_vel": states[idx, LIN].astype(np.float32),
                    "ang_vel": states[idx, ANG].astype(np.float32),
                    "base_yaw": pose_raw[idx - a, 2].astype(np.float32),
                    "hip_yaw": states[idx, HIP_YAW].astype(np.float32),
                    "raw_frame_idx": idx.astype(np.int64), "shard": int(shard), "segment": int(seg[a]), "hz": args.hz,
                }, f, protocol=4)
            summary.append((len(idx), np.abs(delta[:, :2]).sum(1).mean(), np.abs(delta[:, 2]).mean()))
        n_traj += len(plan)
        if args.max_traj and n_traj >= args.max_traj:
            break
    if summary:
        L = np.array([s[0] for s in summary]); dxy = np.array([s[1] for s in summary]); dyaw = np.array([s[2] for s in summary])
        print(f"[done] trajectories={len(summary)} frames={int(L.sum())} len min/med/max={L.min()}/{int(np.median(L))}/{L.max()} "
              f"mean|dpos| per step={dxy.mean():.3f} mean|dyaw| per step={dyaw.mean():.3f} rad")
    else:
        print("[done] nothing written")


if __name__ == "__main__":
    main()
