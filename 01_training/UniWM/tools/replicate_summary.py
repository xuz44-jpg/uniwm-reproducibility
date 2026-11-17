"""Mean and between-run spread of closed-loop metrics over replicate evaluation runs.

Each configuration is given as name=dir1,dir2,...; the script prints, per configuration, the mean over runs of the
24-episode means (distance, closest, ATE, RPE, SR(3 steps), SR(0.5 u), Stop count) with the between-run standard
deviation, plus paired comparisons between configurations using the per-episode mean over runs.

usage: python tools/replicate_summary.py --cfg base=output/evals/1x200_evaltask25,output/evals/base_ext48_evaltask25,output/evals/base_r3_evaltask25 ...
"""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from closed_loop_breakdown import summarise

STEP = 0.096


def load(path, names=None):
    rows = {}
    for n in sorted(os.listdir(path)):
        f = os.path.join(path, n, "eval_result.json")
        if os.path.isfile(f) and (names is None or n in names):
            rows[n] = summarise(None, json.load(open(f)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", nargs="+", required=True, help="name=dir1,dir2,...")
    ap.add_argument("--episodes", default="output/evals/1x200_evaltask25", help="directory defining the episode set (default: the 24 test episodes)")
    ap.add_argument("--pairs", nargs="*", default=[], help="a:b comparisons on per-episode means")
    args = ap.parse_args()
    names = sorted(load(args.episodes)); cfgs = {}
    for c in args.cfg:
        name, dirs = c.split("=", 1); runs = [load(d, names) for d in dirs.split(",")]
        runs = [r for r in runs if len(r) == len(names)]
        cfgs[name] = runs
    keys = [("distance", "dist"), ("closest", "closest"), ("ate", "ATE"), ("rpe", "RPE")]
    print(f"{'config':14s} runs  " + "  ".join(f"{lab:>14s}" for _, lab in keys) + "  SR(3st)  SR(0.5u)   Stop")
    for name, runs in cfgs.items():
        per_run = {k: [np.mean([r[n][k] for n in names]) for r in runs] for k, _ in keys}
        sr3 = [np.mean([r[n]["distance"] < 3 * STEP for n in names]) for r in runs]
        sr5 = [np.mean([r[n]["distance"] < 0.5 for n in names]) for r in runs]
        stops = [sum(r[n]["stop"] for n in names) for r in runs]
        cells = "  ".join(f"{np.mean(per_run[k]):6.3f} ±{np.std(per_run[k], ddof=1) if len(runs) > 1 else 0:.3f}" for k, _ in keys)
        print(f"{name:14s} {len(runs):3d}   {cells}   {np.mean(sr3):.2f}     {np.mean(sr5):.2f}   {np.mean(stops):4.1f}")
    rng = np.random.default_rng(0)
    for pair in args.pairs:
        a, b = pair.split(":")
        for k in ("distance", "ate"):
            ma = np.array([np.mean([r[n][k] for r in cfgs[a]]) for n in names]); mb = np.array([np.mean([r[n][k] for r in cfgs[b]]) for n in names])
            d = ma - mb; boots = [rng.choice(d, len(d)).mean() for _ in range(10000)]
            # between-run noise of the difference: sd of (run-mean a - run-mean b) over run pairs
            ra = [np.mean([r[n][k] for n in names]) for r in cfgs[a]]; rb = [np.mean([r[n][k] for n in names]) for r in cfgs[b]]
            print(f"{a} - {b} {k}: {d.mean():+.3f}  episode-bootstrap CI [{np.percentile(boots, 2.5):+.3f}, {np.percentile(boots, 97.5):+.3f}]  "
                  f"run means a={np.round(ra, 3).tolist()} b={np.round(rb, 3).tolist()}")


if __name__ == "__main__":
    main()
