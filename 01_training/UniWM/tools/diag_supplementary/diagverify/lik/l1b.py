"""L1b: turning-minus-straight contrast for negation (joint trajectory bootstrap); pooled over checkpoints; bounds."""
import sys; sys.path.insert(0, __file__.rsplit("/", 1)[0])
from likcommon import *

rng = np.random.default_rng(0)
allneg = {}
for run in RUNS:
    R = [r for r in load_run(run) if not is_goal(r)]
    rows = [(r["traj"], 1 if turning(r) else (0 if straight(r) else -1), nll(r, "negated") - nll(r, "recorded")) for r in R]
    allneg[run] = rows
    trajs = sorted(set(t for t, _, _ in rows)); by = {t: [x for x in rows if x[0] == t] for t in trajs}
    def did(sample):
        T = [d for t in sample for (_, c, d) in by[t] if c == 1]; S = [d for t in sample for (_, c, d) in by[t] if c == 0]
        return np.mean(T) - np.mean(S)
    est = did(trajs); B = [did(rng.choice(trajs, len(trajs))) for _ in range(3000)]
    lo, hi = np.percentile(B, [2.5, 97.5])
    T = np.array([d for _, c, d in rows if c == 1])
    print(f"{run:7s} turning-minus-straight negated ΔNLL {est:+.3f} [{lo:+.3f},{hi:+.3f}]; turning upper 95% bound (boot) -> LR<=exp(hi)")

# pooled over checkpoints on turning frames, with and without stopft; trajectory bootstrap
for label, runs in (("all six", RUNS), ("five (drop stopft)", [r for r in RUNS if r != "stopft"]), ("de models", ["de50", "de100", "de200", "de459"])):
    per = {}
    for run in runs:
        for (t, c, d), r in zip(allneg[run], [r for r in load_run(run) if not is_goal(r)]):
            if c == 1:
                per.setdefault((t, r["k"]), []).append(d)
    keys = sorted(per); v = [np.mean(per[k]) for k in keys]; g = [k[0] for k in keys]
    m, lo, hi, _ = cluster_boot(v, g)
    print(f"pooled {label}: {m:+.3f} [{lo:+.3f},{hi:+.3f}] n={len(v)} frames")
