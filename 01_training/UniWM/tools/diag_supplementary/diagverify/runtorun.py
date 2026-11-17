"""Per-episode run-to-run variance of final distance: sampled vs greedy vs 5-sample-median runs (local closed-loop JSON)."""
import json, glob, os, itertools, numpy as np
def _repo_root():
    """UniWM repository root: $UNIWM_ROOT if set, else the nearest ancestor of this file that holds output/diag/raw."""
    env = os.environ.get("UNIWM_ROOT")
    if env:
        return env
    d = os.path.dirname(os.path.abspath(__file__))
    while d != os.path.dirname(d):
        if os.path.isdir(os.path.join(d, "output", "diag", "raw")):
            return d
        d = os.path.dirname(d)
    raise SystemExit("cannot locate the UniWM repository (no output/diag/raw above this script); set UNIWM_ROOT")
REPO = _repo_root()
E = os.path.join(REPO, "output", "evals") + os.sep

def load(run):
    out = {}
    for f in glob.glob(E + run + '/*/eval_result.json'):
        r = json.load(open(f)); out[os.path.basename(os.path.dirname(f))] = r
    return out

groups = {
    'base sampled': ['1x200_evaltask25', 'base_r3_evaltask25'],
    'base greedy': ['base_greedy_r1_evaltask25', 'base_greedy_r2_evaltask25'],
    'base sc5-median': ['base_sc5_evaltask25', 'base_sc5_r2_evaltask25', 'base_sc5_r3_evaltask25'],
    'stop sampled': ['stopft_evaltask25', 'stopft_r3_evaltask25'],
    'stop greedy': ['stopft_greedy_r1_evaltask25', 'stopft_greedy_r2_evaltask25'],
    'stop sc5-median': ['stopft_sc5_evaltask25', 'stopft_sc5_r2_evaltask25', 'stopft_sc5_r3_evaltask25'],
    'mempose sampled': ['mempose_evaltask25', 'mempose_r3_evaltask25'],
    'memrecent sampled': ['memrecent_evaltask25', 'memrecent_r3_evaltask25'],
}
rng = np.random.default_rng(0)
for g, runs in groups.items():
    R = [load(r) for r in runs]
    eps = sorted(set.intersection(*[set(x) for x in R]))
    d = np.array([[x[e]['distance_to_goal'] for e in eps] for x in R])   # runs x episodes
    # unbiased per-episode variance across runs, averaged over episodes
    v_ep = d.var(axis=0, ddof=1)
    fp = np.array([[x[e]['final_position'] for e in eps] for x in R])
    v_pos = (fp.var(axis=0, ddof=1)).sum(-1)   # 2-D position variance per episode
    nsteps = np.array([[len(x[e]['trajectory_actions']) for e in eps] for x in R])
    # bootstrap over episodes
    b = [v_ep[rng.integers(0, len(eps), len(eps))].mean() for _ in range(4000)]
    lo, hi = np.percentile(b, [2.5, 97.5])
    print(f"{g:18s} runs={len(runs)} eps={len(eps)} run means {np.round(d.mean(1), 3)}  per-episode var of final dist {v_ep.mean():.4f} [{lo:.4f},{hi:.4f}]"
          f" -> implied SD of 24-episode mean {np.sqrt(v_ep.mean() / len(eps)):.3f}; per-episode 2-D final-position var {v_pos.mean():.4f}; mean steps {nsteps.mean():.1f}")

# direct-integration prediction of action sampling alone (within-image variance at depth 1, variance-decomp run)
Wdx, Wdy = 0.00041, 0.00055
for T in (20, 24.5):
    vpos = T * (Wdx + Wdy)
    vdist = T * (Wdx + Wdy) / 2   # projection on the goal direction, isotropic approximation
    print(f"T={T}: predicted 2-D final-position var from action sampling alone {vpos:.4f}; per-episode distance var ~{vdist:.4f}; SD of 24-episode mean ~{np.sqrt(vdist / 24):.3f}")
