"""Independent re-analysis of the teacher-forced likelihood diagnostic (numpy only)."""
import glob, json, os, sys
import numpy as np

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
ROOT = os.path.join(REPO, "output", "diag", "raw")
RUNS = ["base", "stopft", "de50", "de100", "de200", "de459"]
N_IMG = 784


def load(run):
    fs = sorted(glob.glob(f"{ROOT}/{run}_likelihood*/action_likelihood_results/likelihood_rank*.json"))
    R = []
    for f in fs:
        R += json.load(open(f))
    return R


def txt(r, c):
    s = r["scores"][c]
    return s.get("action_text") or r["scores"][s["same_text_as"]]["action_text"]


def nll(r, c):
    return r["scores"][c]["nll_img"]


def ccI(v, g, n_boot=4000, seed=0):
    v = np.asarray(v, float); g = np.asarray(g)
    if len(v) == 0:
        return (np.nan,) * 3
    u = np.unique(g); idx = {x: np.where(g == x)[0] for x in u}; rng = np.random.default_rng(seed); b = []
    for _ in range(n_boot):
        pick = rng.choice(u, len(u)); b.append(np.concatenate([v[idx[x]] for x in pick]).mean())
    lo, hi = np.percentile(b, [2.5, 97.5])
    return v.mean(), lo, hi


def f3(t):
    return f"{t[0]:+.3f} [{t[1]:+.3f},{t[2]:+.3f}]"


DATA = {run: load(run) for run in RUNS}
per_frame = {}
for run in RUNS:
    R_all = DATA[run]
    G = [r for r in R_all if r["target_is_goal"]]
    R = [r for r in R_all if not r["target_is_goal"]]
    print(f"\n=========== {run}: {len(R_all)} frames, {len(G)} goal-target, n_tgt={R_all[0]['scores']['recorded']['n_tgt']}")
    tot = np.array([nll(r, 'recorded') for r in R]); gt = np.array([nll(r, 'recorded') for r in G])
    print(f"  per-token NLL non-goal {tot.mean() / N_IMG:.3f} (frame total mean {tot.mean():.0f}, range {tot.min():.0f}-{tot.max():.0f}); goal-target {gt.mean() / N_IMG:.3f}")
    T = [r for r in R if abs(r["dyaw_k"]) >= 0.05]
    # --- negation on turning frames
    d = np.array([nll(r, 'negated') - nll(r, 'recorded') for r in T]); g = [r["traj"] for r in T]
    per_frame[run] = {(r["traj"], r["k"]): x for r, x in zip(T, d)}
    print(f"  turning n={len(T)}: dNLL(neg-rec) mean {f3(ccI(d, g))}; mean|d| {np.abs(d).mean():.3f}; median|d| {np.median(np.abs(d)):.3f}; "
          f"SD {d.std(ddof=1):.3f}; share>0 {np.mean(d > 0):.3f}; exact zeros {int(np.sum(d == 0))}")
    # sign split: alignment vs left-text bias
    L = [x for r, x in zip(T, d) if r["dyaw_k"] > 0]; Rr = [x for r, x in zip(T, d) if r["dyaw_k"] < 0]
    print(f"    true-left n={len(L)} mean {np.mean(L):+.3f}; true-right n={len(Rr)} mean {np.mean(Rr):+.3f}; "
          f"alignment (avg) {(np.mean(L) + np.mean(Rr)) / 2:+.3f}; left-text preference (half-diff, + = right text worse) {(np.mean(L) - np.mean(Rr)) / 2:+.3f}")
    # magnitude bins
    for lo, hi in ((0.05, 0.10), (0.10, 0.15), (0.15, 9)):
        S = [(r, x) for r, x in zip(T, d) if lo <= abs(r["dyaw_k"]) < hi]
        if S:
            m = ccI([x for _, x in S], [r["traj"] for r, _ in S])
            print(f"    |dyaw| in [{lo:.2f},{hi:.2f}) n={len(S)} ({len(set(r['traj'] for r, _ in S))} trajs): mean {f3(m)}, mean|d| {np.mean([abs(x) for _, x in S]):.3f}, share>0 {np.mean([x > 0 for _, x in S]):.2f}")
    # null-swap floor: |dyaw|<0.01 -> pos_bin_00 <-> neg_bin_00 (both mean |dyaw|<0.01)
    Z = [r for r in R if abs(r["dyaw_k"]) < 0.01]
    dz = np.array([nll(r, 'negated') - nll(r, 'recorded') for r in Z])
    print(f"  null swap (|dyaw|<0.01, pos_00<->neg_00) n={len(Z)}: mean {dz.mean():+.3f}, mean|d| {np.abs(dz).mean():.3f}, SD {dz.std(ddof=1):.3f}")
    # --- zero candidate stratified
    for name, cond in (("straight |dyaw|<0.02", lambda r: abs(r["dyaw_k"]) < 0.02),
                       ("straight & dx>=0.10", lambda r: abs(r["dyaw_k"]) < 0.02 and r["scores"]["recorded"]["action"][0] >= 0.10),
                       ("turn 0.05-0.10", lambda r: 0.05 <= abs(r["dyaw_k"]) < 0.10),
                       ("turn >=0.10", lambda r: abs(r["dyaw_k"]) >= 0.10)):
        S = [r for r in R if cond(r) and txt(r, 'zero') != txt(r, 'recorded')]
        if S:
            dd = [nll(r, 'zero') - nll(r, 'recorded') for r in S]
            dx = np.mean([r["scores"]["recorded"]["action"][0] for r in S])
            print(f"  zero-rec [{name}] n={len(S)} mean dx {dx:.3f}: {f3(ccI(dd, [r['traj'] for r in S]))} share>0 {np.mean(np.array(dd) > 0):.2f}")
    # --- first-place shares among 4 candidates (ties split) on turning frames
    C4 = ("recorded", "negated", "zero", "other_traj")
    cred = {c: 0.0 for c in C4}; base = [r for r in T if all(c in r["scores"] for c in C4)]
    for r in base:
        v = {c: nll(r, c) for c in C4}; m = min(v.values()); wt = {txt(r, c) for c in C4 if v[c] == m}
        w = [c for c in C4 if txt(r, c) in wt]
        for c in w:
            cred[c] += 1 / len(w)
    print("  first place among 4 on turning frames: " + ", ".join(f"{c} {cred[c] / len(base):.3f}" for c in C4) + f" (n={len(base)})")
    C3 = ("recorded", "negated", "other_traj")
    cred3 = {c: 0.0 for c in C3}
    for r in base:
        v = {c: nll(r, c) for c in C3}; m = min(v.values()); w = [c for c in C3 if v[c] == m]
        for c in w:
            cred3[c] += 1 / len(w)
    print("  first place among 3 (no zero): " + ", ".join(f"{c} {cred3[c] / len(base):.3f}" for c in C3) + " (chance 0.333)")
    # --- lag test mid-turn
    M = [r for r in R if r["dyaw_km1"] is not None and abs(r["dyaw_k"]) >= 0.05 and abs(r["dyaw_km1"]) >= 0.05
         and np.sign(r["dyaw_k"]) == np.sign(r["dyaw_km1"])]
    CL = ("prev2", "prev1", "recorded", "next1")
    lead = [r for r in M if all(c in r["scores"] for c in CL)]
    cl = {c: 0.0 for c in CL}
    for r in lead:
        v = {c: nll(r, c) for c in CL}; m = min(v.values()); wt = {txt(r, c) for c in CL if v[c] == m}
        w = [c for c in CL if txt(r, c) in wt]
        for c in w:
            cl[c] += 1 / len(w)
    same_sign_next = np.mean([np.sign(r["scores"]["next1"]["action"][2]) == np.sign(r["dyaw_k"]) for r in lead])
    same_sign_p2 = np.mean([np.sign(r["scores"]["prev2"]["action"][2]) == np.sign(r["dyaw_k"]) for r in lead])
    dy_gap = np.mean([abs(r["scores"]["prev1"]["action"][2] - r["dyaw_k"]) for r in lead])
    print(f"  lag test mid-turn n={len(lead)}: " + ", ".join(f"{c} {cl[c] / len(lead):.3f}" for c in CL)
          + f"; next1 same heading sign {same_sign_next:.2f}, prev2 same sign {same_sign_p2:.2f}; mean |dyaw[k-1]-dyaw[k]| {dy_gap:.3f}")
    # --- copy control
    dg = [nll(r, 'negated') - nll(r, 'recorded') for r in G if txt(r, 'negated') != txt(r, 'recorded')]
    print(f"  copy control: goal-target per-token {gt.mean() / N_IMG:.3f} (min {gt.min() / N_IMG:.3f}, max {gt.max() / N_IMG:.3f}); dNLL neg-rec {np.mean(dg):+.3f} mean|d| {np.mean(np.abs(dg)):.3f} n={len(dg)}")

print("\n=========== cross-checkpoint correlation of per-frame dNLL(neg-rec) on turning frames")
for a in RUNS:
    row = []
    for b in RUNS:
        ks = sorted(set(per_frame[a]) & set(per_frame[b]))
        x = np.array([per_frame[a][k] for k in ks]); y = np.array([per_frame[b][k] for k in ks])
        row.append(f"{np.corrcoef(x, y)[0, 1]:+.2f}")
    print(f"  {a:7s} " + " ".join(row))
