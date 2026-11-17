import glob, json, os, math
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
RAW = os.path.join(REPO, "output", "diag", "raw")
RUNS = ["base", "stopft", "de50", "de100", "de200", "de459"]
N_IMG = 784


def load_run(run):
    d = glob.glob(os.path.join(RAW, f"{run}_likelihood*", "action_likelihood_results"))
    assert len(d) == 1, d
    recs = []
    for f in sorted(glob.glob(os.path.join(d[0], "likelihood_rank*.json"))):
        recs += json.load(open(f))
    return recs


def text(r, c):
    s = r["scores"][c]
    return s.get("action_text") or r["scores"][s["same_text_as"]]["action_text"]


def nll(r, c):
    return r["scores"][c]["nll_img"]


def act(r, c):
    return r["scores"][c]["action"]


def is_goal(r):
    return r.get("target_is_goal", r["k"] + 1 == r["n"] - 1)


def turning(r):
    return abs(r["dyaw_k"]) >= 0.05


def straight(r):
    return abs(r["dyaw_k"]) < 0.02


def midturn(r):
    return (r["dyaw_km1"] is not None and abs(r["dyaw_k"]) >= 0.05 and abs(r["dyaw_km1"]) >= 0.05
            and np.sign(r["dyaw_k"]) == np.sign(r["dyaw_km1"]))


def cluster_boot(values, groups, n_boot=4000, seed=0, stat=np.mean):
    v = np.asarray(values, float); g = np.asarray(groups)
    uniq = np.unique(g); idx = {u: np.where(g == u)[0] for u in uniq}; rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, len(uniq))
        boots.append(stat(np.concatenate([v[idx[u]] for u in pick])))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(stat(v)), float(lo), float(hi), np.asarray(boots)


T975 = {10: 2.228, 15: 2.131, 18: 2.101, 19: 2.093, 20: 2.086, 21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060, 26: 2.056}


def cluster_t(values, groups):
    v = np.asarray(values, float); g = np.asarray(groups)
    uniq = np.unique(g); G = len(uniq); N = len(v); m = v.mean()
    s = np.array([(v[g == u] - m).sum() for u in uniq])
    var = (G / (G - 1)) * (s ** 2).sum() / N ** 2
    se = math.sqrt(var); t = T975.get(G - 1, 2.06)
    return m, m - t * se, m + t * se, se


def icc_deff(values, groups):
    v = np.asarray(values, float); N = len(v)
    se_c = cluster_t(values, groups)[3]
    se_iid = v.std(ddof=1) / math.sqrt(N)
    return (se_c / se_iid) ** 2


def split_best(r, C):
    v = {c: nll(r, c) for c in C}; m = min(v.values())
    win_txt = {text(r, c) for c in C if v[c] == m}
    w = [c for c in C if text(r, c) in win_txt]
    return {c: 1.0 / len(w) for c in w}
