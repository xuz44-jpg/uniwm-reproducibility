"""Independent re-analysis of the intervention diagnostic (numpy only)."""
import glob, json, math, os, sys
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
TOL = (0.1, 0.1, 0.2)
TURN = 0.05


def load(run):
    d = glob.glob(f"{ROOT}/{run}_intervention*/action_intervention_results")[0]
    R = []
    for f in sorted(glob.glob(d + "/intervention_rank*.json")):
        R += json.load(open(f))
    return R


def b(v, w=0.01):
    return round((1 if v >= 0 else -1) * math.floor(abs(v) / w + 1e-9) * w, 4)


def acc(pred, ref, stop, tol=TOL):
    if stop or pred == "stop":
        return float(bool(stop) and pred == "stop")
    return float(pred is not None and all(abs(p - b(r)) <= t + 1e-9 for p, r, t in zip(pred, ref, tol)))


def cboot(vals, grp, n=4000, seed=0, stat=np.mean):
    v = np.asarray(vals, float); g = np.asarray(grp)
    u = np.unique(g); idx = {x: np.where(g == x)[0] for x in u}; rng = np.random.default_rng(seed); bs = []
    for _ in range(n):
        pk = rng.choice(u, len(u)); bs.append(np.concatenate([v[idx[x]] for x in pk]).mean())
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return v.mean(), lo, hi, len(v), len(u)


def f(t, p=3):
    m, lo, hi, n, k = t
    return f"{m:+.{p}f} [{lo:+.{p}f},{hi:+.{p}f}] (n={n}, {k} traj)"


def rows(R, key):
    out = []
    for r in R:
        for st in r["steps"]:
            out.append(dict(traj=r["traj"], n=r["n"], s=st["s"], rec=st["recorded"], stop=st[key],
                            final=st["s"] == r["n"] - 1, real=st["real"], vq=st["vq"], gen=st["gen"]))
    return out


for run, key in (("base", "stop_label_released"), ("stopft", "stop_label_dist")):
    R = load(run); S = rows(R, key)
    print("=" * 30, run, key, "steps", len(S), "stop-labelled", sum(x["stop"] for x in S),
          "of which final", sum(x["stop"] and x["final"] for x in S))
    # 0) sanity: gen at s=0 is VQ(start)=VQ(real frame 0): greedy should equal vq greedy
    s0 = [x for x in S if x["s"] == 0]
    print("s=0 gen greedy == vq greedy:", sum(x["gen"][0] == x["vq"][0] for x in s0), "/", len(s0),
          "; gen samples == vq samples:", sum(x["gen"] == x["vq"] for x in s0))
    # 1) headline replication
    for nm in ("real", "vq", "gen"):
        a = [acc(x[nm][0], x["rec"], x["stop"]) for x in S]
        print(f"  acc {nm}: {f(cboot(a, [x['traj'] for x in S]), 3)}")
    for oth in ("gen", "vq"):
        d = [acc(x["real"][0], x["rec"], x["stop"]) - acc(x[oth][0], x["rec"], x["stop"]) for x in S]
        print(f"  paired real-{oth} all: {f(cboot(d, [x['traj'] for x in S]))}")
        for lab, cond in (("move-labelled", lambda x: not x["stop"]), ("stop-labelled", lambda x: x["stop"]),
                          ("non-final", lambda x: not x["final"]), ("s>=1 move", lambda x: not x["stop"] and x["s"] >= 1)):
            T = [x for x in S if cond(x)]
            d = [acc(x["real"][0], x["rec"], x["stop"]) - acc(x[oth][0], x["rec"], x["stop"]) for x in T]
            print(f"    {lab:14s}: {f(cboot(d, [x['traj'] for x in T]))}; contribution to overall mean {np.sum(d) / len(S):+.3f}")
    # 2) depth bins: all vs move-only, paired loss per bin
    print("  depth bins (greedy)  all: real/vq/gen | move-only: real/vq/gen, paired real-gen [CI] | #stop in bin")
    for bb in range(5):
        T = [x for x in S if min(x["s"] // 5, 4) == bb]; M = [x for x in T if not x["stop"]]
        al = [np.mean([acc(x[nm][0], x["rec"], x["stop"]) for x in T]) for nm in ("real", "vq", "gen")]
        mv = [np.mean([acc(x[nm][0], x["rec"], x["stop"]) for x in M]) for nm in ("real", "vq", "gen")]
        dm = [acc(x["real"][0], x["rec"], False) - acc(x["gen"][0], x["rec"], False) for x in M]
        da = [acc(x["real"][0], x["rec"], x["stop"]) - acc(x["gen"][0], x["rec"], x["stop"]) for x in T]
        print(f"    {5*bb:2d}-{5*bb+4:2d}: n={len(T):3d} {al[0]:.2f}/{al[1]:.2f}/{al[2]:.2f} paired {f(cboot(da,[x['traj'] for x in T]),3)} | move n={len(M):3d} "
              f"{mv[0]:.2f}/{mv[1]:.2f}/{mv[2]:.2f} paired {f(cboot(dm,[x['traj'] for x in M]),3)} | stop {len(T)-len(M)}")
    # 3) balanced panel: only n=25 trajectories (so bin composition is fixed)
    P = [x for x in S if x["n"] == 25]
    print(f"  balanced panel (n==25 trajectories: {len(set(x['traj'] for x in P))}), move-only paired real-gen by bin:")
    for bb in range(5):
        M = [x for x in P if min(x["s"] // 5, 4) == bb and not x["stop"]]
        dm = [acc(x["real"][0], x["rec"], False) - acc(x["gen"][0], x["rec"], False) for x in M]
        rg = [np.mean([acc(x[nm][0], x["rec"], False) for x in M]) for nm in ("real", "gen")]
        print(f"    {5*bb:2d}-{5*bb+4:2d}: real {rg[0]:.2f} gen {rg[1]:.2f} paired {f(cboot(dm,[x['traj'] for x in M]))}")
    # 4) slope of paired loss on s (move-only), cluster bootstrap
    M = [x for x in S if not x["stop"]]
    sv = np.array([x["s"] for x in M], float); dv = np.array([acc(x["real"][0], x["rec"], False) - acc(x["gen"][0], x["rec"], False) for x in M])
    g = np.array([x["traj"] for x in M]); u = np.unique(g); idx = {t: np.where(g == t)[0] for t in u}; rng = np.random.default_rng(1)
    slope = np.polyfit(sv, dv, 1)[0]; bs = []
    for _ in range(4000):
        pk = np.concatenate([idx[t] for t in rng.choice(u, len(u))]); bs.append(np.polyfit(sv[pk], dv[pk], 1)[0])
    print(f"  move-only paired loss slope per step: {slope:+.4f} [{np.percentile(bs,2.5):+.4f},{np.percentile(bs,97.5):+.4f}] -> per 20 steps {20*slope:+.3f}")
    Sall = S
    sv = np.array([x["s"] for x in Sall], float); dv = np.array([acc(x["real"][0], x["rec"], x["stop"]) - acc(x["gen"][0], x["rec"], x["stop"]) for x in Sall])
    g = np.array([x["traj"] for x in Sall]); u = np.unique(g); idx = {t: np.where(g == t)[0] for t in u}; bs = []
    slope = np.polyfit(sv, dv, 1)[0]
    for _ in range(4000):
        pk = np.concatenate([idx[t] for t in rng.choice(u, len(u))]); bs.append(np.polyfit(sv[pk], dv[pk], 1)[0])
    print(f"  all-steps paired loss slope per step: {slope:+.4f} [{np.percentile(bs,2.5):+.4f},{np.percentile(bs,97.5):+.4f}]")
    # 5) failure mode on move-labelled steps: which component fails / predicts stop / None
    for nm in ("real", "gen"):
        fails = {"stop": 0, "none": 0, "dx": 0, "dy": 0, "dyaw": 0}
        for x in M:
            p = x[nm][0]
            if p == "stop": fails["stop"] += 1; continue
            if p is None: fails["none"] += 1; continue
            for c, i in (("dx", 0), ("dy", 1), ("dyaw", 2)):
                if abs(p[i] - b(x["rec"][i])) > TOL[i] + 1e-9: fails[c] += 1
        print(f"  {nm} move-step failures (n={len(M)}):", fails)
    # 6) turn sign
    Tn = [x for x in S if not x["stop"] and x["rec"] is not None and abs(x["rec"][2]) >= TURN]
    ok = lambda p, r: float(p not in (None, "stop") and np.sign(p[2]) == np.sign(r[2]) and abs(p[2]) >= TURN / 2)
    for nm in ("real", "vq", "gen"):
        print(f"  turn sign {nm}: {f(cboot([ok(x[nm][0], x['rec']) for x in Tn], [x['traj'] for x in Tn]),2)}")
    d = [ok(x["real"][0], x["rec"]) - ok(x["gen"][0], x["rec"]) for x in Tn]
    print(f"  turn sign paired real-gen: {f(cboot(d,[x['traj'] for x in Tn]))}")
    # turn-sign sign-only (drop the |dyaw|>=0.025 magnitude requirement) and 'predicts straight' share
    ok2 = lambda p, r: float(p not in (None, "stop") and np.sign(p[2]) == np.sign(r[2]) and p[2] != 0)
    for nm in ("real", "gen"):
        w = [x[nm][0] for x in Tn]
        wrong_sign = np.mean([p not in (None, "stop") and p[2] != 0 and np.sign(p[2]) != np.sign(x["rec"][2]) for p, x in zip(w, Tn)])
        small = np.mean([p not in (None, "stop") and abs(p[2]) < TURN / 2 for p in w])
        print(f"   {nm}: sign-only correct {np.mean([ok2(p, x['rec']) for p, x in zip(w, Tn)]):.3f}; opposite sign {wrong_sign:.3f}; |pred dyaw|<0.025 {small:.3f}")
    print("  turn-sign paired loss by depth bin (n, real, gen):", [(bb, len(T := [x for x in Tn if min(x['s']//5,4)==bb]),
          round(np.mean([ok(x['real'][0], x['rec']) for x in T]),2), round(np.mean([ok(x['gen'][0], x['rec']) for x in T]),2)) for bb in range(5)])
    # 7) Stop recall splits
    St = [x for x in S if x["stop"]]
    for lab, T in (("all stop-labelled", St), ("final frame", [x for x in St if x["final"]]), ("non-final", [x for x in St if not x["final"]])):
        if not T: continue
        for nm in ("real", "vq", "gen"):
            v = [float(x[nm][0] == "stop") for x in T]
            print(f"  Stop recall {lab:18s} {nm}: {f(cboot(v, [x['traj'] for x in T]),2)}; sample-level {np.mean([p=='stop' for x in T for p in x[nm][1:]]):.2f}")
    # gen Stop predictions anywhere
    gs = [(x["traj"], x["s"], x["n"], x["stop"]) for x in S if x["gen"][0] == "stop"]
    print("  gen greedy Stop predictions:", gs)
    gsa = [(x["traj"], x["s"], x["n"], sum(p == "stop" for p in x["gen"][1:])) for x in S if any(p == "stop" for p in x["gen"][1:])]
    print("  gen sampled Stop anywhere:", gsa)
    # 8) sampled-action accuracy (5 samples) paired
    d = [np.mean([acc(p, x["rec"], x["stop"]) for p in x["real"][1:]]) - np.mean([acc(p, x["rec"], x["stop"]) for p in x["gen"][1:]]) for x in S]
    print(f"  sampled-action paired real-gen all: {f(cboot(d,[x['traj'] for x in S]))}")
    d = [np.mean([acc(p, x["rec"], False) for p in x["real"][1:]]) - np.mean([acc(p, x["rec"], False) for p in x["gen"][1:]]) for x in M]
    print(f"  sampled-action paired real-gen move: {f(cboot(d,[x['traj'] for x in M]))}")
    # 9) tolerance sensitivity (move-only)
    for tol in ((0.1, 0.1, 0.2), (0.05, 0.05, 0.1), (0.03, 0.03, 0.05), (1e9, 1e9, 0.05)):
        ar = [acc(x["real"][0], x["rec"], False, tol) for x in M]; ag = [acc(x["gen"][0], x["rec"], False, tol) for x in M]
        d = [a - c for a, c in zip(ar, ag)]
        print(f"  tol {tol}: move real {np.mean(ar):.3f} gen {np.mean(ag):.3f} paired {f(cboot(d,[x['traj'] for x in M]))}")
    # 10) trivial baselines on move steps
    recs = np.array([x["rec"] for x in M])
    med = [b(v) for v in np.median(recs, 0)]
    const = [acc(med, x["rec"], False) for x in M]
    print(f"  constant median action {med}: move accuracy {np.mean(const):.3f}; straight-only [0.1,0,0]: {np.mean([acc([0.1,0.0,0.0], x['rec'], False) for x in M]):.3f}")
    # predict previous recorded action (persistence baseline)
    prev = {}
    for r in R:
        for st in r["steps"]:
            prev[(r["traj"], st["s"])] = st["recorded"]
    pers = [acc([b(v) for v in prev[(x["traj"], x["s"] - 1)]], x["rec"], False) for x in M if x["s"] >= 1]
    print(f"  persistence baseline (previous recorded action) move accuracy {np.mean(pers):.3f} (n={len(pers)})")
    # dyaw abs error move steps real vs gen paired
    e = [(abs(x["real"][0][2] - x["rec"][2]) if isinstance(x["real"][0], list) else np.nan,
          abs(x["gen"][0][2] - x["rec"][2]) if isinstance(x["gen"][0], list) else np.nan) for x in M]
    e = [(a, c, x["traj"]) for (a, c), x in zip(e, M) if not (np.isnan(a) or np.isnan(c))]
    print(f"  |dyaw err| paired gen-real on move steps: {f(cboot([c-a for a,c,_ in e],[t for *_,t in e]),4)}")
