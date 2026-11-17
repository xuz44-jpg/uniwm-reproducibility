"""Follow-up checks: I1 decision agreement, jackknife CIs, strict-tolerance depth trend, depth-1 comparison from the
variance run, turn-sign by depth with CIs."""
import glob, json
import numpy as np
from interv_verify import load, rows, acc, cboot, f, b, TURN, ROOT

ok = lambda p, r: float(p not in (None, "stop") and np.sign(p[2]) == np.sign(r[2]) and abs(p[2]) >= TURN / 2)


def jack(vals, grp):
    v = np.asarray(vals, float); g = np.asarray(grp); u = np.unique(g); k = len(u)
    th = np.array([v[g != t].mean() for t in u]); m = v.mean()
    se = np.sqrt((k - 1) / k * ((th - th.mean()) ** 2).sum())
    from math import sqrt
    tq = 2.069 if k == 24 else 2.074  # t_{0.975, 23} ~ 2.069, t_{0.975,22} ~ 2.074
    return m, m - tq * se, m + tq * se


def slope_ci(sv, dv, g, seed=1):
    sv = np.asarray(sv, float); dv = np.asarray(dv, float); g = np.asarray(g)
    u = np.unique(g); idx = {t: np.where(g == t)[0] for t in u}; rng = np.random.default_rng(seed); bs = []
    for _ in range(4000):
        pk = np.concatenate([idx[t] for t in rng.choice(u, len(u))]); bs.append(np.polyfit(sv[pk], dv[pk], 1)[0])
    return np.polyfit(sv, dv, 1)[0], np.percentile(bs, 2.5), np.percentile(bs, 97.5)


for run, key in (("base", "stop_label_released"), ("stopft", "stop_label_dist")):
    S = rows(load(run), key); M = [x for x in S if not x["stop"]]
    print("=" * 20, run)
    # I1: exact decision agreement
    same_rv = np.mean([x["real"][0] == x["vq"][0] for x in S])
    # reference: agreement between two independent samples on the same real frame (samples 1 and 2)
    same_ss = np.mean([x["real"][1] == x["real"][2] for x in S])
    same_rg = np.mean([x["real"][0] == x["gen"][0] for x in S])
    same_rv_turn = np.mean([(x["real"][0] == "stop") == (x["vq"][0] == "stop") for x in S])
    print(f"I1 greedy identical real vs vq {same_rv:.3f}; real vs gen {same_rg:.3f}; two samples on real {same_ss:.3f}")
    # agreement within tolerance of each other (real vs vq)
    def close(p, q, tol=(0.1, 0.1, 0.2)):
        if p == "stop" or q == "stop": return p == q
        if p is None or q is None: return False
        return all(abs(a - c) <= t + 1e-9 for a, c, t in zip(p, q, tol))
    print(f"   real~vq within report tolerance {np.mean([close(x['real'][0], x['vq'][0]) for x in S]):.3f}; within (0.03,0.03,0.05) {np.mean([close(x['real'][0], x['vq'][0], (0.03,0.03,0.05)) for x in S]):.3f}; "
          f"real~gen within (0.03,0.03,0.05) {np.mean([close(x['real'][0], x['gen'][0], (0.03,0.03,0.05)) for x in S]):.3f}")
    for tol in ((0.05, 0.05, 0.1), (0.03, 0.03, 0.05)):
        d = [acc(x["real"][0], x["rec"], x["stop"], tol) - acc(x["vq"][0], x["rec"], x["stop"], tol) for x in S]
        print(f"   real-vq paired at tol {tol}: {f(cboot(d,[x['traj'] for x in S]))}")
    d = [ok(x["real"][0], x["rec"]) - ok(x["vq"][0], x["rec"]) for x in S if not x["stop"] and abs(x["rec"][2]) >= TURN]
    g = [x["traj"] for x in S if not x["stop"] and abs(x["rec"][2]) >= TURN]
    print(f"   turn sign real-vq paired {f(cboot(d, g))}")
    # jackknife-t CIs for headline paired numbers
    for lab, T in (("all", S), ("move", M)):
        for oth in ("gen", "vq"):
            d = [acc(x["real"][0], x["rec"], x["stop"]) - acc(x[oth][0], x["rec"], x["stop"]) for x in T]
            m, lo, hi = jack(d, [x["traj"] for x in T])
            print(f"   jackknife-t real-{oth} {lab}: {m:+.3f} [{lo:+.3f},{hi:+.3f}]")
    # trajectory-level sign test: how many trajectories have positive real-gen loss (all / move)
    for lab, T in (("all", S), ("move", M)):
        per = {}
        for x in T:
            per.setdefault(x["traj"], []).append(acc(x["real"][0], x["rec"], x["stop"]) - acc(x["gen"][0], x["rec"], x["stop"]))
        pm = np.array([np.mean(v) for v in per.values()])
        print(f"   per-trajectory real-gen {lab}: {np.sum(pm>0)} positive, {np.sum(pm==0)} zero, {np.sum(pm<0)} negative of {len(pm)}; "
              f"median {np.median(pm):+.3f}; max {pm.max():+.3f}")
    # strict-tolerance depth trend on move steps
    for tol in ((0.1, 0.1, 0.2), (0.05, 0.05, 0.1), (1e9, 1e9, 0.05)):
        dv = [acc(x["real"][0], x["rec"], False, tol) - acc(x["gen"][0], x["rec"], False, tol) for x in M]
        s, lo, hi = slope_ci([x["s"] for x in M], dv, [x["traj"] for x in M])
        bins = [np.mean([d for d, x in zip(dv, M) if min(x["s"] // 5, 4) == bb]) for bb in range(5)]
        print(f"   move tol {tol}: per-bin paired loss {' '.join(f'{v:+.2f}' for v in bins)}; slope/step {s:+.4f} [{lo:+.4f},{hi:+.4f}]")
    # turn-sign paired by bin with CI
    Tn = [x for x in M if abs(x["rec"][2]) >= TURN]
    for bb in range(5):
        T = [x for x in Tn if min(x["s"] // 5, 4) == bb]
        d = [ok(x["real"][0], x["rec"]) - ok(x["gen"][0], x["rec"]) for x in T]
        print(f"   turn-sign paired bin {bb}: {f(cboot(d,[x['traj'] for x in T]),2)}")
    dv = [ok(x["real"][0], x["rec"]) - ok(x["gen"][0], x["rec"]) for x in Tn]
    s, lo, hi = slope_ci([x["s"] for x in Tn], dv, [x["traj"] for x in Tn])
    print(f"   turn-sign paired slope/step {s:+.4f} [{lo:+.4f},{hi:+.4f}]")
    # depth 1..4 only (imagined, not s=0)
    T = [x for x in Tn if 1 <= x["s"] <= 9]
    d = [ok(x["real"][0], x["rec"]) - ok(x["gen"][0], x["rec"]) for x in T]
    print(f"   turn-sign paired s=1..9: {f(cboot(d,[x['traj'] for x in T]),2)}")
    # Stop: where are the dist-rule Stop-labelled non-final steps (s relative to n)
    St = [x for x in S if x["stop"] and not x["final"]]
    if St:
        print("   non-final Stop-labelled steps: steps before end", np.bincount([x["n"] - 1 - x["s"] for x in St]).tolist(),
              "trajectories", len(set(x["traj"] for x in St)))
        # recall by trajectory cluster: share of gen stops from how many trajectories
        hits = [x["traj"] for x in S if x["stop"] and x["gen"][0] == "stop"]
        print("   gen Stop hits by trajectory:", {t: hits.count(t) for t in set(hits)})
        hits = [x["traj"] for x in S if x["stop"] and x["real"][0] == "stop"]
        print("   real Stop hits spread over", len(set(hits)), "trajectories; n hits", len(hits))

# ---------------- depth-1 comparison from the variance run (base; one-step imagination from VQ(real k), recorded delta[k])
d = glob.glob(f"{ROOT}/base_variance*/variance_decomp_results")[0]
V = []
for fn in sorted(glob.glob(d + "/variance_rank*.json")):
    V += json.load(open(fn))
print("=" * 20, "variance run (base):", len(V), "states from", len(set(r["traj"] for r in V)), "trajectories")
rowsA, rowsS, rowsT = [], [], []
for r in V:
    stop = r["stop_next_released"]; ref = r["recorded_next"]
    im = [a for fr in r["imagined"] if fr for a in fr]; re_ = r["real_next"] or []; rq = r["real_next_vq"] or []
    if not im or not re_:
        continue
    if stop:
        rowsS.append((r["traj"], np.mean([a == "stop" for a in re_]), np.mean([a == "stop" for a in rq]), np.mean([a == "stop" for a in im])))
    else:
        if ref is None: continue
        ar = np.mean([acc(a, ref, False) for a in re_]); ai = np.mean([acc(a, ref, False) for a in im])
        rowsA.append((r["traj"], ar, ai))
        if abs(ref[2]) >= TURN:
            rowsT.append((r["traj"], np.mean([ok(a, ref) for a in re_]), np.mean([ok(a, ref) for a in im])))
        for tol in ():
            pass
print(f" move states n={len(rowsA)}: sampled acc real_next {np.mean([a for _,a,_ in rowsA]):.3f} imagined {np.mean([c for *_,c in rowsA]):.3f} "
      f"paired {f(cboot([a-c for _,a,c in rowsA],[t for t,*_ in rowsA]))}")
print(f" turning states n={len(rowsT)}: turn sign real_next {np.mean([a for _,a,_ in rowsT]):.3f} imagined {np.mean([c for *_,c in rowsT]):.3f} "
      f"paired {f(cboot([a-c for _,a,c in rowsT],[t for t,*_ in rowsT]))}")
print(f" goal states n={len(rowsS)}: Stop rate real_next {np.mean([a for _,a,_,_ in rowsS]):.3f} vq {np.mean([q for _,_,q,_ in rowsS]):.3f} imagined(1 step) {np.mean([c for *_,c in rowsS]):.3f} "
      f"; per-state imagined Stop rate distribution {sorted(round(c,2) for *_,c in rowsS)}")
# strict tolerance on the variance run
for tol in ((0.05, 0.05, 0.1),):
    rr = []
    for r in V:
        if r["stop_next_released"] or r["recorded_next"] is None: continue
        im = [a for fr in r["imagined"] if fr for a in fr]; re_ = r["real_next"] or []
        if not im or not re_: continue
        rr.append((r["traj"], np.mean([acc(a, r["recorded_next"], False, tol) for a in re_]) - np.mean([acc(a, r["recorded_next"], False, tol) for a in im])))
    print(f" move states strict tol {tol}: paired real-imagined {f(cboot([v for _,v in rr],[t for t,_ in rr]))}")
