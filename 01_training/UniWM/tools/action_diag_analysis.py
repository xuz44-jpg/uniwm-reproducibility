"""Analysis of the three action diagnostics (local; numpy only).

usage:
  python tools/action_diag_analysis.py likelihood   <dir with likelihood_rank*.json>
  python tools/action_diag_analysis.py intervention <dir with intervention_rank*.json> --stop_rule {released,dist}
  python tools/action_diag_analysis.py variance     <dir with variance_rank*.json>     --stop_rule {released,dist}

--stop_rule must be the checkpoint's own training rule: `released` (Stop only at the last frame; `base`) or
`dist` (distance 0.2, heading 0.3 rad; `stop` and the data-efficiency models).
All intervals are 95% bootstrap intervals clustered by trajectory (frames within a trajectory are not independent).
"""
import argparse, glob, json, math, os
import numpy as np

TOL = (0.1, 0.1, 0.2)          # action-accuracy tolerance of the report (scripts/evaluator.py, Table 6.1)
TURN = 0.05                    # |dyaw| threshold for a turning step


def load(d, pat):
    recs = []
    for f in sorted(glob.glob(os.path.join(d, pat))):
        recs += json.load(open(f))
    return recs


def cluster_ci(values, groups, n_boot=4000, seed=0):
    v = np.asarray(values, float); g = np.asarray(groups)
    if len(v) == 0:
        return float("nan"), float("nan"), float("nan")
    uniq = np.unique(g); idx = {u: np.where(g == u)[0] for u in uniq}; rng = np.random.default_rng(seed); boots = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, len(uniq)); boots.append(np.concatenate([v[idx[u]] for u in pick]).mean())
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(v.mean()), float(lo), float(hi)


def fmt(m, lo, hi, p=3):
    return f"{m:+.{p}f} [{lo:+.{p}f}, {hi:+.{p}f}]"


# ------------------------------------------------------------------ likelihood
def _text(r, c):
    s = r["scores"][c]
    return s.get("action_text") or r["scores"][s["same_text_as"]]["action_text"]


def _split_best(r, C):
    """Credit split over tied winners (ties are identical action texts, hence identical NLL)."""
    nll = {c: r["scores"][c]["nll_img"] for c in C}
    m = min(nll.values())
    win_txt = {_text(r, c) for c in C if nll[c] == m}
    w = [c for c in C if _text(r, c) in win_txt]
    return {c: 1.0 / len(w) for c in w}


def likelihood(d, _args):
    R_all = load(d, "likelihood_rank*.json")
    G = [r for r in R_all if r.get("target_is_goal", r["k"] + 1 == r["n"] - 1)]
    R = [r for r in R_all if not r.get("target_is_goal", r["k"] + 1 == r["n"] - 1)]
    n_img = R_all[0]["scores"]["recorded"]["n_tgt"] - 3
    print(f"frames: {len(R_all)} from {len(set(r['traj'] for r in R_all))} trajectories; {n_img} image tokens per target; "
          f"{len(G)} goal-target frames excluded (their target is the goal image in the prompt; reported below as a copy control)")
    subsets = {
        "all frames": lambda r: True,
        "straight (|dyaw[k]|<0.02)": lambda r: abs(r["dyaw_k"]) < 0.02,
        "turning (|dyaw[k]|>=0.05)": lambda r: abs(r["dyaw_k"]) >= TURN,
        "mid-turn (|dyaw[k-1]|,|dyaw[k]|>=0.05, same sign)": lambda r: r["dyaw_km1"] is not None and abs(r["dyaw_k"]) >= TURN
            and abs(r["dyaw_km1"]) >= TURN and np.sign(r["dyaw_k"]) == np.sign(r["dyaw_km1"]),
    }
    print("\nΔNLL = NLL(candidate) - NLL(recorded), summed over the image tokens of the real next frame (nats), on frames where")
    print("the candidate's binned action text differs from the recorded one. Positive = the real next frame is LESS likely")
    print("under the candidate. 'recorded better' = share of those frames with ΔNLL > 0.")
    for sname, cond in subsets.items():
        S = [r for r in R if cond(r)]
        if not S:
            print(f"\n[{sname}] n=0"); continue
        print(f"\n[{sname}] n={len(S)} frames, {len(set(r['traj'] for r in S))} trajectories")
        for c in ("negated", "zero", "other_traj", "prev1", "prev2", "next1"):
            rows = [r for r in S if c in r["scores"]]
            diff = [r for r in rows if _text(r, c) != _text(r, "recorded")]
            if not diff:
                print(f"  {c:10s} n=0 distinct ({len(rows)} identical text)"); continue
            dv = [r["scores"][c]["nll_img"] - r["scores"]["recorded"]["nll_img"] for r in diff]
            m, lo, hi = cluster_ci(dv, [r["traj"] for r in diff])
            print(f"  {c:10s} n={len(diff):4d} distinct, {len(rows) - len(diff):3d} identical text  ΔNLL {fmt(m, lo, hi, 2)}  "
                  f"per token {m / n_img:+.5f}  recorded better on {np.mean([x > 0 for x in dv]):.2f}")
        C4 = ("recorded", "negated", "zero", "other_traj")
        base = [r for r in S if all(c in r["scores"] for c in C4)]
        if base:
            share = np.mean([_split_best(r, C4).get("recorded", 0.0) for r in base])
            print(f"  recorded's share of first place among {C4} (ties split): {share:.2f} of {len(base)} frames (chance 0.25)")
        CL = ("prev2", "prev1", "recorded", "next1")
        lead = [r for r in S if all(c in r["scores"] for c in CL)]
        if lead:
            cred = {c: 0.0 for c in CL}
            for r in lead:
                for c, v in _split_best(r, CL).items():
                    cred[c] += v
            distinct = [r for r in lead if len({_text(r, c) for c in CL}) == len(CL)]
            line = "  label timing, share of first place (ties split): " + ", ".join(f"{c} {cred[c] / len(lead):.2f}" for c in CL) + f" (n={len(lead)})"
            if distinct:
                best = [min(CL, key=lambda c: r["scores"][c]["nll_img"]) for r in distinct]
                line += "; all four texts distinct: " + ", ".join(f"{c} {best.count(c) / len(best):.2f}" for c in CL) + f" (n={len(distinct)}, {len(lead) - len(distinct)} dropped)"
            print(line)
    if G:
        dv = [r["scores"]["negated"]["nll_img"] - r["scores"]["recorded"]["nll_img"] for r in G if _text(r, "negated") != _text(r, "recorded")]
        if dv:
            print(f"\ncopy control (goal-target frames): mean ΔNLL negated-recorded {np.mean(dv):+.2f} over {len(dv)} frames; "
                  f"mean NLL per image token {np.mean([r['scores']['recorded']['nll_img'] for r in G]) / n_img:.3f} vs "
                  f"{np.mean([r['scores']['recorded']['nll_img'] for r in R]) / n_img:.3f} on other frames")


# ------------------------------------------------------------------ intervention
def _bin(v, w=0.01):
    """Quantisation of action_to_text + extract_bin_values (the gold side of scripts/evaluator.py)."""
    return round((1 if v >= 0 else -1) * math.floor(abs(v) / w) * w, 4)


def _acc(pred, ref, stop):
    """Report definition: Stop must match Stop; otherwise all components within TOL of the binned label."""
    if stop or pred == "stop":
        return bool(stop) and pred == "stop"
    return pred is not None and all(abs(p - _bin(r)) <= t for p, r, t in zip(pred, ref, TOL))


def _turn_ok(pred, ref):
    return pred not in (None, "stop") and np.sign(pred[2]) == np.sign(ref[2]) and abs(pred[2]) >= TURN / 2


def intervention(d, args):
    key = "stop_label_released" if args.stop_rule == "released" else "stop_label_dist"
    R = load(d, "intervention_rank*.json")
    rows = [(r["traj"], st) for r in R for st in r["steps"]]
    print(f"trajectories: {len(R)}, steps: {len(rows)}, max step index {max(st['s'] for _, st in rows)}; "
          f"Stop rule '{args.stop_rule}': {sum(st[key] for _, st in rows)} Stop-labelled steps")
    for name in ("real", "vq", "gen"):
        valid = [(t, st) for t, st in rows if st.get(name)]
        g = [st[name][0] for _, st in valid]                       # greedy prediction
        acc = [_acc(p, st["recorded"], st[key]) for p, (_, st) in zip(g, valid)]
        mv = [(p, st) for p, (_, st) in zip(g, valid) if not st[key] and st["recorded"] is not None]
        err = [abs(p[2] - st["recorded"][2]) for p, st in mv if p not in (None, "stop")]
        turn = [(t, _turn_ok(p, st["recorded"])) for p, (t, st) in zip(g, valid) if st["recorded"] is not None and not st[key] and abs(st["recorded"][2]) >= TURN]
        stop_on = [p == "stop" for p, (_, st) in zip(g, valid) if st[key]]
        stop_off = [p == "stop" for p, (_, st) in zip(g, valid) if not st[key]]
        zero = [p not in (None, "stop") and all(v == 0.0 for v in p) for p in g]
        m, lo, hi = cluster_ci(acc, [t for t, _ in valid])
        tm, tlo, thi = cluster_ci([x for _, x in turn], [t for t, _ in turn]) if turn else (np.nan,) * 3
        print(f"  {name:4s} greedy (n={len(valid)}): accuracy {fmt(m, lo, hi, 2)}; mean |dyaw err| on move steps {np.mean(err):.3f}; "
              f"turn sign correct {fmt(tm, tlo, thi, 2)} (n={len(turn)}); Stop recall {np.mean(stop_on) if stop_on else float('nan'):.2f} "
              f"(n={len(stop_on)}), false Stop {np.mean(stop_off):.3f}; zero action {np.mean(zero):.3f}")
    print("\n  accuracy by imagination depth (greedy): real / vq / gen")
    depth = {}
    for t, st in rows:
        b = min(st["s"] // 5, 4)
        for name in ("real", "vq", "gen"):
            if st.get(name):
                depth.setdefault((b, name), []).append(_acc(st[name][0], st["recorded"], st[key]))
    for b in range(5):
        cells = [f"{np.mean(depth[(b, n)]):.2f}" if (b, n) in depth else "  - " for n in ("real", "vq", "gen")]
        print(f"    steps {5 * b:2d}-{5 * b + 4:2d}{'+' if b == 4 else ' '}: " + " / ".join(cells) + f"  (n={len(depth.get((b, 'real'), []))})")
    for other in ("gen", "vq"):
        paired = [(t, _acc(st["real"][0], st["recorded"], st[key]) - _acc(st[other][0], st["recorded"], st[key])) for t, st in rows if st.get("real") and st.get(other)]
        m, lo, hi = cluster_ci([x for _, x in paired], [t for t, _ in paired])
        print(f"  paired accuracy loss real - {other:3s}: {fmt(m, lo, hi, 3)} (n={len(paired)} steps)")


# ------------------------------------------------------------------ variance decomposition
def variance(d, args):
    key = "stop_next_released" if args.stop_rule == "released" else "stop_next_dist"
    R = load(d, "variance_rank*.json")
    print(f"states: {len(R)} from {len(set(r['traj'] for r in R))} trajectories; Stop-labelled next frames: {sum(r[key] for r in R)}")
    moves = lambda acts: [a for a in (acts or []) if a not in (None, "stop")]
    for comp, ci in (("dx", 0), ("dy", 1), ("dyaw", 2)):
        within, between, between_raw, ref_var, ref_vq_var, grp = [], [], [], [], [], []
        for r in R:
            frames = [[a[ci] for a in moves(fr)] for fr in r["imagined"] if fr]
            frames = [f for f in frames if len(f) >= 2]
            ref = [a[ci] for a in moves(r["real_next"])]; refq = [a[ci] for a in moves(r.get("real_next_vq"))]
            if len(frames) < 2 or len(ref) < 2 or len(refq) < 2:
                continue                                   # a state counts only when every term is defined
            vj = [np.var(f, ddof=1) for f in frames]; nj = [len(f) for f in frames]
            within.append(np.mean(vj))
            b_raw = np.var([np.mean(f) for f in frames], ddof=1)
            between_raw.append(b_raw)
            between.append(b_raw - np.mean([v / nn for v, nn in zip(vj, nj)]))   # remove the action-sampling noise of the frame means
            ref_var.append(np.var(ref, ddof=1)); ref_vq_var.append(np.var(refq, ddof=1)); grp.append(r["traj"])
        if not within:
            print(f"  {comp:4s}: no state with every term defined"); continue
        w = np.mean(within); b = max(np.mean(between), 0.0); braw = np.mean(between_raw)
        print(f"  {comp:4s} (n={len(within)} states): within-image (action sampling) {w:.5f}; between-image, noise-corrected {b:.5f} "
              f"(uncorrected {braw:.5f}); image share {b / (w + b):.2f} (uncorrected {braw / (w + braw):.2f}) | action-sampling variance "
              f"on the real next frame {np.mean(ref_var):.5f}, on its VQ round trip {np.mean(ref_vq_var):.5f}")
    for side, getter in (("imagined frames", lambda r: [a for fr in r["imagined"] if fr for a in fr]),
                         ("real next frame", lambda r: r["real_next"] or []), ("its VQ round trip", lambda r: r.get("real_next_vq") or [])):
        on = [a == "stop" for r in R if r[key] for a in getter(r)]; off = [a == "stop" for r in R if not r[key] for a in getter(r)]
        print(f"  Stop rate on {side}: {np.mean(on) if on else float('nan'):.2f} on Stop-labelled states (n={len(on)}), {np.mean(off) if off else float('nan'):.3f} otherwise")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["likelihood", "intervention", "variance"]); ap.add_argument("dir")
    ap.add_argument("--stop_rule", choices=["released", "dist"], default="dist")
    a = ap.parse_args()
    {"likelihood": likelihood, "intervention": intervention, "variance": variance}[a.kind](a.dir, a)
