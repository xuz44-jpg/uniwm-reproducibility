from ivcommon import *
from ivcommon import _bin
for run, key in (("base", "stop_label_released"), ("stopft", "stop_label_dist")):
    R = load(run)
    print(f"\n== {run}")
    # accuracy of the real greedy prediction from frame s-j scored against label s (move-labelled s, s-j>=0, s>=6 so all j comparable)
    res = {}
    for j in range(0, 7):
        a = []; g = []
        for r in R:
            st_by = {st["s"]: st for st in r["steps"]}
            for st in r["steps"]:
                s = st["s"]
                if st[key] or s < 6: continue
                p = st_by[s - j]["real"][0]
                a.append(acc(p if p != "stop" else None, st["recorded"], False)); g.append(r["traj"])
        res[j] = np.mean(a)
    gen = []; realnow = []
    for r in R:
        for st in r["steps"]:
            if st[key] or st["s"] < 6: continue
            gen.append(acc(st["gen"][0], st["recorded"], False))
    print(f"  move-labelled s>=6 (n={len(gen)}): gen acc {np.mean(gen):.3f}; real frame s-j scored on label s: " + " ".join(f"j={j}:{v:.3f}" for j, v in res.items()))
    # mean |dyaw diff| between gen at s and real at s-j
    out = []
    for j in range(0, 7):
        d = []
        for r in R:
            st_by = {st["s"]: st for st in r["steps"]}
            for st in r["steps"]:
                s = st["s"]
                if s < 6 or st[key]: continue
                a, b = st["gen"][0], st_by[s - j]["real"][0]
                if a in (None, "stop") or b in (None, "stop"): continue
                d.append(abs(a[2] - b[2]))
        out.append(f"j={j}:{np.mean(d):.4f}")
    print("  mean |gen dyaw[s] - real dyaw[s-j]|: " + " ".join(out))
    # turn-sign early failures: does gen predict 'straight' at the first steps of a turn? use onset = first s with |dyaw|>=0.05 after >=2 straight steps
    on = []
    for r in R:
        for st in r["steps"]:
            s = st["s"]
            if st[key] or st["recorded"] is None or abs(st["recorded"][2]) < TURN: continue
            prev = [x for x in r["steps"] if x["s"] in (s - 1,)]
            onset = bool(prev) and prev[0]["recorded"] is not None and abs(prev[0]["recorded"][2]) < TURN
            on.append((onset, turn_ok(st["real"][0], st["recorded"]), turn_ok(st["gen"][0], st["recorded"])))
    for o in (True, False):
        S = [x for x in on if x[0] == o]
        print(f"  turning steps {'at turn onset (prev step not turning)' if o else 'continuing a turn'}: n={len(S)} real {np.mean([x[1] for x in S]):.2f} gen {np.mean([x[2] for x in S]):.2f}")
