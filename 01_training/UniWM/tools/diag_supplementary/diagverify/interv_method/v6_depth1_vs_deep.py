from ivcommon import *
from ivcommon import _bin
import glob, json
V = []
for f in sorted(glob.glob(f"{RAW}/base_variance*/variance_decomp_results/variance_rank*.json")):
    V += json.load(open(f))
R = load("base"); I = {(r["traj"], st["s"]): st for r in R for st in r["steps"]}
key = "stop_label_released"
print(f"variance states {len(V)}; matched to intervention (traj, s=k+1): {sum((v['traj'], v['k']+1) in I for v in V)}")
cols = {"real (interv. samples)": [], "real_next (variance samples)": [], "real_next_vq (variance)": [], "depth-1 imagined (variance)": [], "depth-s imagined (interv. gen samples)": []}
grp = []; depth = []; stopl = []
per_state = []
for v in V:
    st = I.get((v["traj"], v["k"] + 1))
    if st is None: continue
    lab = st["recorded"]; stop = st[key]
    assert stop == v["stop_next_released"], (v["traj"], v["k"])
    if lab is not None and v["recorded_next"] is not None:
        assert np.allclose(lab, v["recorded_next"])
    a_real = np.mean([acc(p, lab, stop) for p in st["real"][1:]])
    a_rn = np.mean([acc(p, lab, stop) for p in v["real_next"]])
    a_rq = np.mean([acc(p, lab, stop) for p in v["real_next_vq"]])
    a_d1 = np.mean([acc(p, lab, stop) for fr in v["imagined"] if fr for p in fr])
    a_ds = np.mean([acc(p, lab, stop) for p in st["gen"][1:]])
    per_state.append((v["traj"], st["s"], stop, a_real, a_rn, a_rq, a_d1, a_ds))
P = per_state
print(f"matched states: {len(P)} (Stop-labelled {sum(p[2] for p in P)}); depths s = {sorted(set(p[1] for p in P))}")
names = ["real (interv. samples)", "real_next (variance)", "real_next_vq (variance)", "depth-1 imagined (variance)", "depth-s imagined (interv. gen)"]
for sub, sel in (("all", lambda p: True), ("move-labelled", lambda p: not p[2]), ("Stop-labelled (goal frame)", lambda p: p[2])):
    S = [p for p in P if sel(p)]
    print(f"  [{sub}] n={len(S)}: " + "; ".join(f"{nm} {np.mean([p[3+i] for p in S]):.3f}" for i, nm in enumerate(names)))
    if len(S) > 5:
        g = [p[0] for p in S]
        print(f"     paired real_next - depth1: {f3(cluster_ci([p[4]-p[6] for p in S], g))};  paired depth1 - depth-s: {f3(cluster_ci([p[6]-p[7] for p in S], g))}; paired real - depth-s: {f3(cluster_ci([p[3]-p[7] for p in S], g))}")
# depth split of move-labelled matched states
for lo, hi in ((1, 9), (10, 24)):
    S = [p for p in P if not p[2] and lo <= p[1] <= hi]
    print(f"  move-labelled s in [{lo},{hi}] n={len(S)}: real {np.mean([p[3] for p in S]):.3f} real_next {np.mean([p[4] for p in S]):.3f} depth-1 {np.mean([p[6] for p in S]):.3f} depth-s {np.mean([p[7] for p in S]):.3f}")
