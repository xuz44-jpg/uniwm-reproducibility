import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "interv_confounds_lens"))  # helper module this script was written against
from ic_common import *
for run, key in (("base", "stop_label_released"), ("stopft", "stop_label_dist")):
    R, rw = rows(run)
    print("=====", run, key, "trajs", len(R), "steps", len(rw))
    # sanity: s=0 gen vs vq identical inputs
    same = [st["gen"][0] == st["vq"][0] for t, n, st in rw if st["s"] == 0]
    same_all = [st["gen"] == st["vq"] for t, n, st in rw if st["s"] == 0]
    print("s=0 gen greedy == vq greedy:", sum(same), "/", len(same), "; all 6 identical:", sum(same_all))
    # None and gen availability
    for nm in ("real", "vq", "gen"):
        print(nm, "missing obs", sum(1 for _, _, st in rw if not st.get(nm)), "greedy None", sum(1 for _, _, st in rw if st.get(nm) and st[nm][0] is None))
    # stop steps by depth bin
    import collections
    c = collections.Counter(min(st["s"] // 5, 4) for _, _, st in rw if st[key]); print("Stop-labelled by bin", sorted(c.items()))
    c2 = collections.Counter(min(st["s"] // 5, 4) for _, _, st in rw); print("all by bin", sorted(c2.items()))
    # stop steps: position relative to end
    print("Stop steps s-(n-1):", collections.Counter(st["s"] - (n - 1) for _, n, st in rw if st[key]))
    # recorded action magnitudes on move steps
    rec = np.array([st["recorded"] for _, _, st in rw if st["recorded"] is not None and not st[key]])
    print("recorded |dx| median %.3f p90 %.3f; |dy| median %.3f p90 %.3f; |dyaw| median %.3f p90 %.3f max %.3f" % (
        np.median(abs(rec[:, 0])), np.percentile(abs(rec[:, 0]), 90), np.median(abs(rec[:, 1])), np.percentile(abs(rec[:, 1]), 90),
        np.median(abs(rec[:, 2])), np.percentile(abs(rec[:, 2]), 90), abs(rec[:, 2]).max()))
    print("share of move steps with |dyaw|>=0.05:", np.mean(abs(rec[:, 2]) >= 0.05), " |dyaw|>0.2:", np.mean(abs(rec[:, 2]) > 0.2))
