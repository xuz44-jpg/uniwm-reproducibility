from ic_common import *
for run, key in (("base", "stop_label_released"), ("stopft", "stop_label_dist")):
    R, rw = rows(run)
    T = np.array([t for t, _, _ in rw]); STOP = np.array([st[key] for _, _, st in rw]); S = np.array([st['s'] for _, _, st in rw])
    A = {nm: np.array([acc(st[nm][0], st['recorded'], st[key]) for _, _, st in rw], float) for nm in ("real", "vq", "gen")}
    print(run, "n", len(rw), "real-vq", f3(cluster_ci(A['real'] - A['vq'], T)), "real-gen", f3(cluster_ci(A['real'] - A['gen'], T)),
          "| Stop steps %d, share of real-gen loss from Stop steps %.2f" % (STOP.sum(), (A['real'] - A['gen'])[STOP].sum() / (A['real'] - A['gen']).sum()))
    B = np.minimum(S // 5, 4)
    print("  bins all real/gen:", " ".join("%.2f/%.2f" % (A['real'][B == b].mean(), A['gen'][B == b].mean()) for b in range(5)))
    print("  Stop recall real %.3f (%d/%d) gen %.3f (%d/%d)" % (np.mean([st['real'][0] == 'stop' for _, _, st in rw if st[key]]), sum(st['real'][0] == 'stop' for _, _, st in rw if st[key]), STOP.sum(), np.mean([st['gen'][0] == 'stop' for _, _, st in rw if st[key]]), sum(st['gen'][0] == 'stop' for _, _, st in rw if st[key]), STOP.sum()))
