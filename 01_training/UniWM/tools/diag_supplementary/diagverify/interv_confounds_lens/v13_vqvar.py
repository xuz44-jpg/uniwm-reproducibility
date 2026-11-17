from ic_common import *
R = load("base_variance", "variance_decomp_results", "variance_rank*.json")
mv = lambda A: [a for a in (A or []) if a not in (None, 'stop')]
for comp, ci in (("dx", 0), ("dy", 1), ("dyaw", 2)):
    d, T, vr, vq = [], [], [], []
    for r in R:
        a = [x[ci] for x in mv(r['real_next'])]; b = [x[ci] for x in mv(r['real_next_vq'])]
        if len(a) >= 2 and len(b) >= 2:
            vr.append(np.var(a, ddof=1)); vq.append(np.var(b, ddof=1)); d.append(vr[-1] - vq[-1]); T.append(r['traj'])
    vr, vq = np.array(vr), np.array(vq)
    print("%s n=%d: mean var real %.5f vq %.5f; median %.5f / %.5f; paired real-vq %s; top-5 real var share of sum %.2f" % (comp, len(d), vr.mean(), vq.mean(), np.median(vr), np.median(vq), f3(cluster_ci(d, T), 5), np.sort(vr)[-5:].sum() / vr.sum()))
# greedy-equivalent: modal agreement of sampled actions real vs vq on the intervention runs (sampled draws)
for run, key in (("base", "stop_label_released"), ("stopft", "stop_label_dist")):
    Ri, rw = rows(run)
    for nm in ("real", "vq", "gen"):
        sd = []
        for t, n, st in rw:
            if st[key]: continue
            xs = np.array([a for a in st[nm][1:] if a not in (None, 'stop')])
            if len(xs) >= 2: sd.append(xs.var(0, ddof=1))
        print("  %s %s: mean within-step sampling var over 5 draws dx %.5f dy %.5f dyaw %.5f" % (run, nm, *np.mean(sd, 0)))
