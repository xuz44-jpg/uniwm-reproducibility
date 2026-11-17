from ic_common import *
R = load("base_variance", "variance_decomp_results", "variance_rank*.json")
print("variance states", len(R))
on = [r for r in R if r["stop_next_released"]]
print("Stop-labelled next (released):", len(on), "k-(n-2):", set(r['k'] - (r['n'] - 2) for r in on))
per = [np.mean([a == 'stop' for fr in r['imagined'] if fr for a in fr]) for r in on]
print("per-state Stop rate on 1-step imagined frames:", np.round(sorted(per), 2).tolist())
print("mean %.3f; states with >=50%% Stop %d/%d" % (np.mean(per), sum(p >= 0.5 for p in per), len(per)))
fr_any = [[any(a == 'stop' for a in fr) if fr else None for fr in r['imagined']] for r in on]
fr_maj = [np.mean([np.mean([a == 'stop' for a in fr]) >= 0.5 for fr in r['imagined'] if fr]) for r in on]
print("share of imagined frames with majority Stop: %.3f" % np.mean(fr_maj))
# which trajs are in the intervention set
Ri, _ = rows("base")
ti = set(r['traj'] for r in Ri)
per_i = [p for p, r in zip(per, on) if r['traj'] in ti]
print("restricted to the 24 intervention trajectories: %.3f (n=%d states)" % (np.mean(per_i), len(per_i)))
# dist rule on next frame
on2 = [r for r in R if r["stop_next_dist"]]
print("dist-rule Stop-labelled next frames:", len(on2), "; base Stop rate on imagined for dist-only (non-goal) states %.3f" % np.mean([a == 'stop' for r in on2 if not r['stop_next_released'] for fr in r['imagined'] if fr for a in fr]) if any(not r['stop_next_released'] for r in on2) else "")
