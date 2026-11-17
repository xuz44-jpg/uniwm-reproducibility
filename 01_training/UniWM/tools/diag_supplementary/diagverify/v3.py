import sys; sys.path.insert(0, __file__.rsplit('/',1)[0])
from common import *
R = V()
S = [r for r in R if r['stop_next_released']]
print("Stop-labelled states:", len(S), "all k=n-2:", all(r['k'] == r['n']-2 for r in S))
per_state = []; per_frame = []
for r in S:
    fr = [f for f in r['imagined'] if f]
    rates = [np.mean([a == 'stop' for a in f]) for f in fr]
    per_frame += rates
    per_state.append((r['traj'], np.mean([a == 'stop' for f in fr for a in f]), len(fr), rates))
ps = np.array([p[1] for p in per_state]); pf = np.array(per_frame)
print(f"pooled Stop on imagined (per sample): {np.mean([a=='stop' for r in S for f in r['imagined'] if f for a in f]):.3f} (n={sum(len(f) for r in S for f in r['imagined'] if f)})")
print(f"per-state Stop rate: mean {ps.mean():.3f}; states at 1.0: {(ps==1).sum()}, >=0.8: {(ps>=0.8).sum()}, <=0.2: {(ps<=0.2).sum()}, in (0.2,0.8): {((ps>0.2)&(ps<0.8)).sum()} of {len(ps)}")
print(f"per-frame Stop rate: frames {len(pf)}; all-5 Stop {(pf==1).sum()}, none {(pf==0).sum()}, mixed {((pf>0)&(pf<1)).sum()}")
print("  mixed frames rates:", sorted(pf[(pf>0)&(pf<1)].round(2).tolist()))
# frame-level majority Stop (how the closed loop w/ majority would act) and greedy-equivalent unknown
print(f"frame-level majority-Stop: {(pf>0.5).mean():.3f} of {len(pf)} frames")
G = {}
for t, p, _, _ in per_state: G.setdefault(t, []).append(p)
lo, hi = cluster_boot(lambda x: np.mean(x), G); print(f"per-state mean {ps.mean():.3f} traj-bootstrap [{lo:.2f},{hi:.2f}]")
for t, p, nf, rates in sorted(per_state, key=lambda x: x[1])[:6]:
    print("   low:", t, round(p,2), [round(x,2) for x in rates])
# intervention: goal-frame (s=n-1) Stop on gen/real/vq, greedy and sampled, base
Ib = I("base")
common = set(r['traj'] for r in Ib)
print("variance-run Stop states on the 24 intervention trajectories:", np.mean([a=='stop' for r in S if r['traj'] in common for f in r['imagined'] if f for a in f]).round(3))
for name in ("real", "vq", "gen"):
    g = []; smp = []
    for r in Ib:
        st = r['steps'][-1]; assert st['stop_label_released'] and st['s'] == r['n']-1
        acts = st[name]
        if acts: g.append(acts[0] == 'stop'); smp += [a == 'stop' for a in acts[1:]]
    print(f"intervention base, s=n-1 (depth n-1), {name}: greedy Stop {np.mean(g):.3f} ({sum(g)}/{len(g)}), sampled Stop {np.mean(smp):.3f} ({sum(smp)}/{len(smp)})")
# the input frame of the variance-run imagination (real frame n-2) never triggers Stop?
g = []; smp = []
for r in Ib:
    st = r['steps'][-2]
    for name in ('real', 'vq'):
        smp += [a == 'stop' for a in st[name][1:]]; g.append(st[name][0] == 'stop')
print(f"intervention base, s=n-2 real/vq frames: greedy Stop {sum(g)}/{len(g)}, sampled Stop {sum(smp)}/{len(smp)}")
# gen Stop by depth on non-final steps (false Stops) for base
dep = {}
for r in Ib:
    for st in r['steps'][:-1]:
        b = min(st['s'] // 5, 4)
        dep.setdefault(b, []).extend([a == 'stop' for a in (st['gen'] or [])[1:]])
print("base gen sampled false-Stop by depth bin:", {5*b: round(np.mean(v),3) for b, v in sorted(dep.items())})
# stopft: Stop recall on gen by steps-to-go (dist rule), sampled and greedy
Is = I("stopft")
bins = {}
for r in Is:
    n = r['n']
    for st in r['steps']:
        if not st['stop_label_dist'] or not st['gen']: continue
        togo = n - 1 - st['s']
        key = 'final' if togo == 0 else ('1-2 before end' if togo <= 2 else '>=3 before end')
        bins.setdefault(key, {'g': [], 's': [], 'real': [], 'depth': []})
        bins[key]['g'].append(st['gen'][0] == 'stop'); bins[key]['s'] += [a == 'stop' for a in st['gen'][1:]]
        bins[key]['real'].append(st['real'][0] == 'stop'); bins[key]['depth'].append(st['s'])
for k, v in bins.items():
    print(f"stopft dist-rule Stop steps [{k}]: n={len(v['g'])}, depth mean {np.mean(v['depth']):.1f}, gen greedy Stop {np.mean(v['g']):.2f}, gen sampled {np.mean(v['s']):.2f}, real greedy {np.mean(v['real']):.2f}")
