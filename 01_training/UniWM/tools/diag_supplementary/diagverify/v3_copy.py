import sys; sys.path.insert(0, __file__.rsplit('/',1)[0])
from common import *
L = load("base_likelihood*/action_likelihood_results/likelihood_rank*.json")
n_img = 784
G = {r['traj']: r for r in L if r.get('target_is_goal')}
O = [r for r in L if not r.get('target_is_goal')]
print("goal-target frames", len(G), "other", len(O))
g = np.array([r['scores']['recorded']['nll_img'] / n_img for r in G.values()])
o = np.array([r['scores']['recorded']['nll_img'] / n_img for r in O])
print(f"NLL/token recorded: goal-target mean {g.mean():.3f} median {np.median(g):.3f} max {g.max():.3f}; other mean {o.mean():.3f} min {o.min():.3f}")
# exp(-nll) per token -> geometric-mean prob
print(f"geometric-mean per-token prob goal-target {np.exp(-g.mean()):.3f}; other {np.exp(-o.mean()):.3f}")
# are goal-target k == n-2 matching the variance run's Stop states?
R = V(); S = {r['traj']: r for r in R if r['stop_next_released']}
rows = []
for t, r in S.items():
    if t not in G: continue
    assert G[t]['k'] == r['k'], (t, G[t]['k'], r['k'])
    stop = np.mean([a == 'stop' for f in r['imagined'] if f for a in f])
    rows.append((t, stop, G[t]['scores']['recorded']['nll_img'] / n_img))
rows.sort(key=lambda x: x[1])
for t, s, nl in rows: print(f"  {t} imagined-Stop {s:.2f}  goal-target NLL/token {nl:.3f}")
a = np.array([x[1] for x in rows]); b = np.array([x[2] for x in rows])
def rank(x): return np.argsort(np.argsort(x))
print("Spearman(stop rate, goal NLL/token) = %.2f (n=%d)" % (np.corrcoef(rank(a), rank(b))[0,1], len(rows)))
print("mean goal NLL/token for states with Stop<0.5: %.3f vs >=0.5: %.3f" % (b[a<0.5].mean(), b[a>=0.5].mean()))
