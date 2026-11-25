"""Summarize results.jsonl: accuracy by angle, order bias, swap consistency, exact binomial tests."""
import json, sys
from math import comb
from pathlib import Path
HERE = Path(__file__).resolve().parent
spec = json.loads((HERE / 'queries.json').read_text())
Q = {q['id']: q for q in spec['queries']}
R = {r['id']: r for r in map(json.loads, (HERE / (sys.argv[1] if len(sys.argv) > 1 else 'results.jsonl')).read_text().splitlines())}
assert set(R) == set(Q), (len(R), len(Q))

def binom_two_sided(k, n):
    p = [comb(n, i) / 2 ** n for i in range(n + 1)]
    return min(1.0, sum(x for x in p if x <= p[k] + 1e-15))

def logit_pred(r): return 'LEFT' if r['logit_left'] > r['logit_right'] else 'RIGHT'

scored = [q for q in Q.values() if q['answer']]
out = dict(by_magnitude={}, by_order={}, by_scene={})
for name, pred in [('greedy', lambda r: r['parsed']), ('logit', logit_pred)]:
    rows = {}
    for m in [3, 5, 7, 10, 15]:
        qs = [q for q in scored if abs(q['yaw']) == m]
        k = sum(pred(R[q['id']]) == q['answer'] for q in qs)
        rows[m] = dict(correct=k, n=len(qs), acc=round(k / len(qs), 3), p_vs_chance=round(binom_two_sided(k, len(qs)), 4))
    k = sum(pred(R[q['id']]) == q['answer'] for q in scored)
    rows['all'] = dict(correct=k, n=len(scored), acc=round(k / len(scored), 3), p_vs_chance=round(binom_two_sided(k, len(scored)), 6))
    out['by_magnitude'][name] = rows
    out['by_order'][name] = {o: sum(pred(R[q['id']]) == q['answer'] for q in scored if q['order'] == o) for o in ['fwd', 'rev']}
    out['by_scene'][name] = {s: sum(pred(R[q['id']]) == q['answer'] for q in scored if q['scene'] == s) for s in ['apartment_1', 'skokloster_castle']}
    # Swap consistency: the answer flips when the two images are swapped.
    pairs = [(q, Q[q['id'][:-3] + 'rev']) for q in scored if q['order'] == 'fwd']
    out.setdefault('swap', {})[name] = dict(
        flips=sum(pred(R[f['id']]) != pred(R[r['id']]) for f, r in pairs),
        both_correct=sum(pred(R[f['id']]) == f['answer'] and pred(R[r['id']]) == r['answer'] for f, r in pairs),
        n_pairs=len(pairs))
    out.setdefault('left_rate', {})[name] = dict(
        scored=round(sum(pred(R[q['id']]) == 'LEFT' for q in scored) / len(scored), 3),
        identical_pairs=sum(pred(R[q['id']]) == 'LEFT' for q in Q.values() if q['order'] == 'same'))
out['unparsed'] = sum(R[q['id']]['parsed'] is None for q in scored)
out['answer_texts'] = sorted({r['text'] for r in R.values()})
print(json.dumps(out, indent=1))
(HERE / (sys.argv[2] if len(sys.argv) > 2 else 'summary.json')).write_text(json.dumps(out, indent=1))
