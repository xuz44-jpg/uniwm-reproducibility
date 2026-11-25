"""Summarize a LEFT/RIGHT/NONE/UNSURE run: per magnitude answer counts, identical pairs, swap consistency."""
import json, sys
from collections import Counter
from pathlib import Path
HERE = Path(__file__).resolve().parent
Q = {q['id']: q for q in json.loads((HERE / 'queries.json').read_text())['queries']}
R = {r['id']: r for r in map(json.loads, (HERE / sys.argv[1]).read_text().splitlines())}
assert set(R) == set(Q)
def logit_pred(r): return max(r['logits'], key=r['logits'].get)
out = {}
for name, pred in [('greedy', lambda r: r['parsed']), ('logit', logit_pred)]:
    o = {}
    for m in [3, 5, 7, 10, 15]:
        qs = [q for q in Q.values() if q['answer'] and abs(q['yaw']) == m]
        c = Counter(); 
        for q in qs:
            p = pred(R[q['id']])
            c['correct' if p == q['answer'] else 'wrong_direction' if p in ('LEFT', 'RIGHT') else str(p)] += 1
        o[f'yaw{m}'] = dict(c, n=len(qs),
                           left_truth_LEFT=sum(pred(R[q['id']]) == 'LEFT' for q in qs if q['answer'] == 'LEFT'),
                           right_truth_RIGHT=sum(pred(R[q['id']]) == 'RIGHT' for q in qs if q['answer'] == 'RIGHT'))
    pairs = [(q, Q[q['id'][:-3] + 'rev']) for q in Q.values() if q['order'] == 'fwd']
    o['pairs_both_correct'] = sum(pred(R[f['id']]) == f['answer'] and pred(R[r['id']]) == r['answer'] for f, r in pairs)
    o['identical_pairs'] = dict(Counter(pred(R[q['id']]) for q in Q.values() if q['order'] == 'same'))
    o['all_scored'] = dict(Counter(pred(R[q['id']]) for q in Q.values() if q['answer']))
    out[name] = o
out['unparsed'] = sum(r['parsed'] is None for r in R.values())
out['answer_texts'] = dict(Counter(r['text'] for r in R.values()))
print(json.dumps(out, indent=1))
(HERE / sys.argv[2]).write_text(json.dumps(out, indent=1))
