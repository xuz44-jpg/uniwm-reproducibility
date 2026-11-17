"""Derived CPU analysis of existing frozen query records (2026-11-10 revision).

Adds, without new predictions or changed labels:
  * direction labels split by command sign (+ = left turn, - = right turn);
  * the zero-response level of raw E_M (mean reference-effect magnitude on M);
  * raw-gain sign counts by command sign;
  * labels under the post-hoc 80% coverage gate, split by sign;
  * exact two-sided paired tests (query-level McNemar, position-level sign test).
All tests are exploratory: the history contrast was chosen after the static confirmation and
uses one paired seed at twelve already inspected positions.
"""
import json
from math import comb
from pathlib import Path
from statistics import mean, median

BASE = Path(__file__).resolve().parent
DATA = BASE / 'data'
LABEL = {'correct': 'A', 'reverse': 'R', 'near_zero': 'Z', 'unknown': 'U'}
HISTORIES = ['static', 'left_approach', 'right_approach']
WEIGHTS = ['pre_habitat', 'adapted']


def exact_two_sided(k, n):
    if n == 0:
        return 1.0
    lo = min(k, n - k)
    return min(1.0, 2 * sum(comb(n, i) for i in range(lo + 1)) / 2 ** n)


def counts(labels):
    return {x: sum(l == x for l in labels) for x in 'ARZU'}


def sign_of(angle):
    return 'plus' if angle > 0 else 'minus'


context = [r for r in json.loads((DATA / 'context_nwm_rows.json').read_text()) if r['angle'] != 0]
assert len(context) == 144
by_arm = {}
for w in WEIGHTS:
    for h in HISTORIES:
        rows = [r for r in context if r['weight_arm'] == w and r['context_arm'] == h]
        assert len(rows) == 24
        lab = [LABEL[r['direction']['label']] for r in rows]
        gate80 = []
        for r, l in zip(rows, lab):
            cov = r['direction'].get('coverage')
            gate80.append('U' if l == 'U' or cov is None or cov < 0.8 else l)
        entry = {'all': counts(lab), 'gate80_all': counts(gate80)}
        for s in ('plus', 'minus'):
            idx = [i for i, r in enumerate(rows) if sign_of(r['angle']) == s]
            gains = [rows[i]['effect']['gain_projection'] for i in idx]
            entry[s] = {'labels': counts([lab[i] for i in idx]),
                        'gate80': counts([gate80[i] for i in idx]),
                        'raw_gain_positive': sum(g > 0 for g in gains), 'raw_gain_median': median(gains),
                        'n': len(idx)}
        entry['raw_epe_mean_px'] = mean(r['effect']['mean_epe_px'] for r in rows)
        entry['zero_response_epe_px'] = mean(r['effect']['reference']['mean_magnitude'] for r in rows)
        by_arm[f'{w}/{h}'] = entry

# The zero-response level depends only on position/command, so it is identical across arms.
levels = {round(v['zero_response_epe_px'], 6) for v in by_arm.values()}
assert len(levels) == 1, levels

tests = {}
for w in WEIGHTS:
    for h in ('left_approach', 'right_approach'):
        ref = {(r['state_id'], r['angle']): LABEL[r['direction']['label']] for r in context
               if r['weight_arm'] == w and r['context_arm'] == 'static'}
        new = {(r['state_id'], r['angle']): LABEL[r['direction']['label']] for r in context
               if r['weight_arm'] == w and r['context_arm'] == h}
        assert set(ref) == set(new) and len(ref) == 24
        gained = sum(ref[k] != 'A' and new[k] == 'A' for k in ref)
        lost = sum(ref[k] == 'A' and new[k] != 'A' for k in ref)
        pos = {}
        for (state, _), l in ref.items():
            pos.setdefault(state, 0)
            pos[state] -= l == 'A'
        for (state, _), l in new.items():
            pos[state] += l == 'A'
        up = sum(v > 0 for v in pos.values())
        down = sum(v < 0 for v in pos.values())
        tests[f'{w}/static->{h}'] = {'queries_gained_A': gained, 'queries_lost_A': lost,
                                     'mcnemar_exact_p': exact_two_sided(gained, gained + lost),
                                     'positions_up': up, 'positions_down': down, 'positions_tied': 12 - up - down,
                                     'position_sign_test_p': exact_two_sided(up, up + down)}

approach = {s: {'A': 0, 'R': 0} for s in ('plus', 'minus')}
for w in WEIGHTS:
    for h in ('left_approach', 'right_approach'):
        for s in ('plus', 'minus'):
            for x in 'AR':
                approach[s][x] += by_arm[f'{w}/{h}'][s]['labels'][x]

confirmation = {}
for model in ('nwm', 'raenwm', 'uniwm'):
    rows = [r for r in json.loads((DATA / f'confirmation_{model}_rows.json').read_text()) if r['angle'] != 0]
    assert len(rows) == 216
    entry = {}
    for s in ('plus', 'minus'):
        sel = [r for r in rows if sign_of(r['angle']) == s]
        entry[s] = counts([LABEL[r['action_direction']['label']] for r in sel])
        entry[s + '_by_magnitude'] = {str(a): counts([LABEL[r['action_direction']['label']] for r in sel
                                                      if abs(r['angle']) == a]) for a in (3, 5, 7)}
    confirmation[model] = entry

scale = json.loads((DATA / 'angle_scale_summary.json').read_text())['models']['nwm']['by_signed_angle']
eleven_angle = {a: {LABEL[k]: scale[a]['direction'][k] for k in LABEL} for a in
                ('3', '-3', '5', '-5', '7', '-7', '10', '-10', '15', '-15')}

out = {'status': 'derived_from_frozen_records', 'date': '2026-11-10',
       'sign_convention': 'plus = positive yaw = left turn; minus = right turn',
       'new_predictions': 0, 'labels_changed': 0,
       'context_nwm': by_arm, 'zero_response_epe_px': levels.pop(),
       'paired_tests_exploratory': tests, 'approach_history_A_R_by_sign': approach,
       'confirmation_by_sign': confirmation, 'eleven_angle_mask_nwm_by_signed_angle': eleven_angle,
       'sources': {n: __import__('hashlib').sha256((DATA / n).read_bytes()).hexdigest() for n in
                   ['context_nwm_rows.json', 'confirmation_nwm_rows.json', 'confirmation_raenwm_rows.json',
                    'confirmation_uniwm_rows.json', 'angle_scale_summary.json']}}
(DATA / 'sign_breakdown_summary.json').write_text(json.dumps(out, indent=2) + '\n')
print(json.dumps({k: out[k] for k in ('zero_response_epe_px', 'paired_tests_exploratory', 'approach_history_A_R_by_sign')}, indent=1))
for k, v in by_arm.items():
    print(k, v['all'], '+', v['plus']['labels'], '-', v['minus']['labels'], 'gate80', v['gate80_all'],
          'gain+ %d/12 %d/12' % (v['plus']['raw_gain_positive'], v['minus']['raw_gain_positive']),
          'med %.3f %.3f' % (v['plus']['raw_gain_median'], v['minus']['raw_gain_median']), 'EPE %.2f' % v['raw_epe_mean_px'])
for m, v in confirmation.items():
    print(m, '+', v['plus'], '-', v['minus'])
