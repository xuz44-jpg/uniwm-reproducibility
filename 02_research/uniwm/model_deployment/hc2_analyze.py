"""Pre-registered analysis (PROTOCOL.md): primary McNemar tests with Holm correction, then secondary summaries."""
import argparse
from collections import defaultdict
from math import comb
from pathlib import Path
import numpy as np
from suite_common import read, digest, atomic
from hc2_common import root, HISTORIES, WEIGHTS

GATE80 = 0.8


def upper_tail(k, n):
    """P(X >= k) for X ~ Binomial(n, 1/2); 1.0 when n == 0."""
    return 1.0 if n == 0 else sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n


def holm(pvalues):
    order = sorted(pvalues, key=pvalues.get); adjusted = {}; running = 0.
    for i, key in enumerate(order):
        running = max(running, min(1., (len(order) - i) * pvalues[key])); adjusted[key] = running
    return adjusted


def mean(xs):
    xs = [x for x in xs if x is not None]; return float(np.mean(xs)) if xs else None


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--study', type=Path, required=True); a = ap.parse_args()
    out = root(a.study); done = read(out / 'generation/analysis/complete.json'); assert done['status'] == 'passed'
    rows = read(out / 'generation/analysis/rows.json'); assert digest(out / 'generation/analysis/rows.json') == done['rows_sha256']
    cal = read(out / 'flow/references/summary.json')['states']
    nz = [r for r in rows if r['angle']]; zero = [r for r in rows if not r['angle']]
    label = lambda r: r['direction']['label']
    a80 = lambda r: 'unknown' if label(r) == 'correct' and r['direction']['coverage'] < GATE80 else label(r)
    idx = {(r['weight_arm'], r['context_arm'], r['state_id'], r['angle'], r['repeat']): r for r in nz}
    pair_keys = sorted({(r['state_id'], r['angle'], r['repeat']) for r in nz})
    assert len(pair_keys) == 72

    def counts(rs, fn=label):
        c = {k: sum(fn(r) == k for r in rs) for k in ['correct', 'reverse', 'near_zero', 'unknown']}
        return dict(A=c['correct'], R=c['reverse'], Z=c['near_zero'], U=c['unknown'], n=len(rs))

    arms = {}
    for w in WEIGHTS:
        for h in HISTORIES:
            rs = [r for r in nz if r['weight_arm'] == w and r['context_arm'] == h]
            zs = [r for r in zero if r['weight_arm'] == w and r['context_arm'] == h]
            arms[f'{w}/{h}'] = dict(all=counts(rs), gate80=counts(rs, a80),
                A_plus=sum(label(r) == 'correct' for r in rs if r['angle'] > 0),
                A_minus=sum(label(r) == 'correct' for r in rs if r['angle'] < 0),
                by_seed={s: counts([r for r in rs if r['repeat'] == s]) for s in range(3)},
                by_scene={sc: counts([r for r in rs if r['scene_id'] == sc]) for sc in sorted({r['scene_id'] for r in rs})},
                raw_epe_px=mean([r['effect']['mean_epe_px'] for r in rs]),
                gain=mean([r['effect']['gain_projection'] for r in rs]),
                zero_drift_px=mean([r['zero_drift']['mean_magnitude'] for r in zs]),
                same_as_zero=sum(r['same_as_zero'] for r in rs))
    primary = {}; secondary = {}
    for w in WEIGHTS:
        for h in ['left_approach', 'right_approach']:
            b = c = 0
            for sid, ang, rep in pair_keys:
                rot = label(idx[(w, h, sid, ang, rep)]) == 'correct'; sta = label(idx[(w, 'static', sid, ang, rep)]) == 'correct'
                b += rot and not sta; c += sta and not rot
            primary[f'{w}/{h}'] = dict(rotating_only=b, static_only=c, p_one_sided=upper_tail(b, b + c),
                A_rotating=arms[f'{w}/{h}']['all']['A'], A_static=arms[f'{w}/static']['all']['A'])
            up = down = 0
            for sid in sorted({k[0] for k in pair_keys}):
                ar = sum(label(idx[(w, h, sid, ang, rep)]) == 'correct' for _, ang, rep in [k for k in pair_keys if k[0] == sid])
                ast = sum(label(idx[(w, 'static', sid, ang, rep)]) == 'correct' for _, ang, rep in [k for k in pair_keys if k[0] == sid])
                up += ar > ast; down += ar < ast
            secondary[f'{w}/{h}'] = dict(positions_up=up, positions_down=down, sign_test_p_one_sided=upper_tail(up, up + down))
    adjusted = holm({k: v['p_one_sided'] for k, v in primary.items()})
    for k in primary: primary[k]['p_holm'] = adjusted[k]
    significant = sum(v['p_holm'] < .05 for v in primary.values())
    verdict = 'replicated' if significant == 4 else 'partially replicated' if significant else 'not replicated'
    summary = dict(status='passed', protocol_sha256=digest(out / 'protocol.json'), verdict=verdict,
                   significant_primary=significant, primary=primary, position_sign_tests=secondary, arms=arms,
                   reference_gates={sid: s['reference_gate_passed'] for sid, s in cal.items()},
                   note='queries within a position are not independent; the position-level test is the conservative one')
    atomic(out / 'summary.json', summary)
    print('HC2_VERDICT', verdict, {k: (v['A_static'], v['A_rotating'], round(v['p_holm'], 4)) for k, v in primary.items()}, flush=True)


if __name__ == '__main__':
    main()
