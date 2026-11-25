"""CPU analysis of uniwm_train_probe: image response (identity, LPIPS, phase-correlation shift) and action accuracy."""
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

OUT = Path('external/habitat-yaw-reference/v2/uniwm_train_probe')
EXP = Path('external/habitat-yaw-reference/v2/uniwm_symmetric_aligned_v3')
S = 256


def load(p):
    return Image.open(p).convert('RGB').resize((S, S), Image.BICUBIC)


def gray(im):
    return np.asarray(im, dtype=np.float64).mean(2)


def xshift(a, b):
    """Horizontal shift (px) of b relative to a by phase correlation on a Hann-windowed image."""
    w = np.outer(np.hanning(S), np.hanning(S)); A = np.fft.fft2((a - a.mean()) * w); B = np.fft.fft2((b - b.mean()) * w)
    R = B * np.conj(A); R /= np.abs(R) + 1e-12; r = np.fft.ifft2(R).real
    y, x = np.unravel_index(np.argmax(r), r.shape)
    return float(x if x < S // 2 else x - S), float(r.max())


def main():
    torch.set_num_threads(2)
    import lpips
    net = lpips.LPIPS(net='alex', verbose=False)
    T = lambda im: torch.tensor(np.asarray(im) / 127.5 - 1).permute(2, 0, 1)[None].float()
    L = lambda a, b: float(net(T(a), T(b)).detach())
    rows = {r['sample_id']: r for r in json.load(open(EXP / 'train.json'))}
    meta = json.load(open(OUT / 'samples.json')); out = []
    for sid in meta['samples']:
        r = rows[sid]; k = meta['kinds'][sid]; d = OUT / 'image' / sid
        cur, tgt = load(r['history_paths'][-1]), load(r['target_image'])
        gens = {v: load(d / f'{v}.png') for v in ['recorded', 'zero', 'negated']}
        sha = {v: json.load(open(d / f'{v}.json'))['image_sha256'] for v in gens}
        truth = xshift(gray(cur), gray(tgt))[0]
        shifts = {v: xshift(gray(cur), gray(g))[0] for v, g in gens.items()}
        act = json.load(open(OUT / 'action' / f'{sid}.json'))
        rec = act['recorded']; g0 = act['outputs'][0]['values']; samples = [o['values'] for o in act['outputs'][1:]]
        sgn = lambda x: 0 if abs(x) < .005 else (1 if x > 0 else -1)
        g = g0 if g0 is not None else [float('nan')] * 3
        out.append(dict(sample_id=sid, kind=k, recorded=rec, truth_shift_px=truth, gen_shift_px=shifts,
            recorded_equals_zero=sha['recorded'] == sha['zero'], recorded_equals_negated=sha['recorded'] == sha['negated'],
            lpips_recorded_target=L(gens['recorded'], tgt), lpips_copy_target=L(cur, tgt), lpips_recorded_current=L(gens['recorded'], cur),
            lpips_recorded_zero=L(gens['recorded'], gens['zero']), lpips_recorded_negated=L(gens['recorded'], gens['negated']),
            action_greedy=g, action_greedy_text=act['outputs'][0]['text'], action_samples=samples,
            greedy_well_formed=g0 is not None, greedy_stop=act['outputs'][0].get('stop', False),
            greedy_dyaw_sign_correct=(g0 is not None and sgn(g[2]) == sgn(rec[2])) if k != 'translation' else None,
            sample_dyaw_sign_correct=sum(s is not None and sgn(s[2]) == sgn(rec[2]) for s in samples) if k != 'translation' else None,
            sample_dyaw_signs=[None if s is None else sgn(s[2]) for s in samples],
            greedy_abs_err=[abs(a - b) for a, b in zip(g, rec)]))
    summ = {}
    for k in ['left', 'right', 'translation']:
        rs = [x for x in out if x['kind'] == k]; m = lambda f: float(np.mean([f(x) for x in rs]))
        summ[k] = dict(n=len(rs), recorded_equals_zero=sum(x['recorded_equals_zero'] for x in rs),
            recorded_equals_negated=sum(x['recorded_equals_negated'] for x in rs),
            lpips_recorded_target=m(lambda x: x['lpips_recorded_target']), lpips_copy_target=m(lambda x: x['lpips_copy_target']),
            lpips_recorded_vs_zero=m(lambda x: x['lpips_recorded_zero']), lpips_recorded_vs_negated=m(lambda x: x['lpips_recorded_negated']),
            truth_shift_px=m(lambda x: x['truth_shift_px']), recorded_shift_px=m(lambda x: x['gen_shift_px']['recorded']),
            zero_shift_px=m(lambda x: x['gen_shift_px']['zero']), negated_shift_px=m(lambda x: x['gen_shift_px']['negated']),
            recorded_shift_sign_matches_truth=sum(np.sign(x['gen_shift_px']['recorded']) == np.sign(x['truth_shift_px']) and x['gen_shift_px']['recorded'] != 0 for x in rs),
            greedy_dyaw_sign_correct=sum(bool(x['greedy_dyaw_sign_correct']) for x in rs) if k != 'translation' else None,
            sample_dyaw_sign_correct=sum(x['sample_dyaw_sign_correct'] for x in rs) if k != 'translation' else None,
            greedy_mean_abs_err=[float(np.nanmean([x['greedy_abs_err'][i] for x in rs])) if any(x['greedy_well_formed'] for x in rs) else None for i in range(3)],
            greedy_well_formed=sum(x['greedy_well_formed'] for x in rs), greedy_stop=sum(x['greedy_stop'] for x in rs),
            greedy_dyaw_sign_counts={str(v): sum(1 for x in rs if x['greedy_well_formed'] and np.sign(round(x['action_greedy'][2], 3)) == v) for v in (-1, 0, 1)})
    conv = lambda o: o.item() if hasattr(o, 'item') else str(o)
    json.dump(dict(rows=out, summary=summ), open(OUT / 'analysis.json', 'w'), indent=1, default=conv)
    print(json.dumps(summ, indent=1, default=conv))
    # Contact sheet: 4 left, 4 right, 2 translation; columns current | target | recorded | zero | negated.
    pick = [x for x in out if x['kind'] == 'left'][::4] + [x for x in out if x['kind'] == 'right'][::4] + [x for x in out if x['kind'] == 'translation'][::4]
    cell, pad, head = 200, 6, 40; W = 5 * (cell + pad) + 300; H = head + len(pick) * (cell + pad)
    sheet = Image.new('RGB', (W, H), 'white'); dr = ImageDraw.Draw(sheet)
    try: font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 15)
    except OSError: font = ImageFont.load_default()
    for j, t in enumerate(['current', 'target (truth)', 'gen: recorded', 'gen: zero', 'gen: negated']): dr.text((j * (cell + pad) + 4, 12), t, fill='black', font=font)
    dr.text((5 * (cell + pad) + 4, 12), 'action: recorded / predicted', fill='black', font=font)
    for i, x in enumerate(pick):
        r = rows[x['sample_id']]; d = OUT / 'image' / x['sample_id']; y = head + i * (cell + pad)
        ims = [r['history_paths'][-1], r['target_image'], d / 'recorded.png', d / 'zero.png', d / 'negated.png']
        for j, p in enumerate(ims): sheet.paste(Image.open(p).convert('RGB').resize((cell, cell)), (j * (cell + pad), y))
        rec, g = x['recorded'], x['action_greedy']
        txt = (f"{x['sample_id']}\n{x['kind']}\nrec  dx {rec[0]:+.2f} dy {rec[1]:+.2f}\n     dyaw {rec[2]:+.3f}\n"
               f"pred dx {g[0]:+.2f} dy {g[1]:+.2f}\n     dyaw {g[2]:+.3f}\nshift true {x['truth_shift_px']:+.0f}px\n"
               f"gen {x['gen_shift_px']['recorded']:+.0f}/{x['gen_shift_px']['zero']:+.0f}/{x['gen_shift_px']['negated']:+.0f}px\n"
               f"rec==zero {x['recorded_equals_zero']}")
        dr.multiline_text((5 * (cell + pad) + 4, y + 2), txt, fill='black', font=font, spacing=3)
    sheet.save(OUT / 'contact_sheet.png')
    print('PROBE_ANALYZED', flush=True)


if __name__ == '__main__':
    main()
