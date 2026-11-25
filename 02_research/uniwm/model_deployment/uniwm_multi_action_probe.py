"""Diagnostic: training images that appear with several recorded actions (start/return views).

For 6 such images, generate the next view under every recorded action of that image (same seed), and score each
output's horizontal shift against its own target. Here the image alone cannot determine the target.
"""
import collections
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from suite_common import read, atomic, digest

OUT = Path('external/habitat-yaw-reference/v2/uniwm_train_probe/multi_action')
EXP = Path('external/habitat-yaw-reference/v2/uniwm_symmetric_aligned_v3')
SEED = 20261118; S = 256


def xshift(a, b):
    w = np.outer(np.hanning(S), np.hanning(S)); A = np.fft.fft2((a - a.mean()) * w); B = np.fft.fft2((b - b.mean()) * w)
    R = B * np.conj(A); R /= np.abs(R) + 1e-12; r = np.fft.ifft2(R).real
    y, x = np.unravel_index(np.argmax(r), r.shape); return float(x if x < S // 2 else x - S)


def gray(p):
    return np.asarray(Image.open(p).convert('RGB').resize((S, S), Image.BICUBIC), dtype=np.float64).mean(2)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = read(EXP / 'train.json'); done = read(EXP / 'training/complete.json'); weight = EXP / 'training/final_adapter'
    assert digest(weight / 'adapter_model.safetensors') == done['adapter_sha256']
    h = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
    groups = collections.defaultdict(list)
    for r in rows: groups[h(r['history_paths'][-1])].append(r)
    multi = sorted([g for g in groups.values() if len({tuple(round(v, 4) for v in r['action']) for r in g}) > 1],
                   key=lambda g: g[0]['sample_id'])
    pick = multi[::max(1, len(multi) // 6)][:6]
    atomic(OUT / 'selection.json', dict(multi_action_images=len(multi), single_action_images=len(groups) - len(multi),
           rows_in_multi=sum(len(g) for g in multi), picked=[[r['sample_id'] for r in g] for g in pick]))
    from suite_backends import UniWMBackend
    from uniwm_v3_common import action_text
    backend = UniWMBackend(dict(checkpoint=str(weight)), None); backend.model.eval(); torch.set_num_threads(2)
    res = []
    for gi, g in enumerate(pick):
        for r in g:
            png = OUT / f"{r['sample_id']}.png"
            if not png.exists():
                case = dict(history_paths=r['history_paths'], start_image=r['start_image'], goal_image=r['goal_image'],
                            action=r['action'], action_text=action_text(r['action']), start_pose=r['start_pose'],
                            memory_entries=[], memory_mode='none', current_index=r['current_index'], seed=SEED)
                with torch.inference_mode(): im, _, _ = backend.predict(case, lambda p: Path(p))
                tmp = png.with_suffix('.tmp'); im.save(tmp, format='PNG'); tmp.replace(png)
            cur = gray(r['history_paths'][-1])
            res.append(dict(group=gi, sample_id=r['sample_id'], action=r['action'], image_sha256=digest(png),
                            truth_shift_px=xshift(cur, gray(r['target_image'])), gen_shift_px=xshift(cur, gray(png))))
            print('MULTI', gi, r['sample_id'], [round(v, 3) for v in r['action']], res[-1]['truth_shift_px'], res[-1]['gen_shift_px'], flush=True)
    summ = []
    for gi in range(len(pick)):
        rs = [x for x in res if x['group'] == gi]
        summ.append(dict(group=gi, n=len(rs), distinct_outputs=len({x['image_sha256'] for x in rs}),
                         truth_shifts=[x['truth_shift_px'] for x in rs], gen_shifts=[x['gen_shift_px'] for x in rs],
                         corr=float(np.corrcoef([x['truth_shift_px'] for x in rs], [x['gen_shift_px'] for x in rs])[0, 1])
                         if len({x['gen_shift_px'] for x in rs}) > 1 else None))
    atomic(OUT / 'summary.json', dict(rows=res, groups=summ))
    print(json.dumps(summ, indent=1), flush=True)


if __name__ == '__main__':
    main()
