"""Diagnostic (not pre-registered): run the Habitat-adapted UniWM on its own training trajectories.

Image task: for each selected training row, generate the next view under the recorded action, the zero action
and the sign-negated action (same seed, same images). Action task: with the native action prompt and the same
start/goal/current images, predict the next action (greedy + 4 samples) and compare with the recorded one.
Resumable: every record is written atomically and skipped when present.
"""
import argparse
import json
import re
import sys
from pathlib import Path
import torch
from suite_common import read, atomic, digest

OUT = Path('external/habitat-yaw-reference/v2/uniwm_train_probe')
EXP = Path('external/habitat-yaw-reference/v2/uniwm_symmetric_aligned_v3')
SEED = 20261118


def kind(a):
    dx, dy, dyaw = a
    if abs(dx) < 1e-6 and abs(dy) < 1e-6 and dyaw > .01: return 'left'
    if abs(dx) < 1e-6 and abs(dy) < 1e-6 and dyaw < -.01: return 'right'
    if abs(dyaw) <= .01 and (abs(dx) > 1e-6 or abs(dy) > 1e-6): return 'translation'
    return 'other'


def select(rows):
    chosen = []
    for k, n in [('left', 16), ('right', 16), ('translation', 8)]:
        # Spread over yaw magnitudes (3/5/7 deg) and trajectories; skip first frames, where goal == current.
        pool = sorted([r for r in rows if kind(r['action']) == k and r['current_index'] > 0],
                      key=lambda r: (round(abs(r['action'][2]), 3), r['sample_id']))
        step = len(pool) / n
        chosen += [dict(r, kind=k) for r in (pool[int(i * step)] for i in range(n))]
    return chosen


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--smoke', action='store_true'); a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rows = read(EXP / 'train.json'); done = read(EXP / 'training/complete.json'); assert done['status'] == 'passed'
    weight = EXP / 'training/final_adapter'
    assert digest(weight / 'adapter_model.safetensors') == done['adapter_sha256']
    samples = select(rows)
    if not (OUT / 'samples.json').exists():
        atomic(OUT / 'samples.json', dict(weight=str(weight), adapter_sha256=done['adapter_sha256'], seed=SEED,
               samples=[s['sample_id'] for s in samples], kinds={s['sample_id']: s['kind'] for s in samples}))
    assert read(OUT / 'samples.json')['samples'] == [s['sample_id'] for s in samples]
    if a.smoke: samples = samples[:1] + samples[16:17]
    from suite_backends import UniWMBackend, seed_all
    from uniwm_v3_common import action_text
    backend = UniWMBackend(dict(checkpoint=str(weight)), None)  # puts the UniWM repo on sys.path
    from scripts.prompt_builder import build_action_prompt
    from scripts.action_utils import DATASET_RANGES, extract_bin_values
    backend.model.eval(); torch.set_num_threads(2)
    rng = DATASET_RANGES['1x_nav']; tok = backend.processor.tokenizer
    for i, s in enumerate(samples):
        sid = s['sample_id']; base = OUT / 'image' / sid; base.mkdir(parents=True, exist_ok=True)
        rec = list(s['action']); variants = dict(recorded=rec, zero=[0., 0., 0.], negated=[rec[0], rec[1], -rec[2]] if s['kind'] != 'translation' else [-rec[0], -rec[1], rec[2]])
        for name, act in variants.items():
            meta = base / f'{name}.json'
            if meta.exists(): continue
            case = dict(history_paths=s['history_paths'], start_image=s['start_image'], goal_image=s['goal_image'],
                        action=act, action_text=action_text(act), start_pose=s['start_pose'], memory_entries=[],
                        memory_mode='none', current_index=s['current_index'], seed=SEED)
            with torch.inference_mode(): im, _, info = backend.predict(case, lambda p: Path(p))
            info.pop('_anchor', None); png = base / f'{name}.png'; tmp = png.with_suffix('.tmp'); im.save(tmp, format='PNG'); tmp.replace(png)
            atomic(meta, dict(sample_id=sid, variant=name, action=act, action_text=case['action_text'], image_sha256=digest(png), **info))
            print('PROBE_IMAGE', i, sid, name, flush=True)
        ameta = OUT / 'action' / f'{sid}.json'
        if ameta.exists(): continue
        ameta.parent.mkdir(parents=True, exist_ok=True)
        current = backend.sanitize(Path(s['history_paths'][-1]), False)
        from suite_backends import pil
        start, goal = pil(Path(s['start_image']), (256, 256)), pil(Path(s['goal_image']), (256, 256))
        sp = s['start_pose']; pose = f'Starting Point Coordinate: x={sp[0]:.3f}, y={sp[1]:.3f}, yaw={sp[2]:.3f}\n'
        prompt = build_action_prompt(pose, rng['dxy'], rng['dyaw'])
        inputs = backend.processor(text=[prompt], images=[start, goal, current], return_tensors='pt', padding=False, truncation=False).to(backend.model.device)
        n_in = inputs['input_ids'].shape[1]; outs = []
        for j, sampling in enumerate([dict(do_sample=False)] + [dict(do_sample=True, temperature=.7, top_p=.9)] * 4):
            seed_all(SEED + j)
            with torch.inference_mode(), torch.autocast(device_type='cuda', dtype=torch.float16):
                row = backend.model.generate(**inputs, num_beams=1, max_new_tokens=40, **sampling)[0]
            if row.ndim == 2: row = row[0]  # UniWM's generate returns (tokens[batch, new], ...), as in suite_backends
            ids = inputs['input_ids'][0]
            if len(row) > n_in and torch.equal(row[:n_in].to(ids.device), ids): row = row[n_in:]
            text = re.sub(r'<reserved\d+>|</s>|<pad>', '', tok.decode(row, skip_special_tokens=False)).strip()
            vals = [extract_bin_values(text, k, .01) for k in ('dx', 'dy', 'dyaw')]
            ok = all(re.search(rf'<{k}_(pos|neg)_bin_\d+>', text) for k in ('dx', 'dy', 'dyaw'))
            outs.append(dict(mode='greedy' if j == 0 else 'sample', text=text, values=vals if ok else None,
                             stop=text.lower().startswith('stop'), well_formed=ok))
        atomic(ameta, dict(sample_id=sid, kind=s['kind'], recorded=rec, recorded_text=action_text(rec), outputs=outs))
        print('PROBE_ACTION', i, sid, outs[0]['values'], rec, flush=True)
    if a.smoke: atomic(OUT / 'smoke.json', dict(status='passed', samples=[s['sample_id'] for s in samples])); return
    atomic(OUT / 'complete.json', dict(status='passed', samples=len(samples), generations=3 * len(samples)))
    print('PROBE_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
