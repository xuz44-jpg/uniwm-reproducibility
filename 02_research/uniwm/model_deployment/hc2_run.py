"""Resumable frozen-weight NWM inference for the confirmation; mirrors run_context_controls_v1.py (NWM path)."""
import argparse
from collections import defaultdict
import gc
from pathlib import Path
import numpy as np
import torch
from suite_common import read, digest, atomic, frozen
from hc2_common import root, WEIGHTS


def thash(x):
    import hashlib
    return hashlib.sha256(x.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--study', type=Path, required=True)
    ap.add_argument('--root', type=Path, required=True); ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args(); out = root(a.study); freeze = read(out / 'execution_freeze.json')
    for path, sha in freeze['source_hashes'].items(): assert digest(path) == sha, path
    assert digest(out / 'protocol.json') == freeze['protocol_sha256']
    m = read(out / 'cases_nwm.json'); cases = m['cases']; weights = read(out / 'protocol.json')['weights']
    dest = out / 'generation'; dest.mkdir(parents=True, exist_ok=True)
    frozen(dest / 'config.json', dict(manifest_sha256=digest(out / 'cases_nwm.json'),
           execution_sha256=digest(out / 'execution_freeze.json'), weights=weights, updates=0))
    records = {}
    for c in cases:
        path = dest / (c['key'] + '.json')
        if path.exists():
            r = read(path); assert r['case'] == c
            if r['status'] == 'ok': assert digest(path.with_suffix('.png')) == r['image_sha256']
            records[c['key']] = r
    from suite_backends import ReConBackend
    from train_habitat_model import CHECKPOINTS
    first_sid = cases[0]['state_id']
    for arm in WEIGHTS:
        arm_cases = [c for c in cases if c['weight_arm'] == arm]
        if a.smoke: arm_cases = [c for c in arm_cases if c['state_id'] == first_sid and c['repeat'] == 0]
        pending = [c for c in arm_cases if c['key'] not in records]
        if not pending: continue
        weight = weights[arm]; p = Path(weight['path']); assert digest(p) == weight['sha256']
        backend = ReConBackend(dict(model='nwm', checkpoint=CHECKPOINTS['nwm']), a.root)
        if arm == 'adapted':
            state = torch.load(p, map_location='cpu', weights_only=False)
            actual = dict(backend.model.named_parameters()); expected = {n for n, v in actual.items() if v.requires_grad}
            assert set(state['trainable']) == expected
            missing, unexpected = backend.model.load_state_dict(state['trainable'], strict=False)
            assert not unexpected and not set(missing) & expected
            for n, v in state['trainable'].items(): assert torch.equal(actual[n].detach().cpu(), v)
            audit = dict(status='passed', trainable_tensors=len(expected), weight=weight); del state
        else:
            assert str(p) == CHECKPOINTS['nwm']
            audit = dict(status='passed', strict_released_ema_load=True, weight=weight)
        backend.model.eval().requires_grad_(False); torch.set_num_threads(2)
        atomic(dest / f'loaded_{arm}.json', audit)
        for c in pending:
            path = dest / (c['key'] + '.png'); path.parent.mkdir(parents=True, exist_ok=True)
            capture = {}; input_hashes = {}
            allowed = set(c['history_paths'] + [c['start_image'], c['goal_image']])

            def resolve(q):
                assert q in allowed, 'Unregistered image or target passed to model'
                assert digest(q) == m['input_hashes'][q]; input_hashes[q] = digest(q); return Path(q)

            def before(module, args, kw):
                if not capture:
                    capture.update(conditioning_sha256=thash(kw['x_cond']), initial_noise_sha256=thash(args[0]),
                                   observed_action=kw['y'].detach().float().cpu().tolist(),
                                   observed_rel_t=kw['rel_t'].detach().float().cpu().tolist())
            hook = backend.model.register_forward_pre_hook(before, with_kwargs=True)
            try:
                with torch.inference_mode(): im, feedback, info = backend.predict(c, resolve)
                info.pop('_anchor', None); del feedback
                tmp = path.with_suffix('.tmp'); im.save(tmp, format='PNG'); tmp.replace(path)
                assert np.allclose(capture['observed_action'], [backend.encode_action(c['action'])], rtol=0, atol=1e-7)
                assert abs(capture['observed_rel_t'][0] - c['horizon_frames'] / 128) < 1e-9
                r = dict(status='ok', case=c, image_sha256=digest(path), origin='new', actual_input_hashes=input_hashes, **info, **capture)
            finally:
                hook.remove()
            atomic(path.with_suffix('.json'), r); records[c['key']] = r
            atomic(dest / 'progress.json', dict(records=len(records), total=len(cases), last=c['key']))
            print('HC2_GENERATION', len(records), len(cases), c['key'], flush=True)
        del backend; gc.collect(); torch.cuda.empty_cache()
    if a.smoke:
        smoke = [c for c in cases if c['state_id'] == first_sid and c['repeat'] == 0]
        assert all(records[c['key']]['status'] == 'ok' for c in smoke)
        atomic(dest / 'smoke.json', dict(status='passed', checked_cases=len(smoke), records_in_full_budget=True)); return
    assert len(records) == len(cases)
    groups = defaultdict(list)
    for r in records.values():
        c = r['case']; groups[(c['weight_arm'], c['context_arm'], c['state_id'], c['repeat'])].append(r)
    for g in groups.values():
        assert len(g) == 3
        for k in ['conditioning_sha256', 'initial_noise_sha256']: assert len({r[k] for r in g}) == 1, k
    atomic(dest / 'complete.json', dict(status='passed', records=len(records),
           record_hashes={k: digest(dest / (k + '.json')) for k in records}, world_model_updates=0))


if __name__ == '__main__':
    main()
