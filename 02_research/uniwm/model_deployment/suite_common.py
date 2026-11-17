"""Immutable manifests and atomic resumable artifacts for inference only."""
import hashlib
import json
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def atomic(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def frozen(path, obj):
    path = Path(path)
    if path.exists():
        assert read(path) == obj, f'Immutable configuration changed: {path}'
    else:
        atomic(path, obj)


def stable_seed(state_seed, step=0):
    # Condition/history deliberately absent: all controllable RNGs are paired.
    return (int(state_seed) + 1000003 * int(step)) % (2**31 - 1)


def cell_key(state, experiment, history, repeat, step, condition):
    return f'{state}/{experiment}/{history}/r{repeat}/step{step:02d}_{condition}'


def verify_record(out, case):
    path = Path(out) / (case['key'] + '.json')
    if not path.exists():
        return None
    r = read(path)
    assert r['case'] == case, str(path)
    if r['status'] == 'ok':
        assert digest(path.with_suffix('.png')) == r['image_sha256']
        if r.get('anchor_sha256'):
            assert digest(path.with_suffix('.anchor.png')) == r['anchor_sha256']
        if r.get('feedback_sha256'):
            assert digest(path.with_suffix('.pt')) == r['feedback_sha256']
        for source, sha in r.get('input_hashes', {}).items():
            assert digest(Path(source)) == sha, source
    else:
        assert r['status'] in ['parse_failed', 'blocked_dependency']
    return r
