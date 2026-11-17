"""Freeze a development-only, source-disjoint UniWM-1X pilot without reading predictions."""
import argparse
import hashlib
import json
import pickle
import random
import sys
from pathlib import Path

import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', type=Path, required=True)
    ap.add_argument('--data', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    sys.path.insert(0, str(args.repo.resolve()))
    from scripts.action_utils import action_to_text
    names = sorted(p.name for p in args.data.iterdir() if p.is_dir())
    assert len(names) == 510, 'Re-audit the baseline split before using another dataset version'
    shuffled = names.copy()
    random.Random(42).shuffle(shuffled)
    meta = {}
    for name in names:
        with (args.data / name / 'traj_data.pkl').open('rb') as f:
            meta[name] = pickle.load(f)
    group = lambda name: (int(meta[name]['shard']), int(meta[name]['segment']))
    train = shuffled[:200]
    train_groups = {group(name) for name in train}
    clean = [name for name in names if group(name) not in train_groups]
    candidates = {1: [], -1: []}
    for name in clean:
        m = meta[name]
        for k in range(2, len(m['delta']) - 2):
            action = np.asarray(m['delta'][k], dtype=float)[:3]
            if not np.isfinite(action).all() or not (0.05 <= abs(action[2]) <= 0.20):
                continue
            if max(abs(action[0]), abs(action[1])) > 0.30:
                continue
            variants = {
                'original': action.tolist(), 'zero': [0., 0., 0.],
                'yaw_flip': [float(action[0]), float(action[1]), float(-action[2])],
                'perturbed': [float(action[0]), float(action[1]), float(1.2 * action[2])],
            }
            texts = {key: action_to_text(val) for key, val in variants.items()}
            if len(set(texts.values())) != 4:
                continue
            candidates[1 if action[2] > 0 else -1].append((name, k, variants, texts))
    rng = random.Random(20261021)
    for pool in candidates.values():
        rng.shuffle(pool)
    selected, used = [], set()
    # Interleave signs and keep one state per source (shard, segment).
    for i in range(20):
        sign = 1 if i % 2 == 0 else -1
        item = next(x for x in candidates[sign] if group(x[0]) not in used)
        name, k, actions, texts = item
        used.add(group(name))
        m = meta[name]
        n = len(m['delta']) + 1
        refs = {'start': f'{name}/0.jpg', 'goal': f'{name}/{n-1}.jpg',
                'current': f'{name}/{k}.jpg', 'next': f'{name}/{k+1}.jpg',
                'metadata': f'{name}/traj_data.pkl'}
        state_id = f'{name}_k{k:02d}'
        seeds = [int.from_bytes(hashlib.sha256(f'20261021:{state_id}:{r}'.encode()).digest()[:4], 'big') % (2**31) for r in range(3)]
        selected.append(dict(state_id=state_id, traj=name, frame=k, source=list(group(name)),
            stratum='left' if sign > 0 else 'right', actions=actions, action_text=texts,
            seeds=seeds, files=refs, sha256={key: digest(args.data / path) for key, path in refs.items()},
            start_pose=[float(m['position'][0][0]), float(m['position'][0][1]), float(m['yaw'][0])],
            raw_frame_indices=[int(m['raw_frame_idx'][k]), int(m['raw_frame_idx'][k+1])],
            nominal_hz=float(m['hz']), nominal_dt_seconds=1 / float(m['hz']),
            action_alignment='native delta[k] -> frame k+1; physical sensor alignment not yet calibrated',
            counterfactual_feasibility='unknown in offline data; encoding range checked'))
    manifest = dict(schema_version=1, purpose='development pilot, not confirmatory results',
        selection_seed=20261021, model_init_seed=42, expected_images=240,
        selection='10 left + 10 right; 0.05 <= abs(dyaw) <= 0.20 rad; 20 distinct source groups; no flow/prediction filtering',
        conditions=['original', 'zero', 'yaw_flip', 'perturbed'],
        perturbation='multiply recorded yaw by 1.2; keep dx,dy; token strings must be distinct',
        protocol='native start + goal + VQ-reconstructed current; empty memory; raw next is reference only',
        split=dict(evidence='Reconstructed from local navigation.py seed=42, trajectory split, max_train_trajs=200 and PLAN_2026.md; not a saved training manifest',
            training_trajectories=train, training_sources=sorted(map(list, train_groups)),
            unseen_trajectories=len(clean), unseen_sources=len({group(n) for n in clean}),
            pilot_sources=sorted(map(list, used)),
            unused_sources=sorted(map(list, {group(n) for n in clean} - used)),
            future_models='Re-audit against each checkpoint training-source union before reusing these samples'),
        states=selected)
    assert len(used) == 20 and not used & train_groups
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(json.dumps(dict(states=20, images=240, train_sources=len(train_groups), unseen_trajectories=len(clean),
                         unseen_sources=len({group(n) for n in clean}), manifest_sha256=digest(args.output))))


if __name__ == '__main__':
    main()
