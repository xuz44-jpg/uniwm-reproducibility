"""Freeze the protocol and code, then append the job family to the shared GPU job queue."""
import argparse
from datetime import datetime, timezone
import fcntl
from pathlib import Path
from suite_common import read, digest, atomic, frozen
from hc2_common import (root, RUN, REF_ANGLES, HISTORY_ANGLES, QUERY_ANGLES, HISTORIES, SEEDS, WEIGHTS,
                        SELECTION_SEED, POSITIONS_PER_SCENE, MIN_SEPARATION_M, MAX_ATTEMPTS)

CODE = ['hc2_common.py', 'hc2_render.py', 'hc2_reference.py', 'hc2_run.py', 'hc2_score.py', 'hc2_analyze.py',
        'hc2_queue.py', 'suite_common.py', 'suite_backends.py', 'probe_habitat_flow_v1.py', 'habitat_flow_common.py',
        'habitat_study_common.py', 'habitat_v2_metrics.py', 'uniwm_v3_common.py', 'train_habitat_model.py']


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--study', type=Path, required=True)
    ap.add_argument('--root', type=Path, required=True); ap.add_argument('--append', action='store_true')
    a = ap.parse_args(); out = root(a.study); out.mkdir(parents=True, exist_ok=True); monitor = a.root / 'monitor'
    assert (out / 'PROTOCOL.md').exists()
    old = read(a.study / 'v2/input_context_controls_v1/weights.json')['nwm']
    weights = {w: dict(path=old[w]['path'], sha256=old[w]['sha256']) for w in WEIGHTS}
    for w in WEIGHTS: assert digest(weights[w]['path']) == weights[w]['sha256'], w
    frozen(out / 'protocol.json', dict(run=RUN, protocol_text_sha256=digest(out / 'PROTOCOL.md'),
        model='nwm', weights=weights, reference_angles=REF_ANGLES, history_angles=HISTORY_ANGLES,
        query_angles=QUERY_ANGLES, histories=HISTORIES, seeds=SEEDS, selection=dict(seed=SELECTION_SEED,
        positions_per_scene=POSITIONS_PER_SCENE, min_separation_m=MIN_SEPARATION_M, max_attempts=MAX_ATTEMPTS,
        exclude='12 positions of paper_confirmation_v1', shortfall='drawn from skokloster_castle', replacement=False),
        generations=648, primary=dict(endpoint='A among 72 nonzero queries per weight x history',
        hypotheses='A(rotating) > A(static) for each weight and rotating history (4)',
        test='exact one-sided McNemar on paired queries, Holm over 4, alpha 0.05'),
        world_model_updates=0))
    frozen(out / 'execution_freeze.json', dict(status='frozen',
        source_hashes={str(a.study / n): digest(a.study / n) for n in CODE}, protocol_sha256=digest(out / 'protocol.json')))
    py = str(a.root / 'venv/bin/python'); sim = str(a.study / 'env/bin/python'); nwm_cwd = str(a.root / 'repos/nwm'); jobs = []

    def job(name, script, args=(), minimum=5000, env=py, cwd=None):
        jobs.append(dict(id='hc2_' + name, resumable=True, cwd=cwd or str(a.study), min_free_mib=minimum,
                         argv=[env, '-u', str(a.study / script), '--study', str(a.study), *args]))
    job('render', 'hc2_render.py', minimum=1000, env=sim)
    job('reference', 'hc2_reference.py', minimum=3000)
    job('nwm_smoke', 'hc2_run.py', ['--root', str(a.root), '--smoke'], 10000, cwd=nwm_cwd)
    job('nwm_generate648', 'hc2_run.py', ['--root', str(a.root)], 10000, cwd=nwm_cwd)
    job('nwm_score648', 'hc2_score.py', minimum=3000)
    job('analyze', 'hc2_analyze.py', minimum=0)
    frozen(out / 'queue_append.json', dict(jobs=jobs))
    if not a.append: print('QUEUE_PREPARED', len(jobs)); return
    assert not (monitor / 'pause').exists(), 'An explicit research pause is active'
    status = read(monitor / 'status.json')
    age = (datetime.now(timezone.utc) - datetime.fromisoformat(status['utc'])).total_seconds()
    assert age < 10 and status['phase'] == 'complete' and status['worker_pid'] is None, status
    lock = (monitor / 'hc2_queue_update.lock').open('w'); fcntl.flock(lock, fcntl.LOCK_EX)
    q = read(monitor / 'queue.json'); states = read(monitor / 'jobs_state.json')
    assert all(states[j['id']]['status'] == 'done' for j in q['jobs'])
    assert not {j['id'] for j in jobs} & {j['id'] for j in q['jobs']}
    frozen(out / 'queue_before.json', q); q['jobs'] += jobs; atomic(monitor / 'queue.json', q)
    atomic(out / 'queue_deployment.json', dict(status='appended', utc=datetime.now(timezone.utc).isoformat(),
           jobs=[j['id'] for j in jobs], queue_sha256=digest(monitor / 'queue.json'), protocol_sha256=digest(out / 'protocol.json')))
    print('QUEUE_APPENDED', len(jobs), flush=True)


if __name__ == '__main__':
    main()
