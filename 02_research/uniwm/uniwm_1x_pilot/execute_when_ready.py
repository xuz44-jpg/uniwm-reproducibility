"""One-off dependency/transfer gate, smoke test, inference and analysis (no service changes)."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--wait-hours', type=float, default=3)
    a = ap.parse_args()
    os.chdir(a.root)
    lock = open('queue.lock', 'w')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    state_path = Path('pipeline_status.json')
    def status(stage, **kw):
        obj = dict(stage=stage, utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), pid=os.getpid(), **kw)
        tmp = state_path.with_suffix('.tmp'); tmp.write_text(json.dumps(obj, indent=2)); tmp.replace(state_path)
        print(json.dumps(obj), flush=True)
    expected = json.loads(Path('pilot/checkpoint_manifest.json').read_text())
    py = str(a.root / 'venv/bin/python')
    start = time.monotonic()
    previous = None
    checked_checkpoint = False
    try:
        while True:
            missing = []
            for name, info in expected.items():
                p = Path('checkpoints/1x200') / name
                if not p.is_file() or p.stat().st_size != info['bytes']:
                    missing.append('checkpoint:' + name)
            index = Path('models/anole/model.safetensors.index.json')
            shards = []
            if index.exists():
                shards = sorted(set(json.loads(index.read_text())['weight_map'].values()))
                for name in shards:
                    p = Path('models/anole') / name
                    if not p.is_file() or not (p.parent / '.cache/huggingface/download' / (name + '.metadata')).is_file():
                        missing.append('base:' + name)
            else:
                missing.append('base:index')
            dep = subprocess.run([py, '-c', 'from transformers.generation.logits_process import AllowOnlyTokensAtRelativeOffsetLogitsProcessor; import peft, accelerate; import sys; sys.path.insert(0,"repo"); from scripts.load_model import load_model; from uniwm.wrapped_visualizer import AnoleforConditionalGeneration'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            if dep.returncode:
                missing.append('dependencies')
                Path('logs/dependency_check.log').write_text(dep.stdout)
            if not missing:
                if not checked_checkpoint:
                    for name, info in expected.items():
                        assert digest(Path('checkpoints/1x200') / name) == info['sha256'], 'Checkpoint hash mismatch: ' + name
                    checked_checkpoint = True
                base_hashes = {}
                for name in shards:
                    p = Path('models/anole') / name
                    meta = (p.parent / '.cache/huggingface/download' / (name + '.metadata')).read_text().splitlines()
                    assert meta[0] == '96df52301e844d8a624a13953051ead4c008343b'
                    actual = digest(p)
                    assert len(meta[1]) == 64 and actual == meta[1], 'Base shard hash mismatch: ' + name
                    base_hashes[name] = actual
                Path('base_shard_hashes.json').write_text(json.dumps(base_hashes, indent=2))
                break
            if missing != previous:
                status('waiting_for_files_or_environment', missing=missing)
                previous = missing
            if time.monotonic() - start > a.wait_hours * 3600:
                raise TimeoutError('Preparation did not complete within the bounded wait')
            time.sleep(20)
        env = os.environ.copy()
        env.update(PYTHONUNBUFFERED='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
        command = [py, '-u', 'pilot/run_pilot.py', '--repo', 'repo', '--data', 'data/1x_nav',
            '--checkpoint', 'checkpoints/1x200', '--base-model', 'models/anole',
            '--manifest', 'pilot/manifest.json', '--output', 'runs/uniwm_1x_dev20_v1']
        status('smoke_inference')
        with open('logs/inference.log', 'a') as log:
            subprocess.run(command + ['--max-images', '4'], env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        # Technical smoke gate only; visual/measurement review is still required before scientific use.
        run = Path('runs/uniwm_1x_dev20_v1')
        records = [json.loads(p.read_text()) for p in run.glob('traj_*/*_s0.json')]
        assert len(records) == 4 and all(r['status'] == 'ok' and r['output_size'] == [448, 448] for r in records)
        assert len({r['seed'] for r in records}) == 1
        assert len({r['action_text'] for r in records}) == 4
        status('full_inference', smoke_images=4, remaining_images=236,
               smoke_median_seconds=sorted(r['seconds'] for r in records)[2],
               peak_reserved_bytes=max(r['peak_reserved_bytes'] for r in records))
        with open('logs/inference.log', 'a') as log:
            subprocess.run(command + ['--resume'], env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        summary = json.loads((run / 'summary.json').read_text())
        assert summary['attempts'] == 240
        status('generation_complete', summary=summary, analysis='pending')
        # Pre-installed metrics run after the generation process releases the model from GPU memory.
        with open('logs/analysis.log', 'a') as log:
            result = subprocess.run([py, '-u', 'pilot/analyze_pilot.py', '--run', str(run)], stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            status('generation_complete_analysis_failed', summary=summary, analysis_log='logs/analysis.log')
        else:
            status('complete_pending_scientific_review', summary=summary)
    except Exception as error:
        status('failed', error=repr(error))
        raise


if __name__ == '__main__':
    main()
