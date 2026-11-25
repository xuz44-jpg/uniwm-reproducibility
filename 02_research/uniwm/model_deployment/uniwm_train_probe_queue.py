"""Append the train-probe jobs to the shared GPU job queue (same checks as hc2_queue.py)."""
from datetime import datetime, timezone
import fcntl
from pathlib import Path
from suite_common import read, atomic

ROOT = Path('external/navigation-world-models'); STUDY = Path('external/habitat-yaw-reference'); MON = ROOT / 'monitor'
UNI = 'external/uniwm-action-pilot/venv/bin/python'; NAV = str(ROOT / 'venv/bin/python')
import sys
SUFFIX = sys.argv[1] if len(sys.argv) > 1 else ''
jobs = [dict(id='uprobe_generate120' + SUFFIX, resumable=True, cwd=str(STUDY), min_free_mib=20000,
             argv=[UNI, '-u', str(STUDY / 'uniwm_train_probe.py')]),
        dict(id='uprobe_analyze' + SUFFIX, resumable=True, cwd=str(STUDY), min_free_mib=0,
             argv=[NAV, '-u', str(STUDY / 'uniwm_train_probe_analyze.py')])]
assert not (MON / 'pause').exists()
status = read(MON / 'status.json')
age = (datetime.now(timezone.utc) - datetime.fromisoformat(status['utc'])).total_seconds()
assert age < 10 and status['phase'] == 'complete' and status['worker_pid'] is None, status
lock = (MON / 'uprobe_queue_update.lock').open('w'); fcntl.flock(lock, fcntl.LOCK_EX)
q = read(MON / 'queue.json'); states = read(MON / 'jobs_state.json')
assert all(states[j['id']]['status'] == 'done' for j in q['jobs'])
assert not {j['id'] for j in jobs} & {j['id'] for j in q['jobs']}
q['jobs'] += jobs; atomic(MON / 'queue.json', q)
print('QUEUE_APPENDED', len(jobs), flush=True)
