"""Read-only tmux view of this study and speech-priority guard summaries."""
import argparse
import json
from pathlib import Path
import time
from suite_common import read
from context_controls_common import root


ap=argparse.ArgumentParser();ap.add_argument('--study',type=Path,required=True);a=ap.parse_args();out=root(a.study)
while True:
    print('\033[2J\033[HINPUT CONTEXT CONTROLS v1 - INFERENCE ONLY',flush=True)
    s=read('external/navigation-world-models/monitor/status.json')
    print(json.dumps({k:s[k] for k in ['utc','phase','active_job','manual_pause','services']},indent=2))
    for cohort in ['native','habitat']:
        for model in ['raenwm','nwm','uniwm']:
            p=out/cohort/model/'progress.json'
            print(cohort,model,read(p) if p.exists() else 'pending')
    for name,d in s['jobs'].items():
        if name.startswith('context_v1_'):print(name,d['status'],d.get('preemptions',0))
    time.sleep(15)
