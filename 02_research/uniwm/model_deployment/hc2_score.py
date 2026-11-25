"""Score generations with the unchanged flow measurement (mirrors score_context_controls_v1.py; seeds kept)."""
import argparse
from pathlib import Path
import numpy as np
import torch
from suite_common import read, digest, atomic, frozen
from habitat_flow_common import compare_flow, direction_label, vector_stats
from hc2_common import root


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--study', type=Path, required=True); a = ap.parse_args()
    out = root(a.study); freeze = read(out / 'execution_freeze.json')
    for path, sha in freeze['source_hashes'].items(): assert digest(path) == sha, path
    run = out / 'generation'; done = read(run / 'complete.json'); assert done['status'] == 'passed'
    m = read(out / 'cases_nwm.json'); cases = m['cases']
    caldir = out / 'flow/references'; assert read(caldir / 'complete.json')['status'] == 'complete'
    cal = read(caldir / 'summary.json')
    dest = run / 'analysis'; dest.mkdir(exist_ok=True)
    frozen(dest / 'config.json', dict(generation_complete_sha256=digest(run / 'complete.json'),
           protocol_sha256=digest(out / 'protocol.json'), script_sha256=digest(__file__),
           reference_audit_sha256=digest(caldir / 'complete.json'), unknown_is_observed_failure=False))
    torch.set_num_threads(2); torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    from probe_habitat_flow_v1 import FlowStore
    flow = FlowStore(out / 'flow'); data = {}
    bygroup = {(c['weight_arm'], c['context_arm'], c['state_id'], c['repeat'], c['target_yaw_deg']): c for c in cases}
    rows = []
    for c in cases:
        path = run / (c['key'] + '.json'); assert digest(path) == done['record_hashes'][c['key']]
        rec = read(path); assert rec['case'] == c and rec['status'] == 'ok'
        rowpath = dest / 'rows' / (c['key'] + '.json')
        if rowpath.exists(): rows.append(read(rowpath)); continue
        sid = c['state_id']; angle = c['target_yaw_deg']
        row = dict(key=c['key'], state_id=sid, scene_id=c['scene_id'], weight_arm=c['weight_arm'],
                   context_arm=c['context_arm'], repeat=c['repeat'], angle=angle, record_sha256=digest(path))
        png = path.with_suffix('.png'); assert digest(png) == rec['image_sha256']; source = c['history_paths'][-1]
        if sid not in data:
            with np.load(caldir / (sid + '.npz'), allow_pickle=False) as f: data[sid] = {k: f[k] for k in f.files}
        d = data[sid]; mask = d['common_mask']; gate = cal['states'][sid]['reference_gate_passed']; row['reference_gate'] = gate
        pred = flow.get(source, png); truth = d['flow_' + str(angle)] - d['flow_0']
        if angle:
            zero = bygroup[(c['weight_arm'], c['context_arm'], sid, c['repeat'], 0)]
            zp = run / (zero['key'] + '.png'); f0 = flow.get(source, zp); delta = pred['flow'] - f0['flow']
            row.update(effect=compare_flow(delta, truth, mask),
                       direction=direction_label(delta, truth, mask, pred['valid'] & f0['valid'], gate),
                       same_as_zero=digest(png) == digest(zp))
        else:
            row['zero_drift'] = vector_stats(pred['flow'] - d['flow_0'], mask)
        atomic(rowpath, row); rows.append(row)
        atomic(dest / 'progress.json', dict(records=len(rows), total=len(cases)))
        print('HC2_SCORE', len(rows), len(cases), flush=True)
    atomic(dest / 'rows.json', rows)
    atomic(dest / 'complete.json', dict(status='passed', records=len(rows), rows_sha256=digest(dest / 'rows.json'),
           world_model_updates=0))


if __name__ == '__main__':
    main()
