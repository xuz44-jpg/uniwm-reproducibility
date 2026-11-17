"""Freeze metadata and reuse old predictions; no model or GPU is loaded."""
import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
from collections import Counter
from suite_common import atomic, frozen, read, digest
from angle_scale_common import root, selected_weights, ANGLES, NEW_ANGLES


def link_exact(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        assert digest(source) == digest(target)
    else:
        os.link(source, target)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--study',type=Path,required=True)
    ap.add_argument('--protocol',type=Path,required=True);a=ap.parse_args()
    out=root(a.study);old=a.study/'v2/paper_confirmation_v1'
    p=read(a.protocol);frozen(out/'protocol.json',p)
    assert digest(old/'protocol.json') == p['previous_protocol_sha256']
    assert read(old/'complete.json')['status'] == 'passed'
    for m,w in selected_weights(a.study).items():
        assert w['sha256'] == p['selected_checkpoints'][m]['sha256']
    training=a.study/'v2/uniwm_symmetric_aligned_v3'
    ranges={}
    for split,count in [('train',336),('validation',42)]:
        rows=read(training/(split+'.json'));assert len(rows)==count
        import math
        vals=[math.degrees(r['action'][2]) for r in rows]
        assert max(abs(x) for x in vals) < 7.0001
        ranges[split]=dict(pairs=count,min_yaw_deg=min(vals),max_yaw_deg=max(vals),
            yaw_histogram=dict(Counter(str(round(x,4)) for x in vals)),manifest_sha256=digest(training/(split+'.json')))
    frozen(out/'training_support_audit.json',dict(shared_by_models=['nwm','raenwm','uniwm'],splits=ranges,
        new_angles=NEW_ANGLES,within_habitat_adaptation_support=False,full_base_pretraining_support='unknown',
        source='actual symmetric adaptation manifests; previously independently checked for all3 models'))
    reused={}
    for m in ['nwm','raenwm','uniwm']:
        source=old/'confirmation'/m;done=read(source/'complete.json');assert done['records']==252
        entries={}
        for key,sha in done['record_hashes'].items():
            rp=source/(key+'.json');r=read(rp);assert digest(rp)==sha
            link_exact(rp,out/'confirmation'/m/(key+'.json'))
            if r['status']=='ok':
                im=rp.with_suffix('.png');assert digest(im)==r['image_sha256']
                link_exact(im,out/'confirmation'/m/(key+'.png'))
            entries[key]=dict(source_record=str(rp),record_sha256=sha,image_sha256=r.get('image_sha256'))
        reused[m]=dict(records=entries,source_complete_sha256=digest(source/'complete.json'))
    frozen(out/'reused_predictions.json',reused)
    # Old immutable cache entries are hard-linked for read-only reuse, never rewritten.
    cache=out/'flow/cache';cache.mkdir(parents=True,exist_ok=True)
    for source in (old/'flow/cache').glob('*.npz'):link_exact(source,cache/source.name)
    prior=read(old/'execution_freeze.json');sources=dict(prior['source_hashes'])
    new=['angle_scale_common.py','prepare_angle_scale_v1.py','render_angle_scale_v1.py',
        'run_angle_scale_v1.py','score_angle_scale_v1.py','summarize_angle_scale_v1.py']
    for name in new:sources[str(a.study/name)]=digest(a.study/name)
    for path,sha in sources.items():assert digest(path)==sha
    frozen(out/'execution_freeze.json',dict(status='frozen',source_hashes=sources,
        protocol_sha256=digest(out/'protocol.json'),prior_execution_sha256=digest(old/'execution_freeze.json'),
        training_support_sha256=digest(out/'training_support_audit.json'),
        reused_predictions_sha256=digest(out/'reused_predictions.json'),world_model_updates=0))
    atomic(out/'prepared.json',dict(status='passed',new_predictions=432,reused_predictions=756,
        new_reference_rgb=48,new_reference_depth=48,angles=ANGLES,world_model_updates=0,
        utc=datetime.now(timezone.utc).isoformat()))
    print('PREPARED:432 new,756 reused; no training; ±10/15deg outside adaptation range',flush=True)


if __name__=='__main__':main()
