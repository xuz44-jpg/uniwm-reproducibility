"""Calibrated flow and image measurements with full denominators and no fitting."""
import argparse
from collections import Counter,defaultdict
import itertools
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from PIL import Image
from suite_common import read,digest,atomic,frozen
from angle_scale_common import root,ANGLES
from habitat_flow_common import compare_flow,direction_label,direction_counts,vector_stats
from habitat_v2_metrics import pixel_metrics,rank_match


def mean(values):
    xs=[v for v in values if v is not None]
    return float(np.mean(xs)) if xs else None


def aggregate(rows):
    nonzero=[r for r in rows if r['angle']!=0];zero=[r for r in rows if r['angle']==0]
    return dict(records=len(rows),successful=sum(r['status']=='ok' for r in rows),
        action_direction=direction_counts([r.get('action_direction',dict(label='unknown')) for r in nonzero]),
        calibrated_nonzero_records=sum(r.get('reference_gate_passed',False) for r in nonzero),
        action_epe_px=mean([r.get('action_effect',{}).get('mean_epe_px') for r in nonzero if r.get('reference_gate_passed')]),
        action_gain=mean([r.get('action_effect',{}).get('gain_projection') for r in nonzero if r.get('reference_gate_passed')]),
        reference_lpips=mean([r.get('correct_reference',{}).get('lpips') for r in rows]),
        reference_matches=dict(Counter(r.get('reference_match',{}).get('status','unknown') for r in rows)),
        paired_zero_identical=dict(count=sum(r.get('image_identical_to_paired_zero',False) for r in nonzero),total=len(nonzero)),
        zero_drift_px=mean([r.get('zero_drift',{}).get('mean_magnitude') for r in zero]),
        copy_token_fraction={name:mean([r.get('copy_tokens',{}).get(name,{}).get('equal')/784
            for r in rows if r.get('copy_tokens',{}).get(name,{}).get('equal') is not None]) for name in ['start','goal','current']})


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--study',type=Path,required=True)
    ap.add_argument('--phase',choices=['reference','score','summary'],required=True)
    ap.add_argument('--model',choices=['nwm','raenwm','uniwm']);ap.add_argument('--cohort',choices=['confirmation'])
    a=ap.parse_args();out=root(a.study)
    freeze=read(out/'execution_freeze.json');assert freeze['status']=='frozen'
    for path,sha in freeze['source_hashes'].items():assert digest(path)==sha
    if a.phase=='summary':
        raise ValueError('Use summarize_angle_scale_v1.py for the extension summary')
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    from probe_habitat_flow_v1 import FlowStore,references
    flow=out/'flow';flow.mkdir(exist_ok=True);engine=FlowStore(flow)
    if a.phase=='reference':
        # The existing reference validator is reused verbatim through an explicit adapter.
        stub=out/'reference_audit_input';folder=stub/'v2/test_references';folder.mkdir(parents=True,exist_ok=True)
        frozen(stub/'v2/eval_manifest.json',read(out/'confirmation_manifest.json'))
        frozen(folder/'manifest.json',read(out/'references_manifest.json'))
        references(SimpleNamespace(study=stub),flow,engine)
        return
    assert a.model and a.cohort
    run=out/a.cohort/a.model;done=read(run/'complete.json');assert done['status']=='passed'
    if a.cohort=='confirmation':
        manifest=read(out/'confirmation_manifest.json');refs=read(out/'references_manifest.json')
        caldir=flow/'references';codec=read(out/'codec_controls'/a.model/'complete.json')['rows']
    else:
        assert a.model=='uniwm';manifest=read(out/'mechanism_manifest.json');refs=read(a.study/'v2/test_references/manifest.json')
        caldir=a.study/'v2/flow_probe_v1/references';codec=[]
    cal=read(caldir/'summary.json');cases=manifest['cases'];records={};bygroup={}
    for c in cases:
        path=run/(c['key']+'.json');assert digest(path)==done['record_hashes'][c['key']]
        rec=read(path);assert rec['case']==c
        if rec['status']=='ok':assert digest(path.with_suffix('.png'))==rec['image_sha256']
        records[c['key']]=rec
        bygroup[(c.get('arm','native'),c['state_id'],c['repeat'],c['target_yaw_deg'])]=c
    dest=run/'analysis';dest.mkdir(exist_ok=True)
    frozen(dest/'config.json',dict(script_sha256=digest(__file__),generation_complete_sha256=digest(run/'complete.json'),
        protocol_sha256=digest(out/'protocol.json'),reference_audit_sha256=digest(caldir/'complete.json'),
        weights=read(run/'config.json')['weights'],criteria='unchanged original Habitat flow criteria',
        errors_mask='real-reference-only, never generated-success selection',unknown_is_failure=False,physical_angle_accuracy=False))
    if (dest/'complete.json').exists():
        d=read(dest/'complete.json');assert digest(dest/'summary.json')==d['summary_sha256'];return
    import lpips
    metric=lpips.LPIPS(net='alex',verbose=False).cuda().eval();tensors={};pair_cache={}
    frozen(dest/'metric_weights.json',{'raft':read(flow/'raft_config.json')['weight_sha256'],
        'alexnet':digest(Path(torch.hub.get_dir())/'checkpoints/alexnet-owt-7be5be79.pth'),
        'lpips_alex':digest(Path(lpips.__file__).parent/'weights/v0.1/alex.pth')})
    def tensor(path):
        key=str(path)
        if key not in tensors:tensors[key]=engine.tensor(engine.image(path))
        return tensors[key]
    def measure(x,y):
        key=tuple(sorted([str(x),str(y)]))
        if key not in pair_cache:
            with torch.inference_mode():value=float(metric(tensor(x),tensor(y)))
            assert np.isfinite(value)
            pair_cache[key]=dict(lpips=value,**pixel_metrics(engine.image(x),engine.image(y)))
        return pair_cache[key]
    data={}
    for sid in refs['references']:
        with np.load(caldir/(sid+'.npz'),allow_pickle=False) as f:data[sid]={k:f[k] for k in f.files}
    rows=[]
    for c in cases:
        path=dest/'rows'/(c['key']+'.json')
        if path.exists():rows.append(read(path));continue
        rec=records[c['key']];sid=c['state_id'];angle=c['target_yaw_deg'];arm=c.get('arm','native')
        row=dict(key=c['key'],state_id=sid,scene_id=c['scene_id'],arm=arm,repeat=c['repeat'],angle=angle,status=rec['status'],
            reference_gate_passed=cal['states'][sid]['reference_gate_passed'])
        if rec['status']=='ok':
            prediction=run/(c['key']+'.png');d=data[sid];mask=d['common_mask'];source=c['start_image']
            pred=engine.get(source,prediction);ref=d['flow_'+str(angle)]
            row['correct_reference']=measure(prediction,c['target_image']);row['copy_initial_control']=measure(source,c['target_image'])
            distances=[measure(prediction,refs['references'][sid]['reference_views'][str(v)]['path'])['lpips'] for v in ANGLES]
            row['all_reference_lpips']=distances;row['reference_match']=rank_match(distances,ANGLES.index(angle),1e-6)
            row['reference_error']=compare_flow(pred['flow'],ref,mask)
            row['generated_fb_coverage']=float((mask&pred['valid']).sum()/max(1,mask.sum()))
            row['copy_tokens']=rec.get('copy_tokens',{});row['sampling_support']=rec.get('sampling_support')
            if angle:
                z=bygroup[(arm,sid,c['repeat'],0)];zpath=run/(z['key']+'.png')
                if records[z['key']]['status']=='ok':
                    f0=engine.get(source,zpath);delta=pred['flow']-f0['flow'];truth=ref-d['flow_0']
                    row['action_effect']=compare_flow(delta,truth,mask)
                    row['action_direction']=direction_label(delta,truth,mask,pred['valid']&f0['valid'],row['reference_gate_passed'])
                    row['image_identical_to_paired_zero']=digest(prediction)==digest(zpath)
                    row['action_vs_zero_lpips']=measure(prediction,zpath)['lpips']
                else:row['action_direction']=dict(label='unknown',reason='zero_branch_parse_failed')
            else:row['zero_drift']=vector_stats(pred['flow']-ref,mask)
        else:row['action_direction']=dict(label='unknown',reason='parse_failed')
        atomic(path,row);rows.append(row)
        atomic(dest/'progress.json',dict(scored=len(rows),total=len(cases),last=c['key']))
        print('SCORE',a.cohort,a.model,len(rows),len(cases),flush=True)
    controls=[]
    for c in codec:
        assert digest(c['source_path'])==c['source_sha256'] and digest(c['output_path'])==c['image_sha256']
        sid=c['state_id'];angle=c['angle_deg'];d=data[sid]
        f=engine.get(refs['references'][sid]['reference_views']['0']['path'],c['output_path'])
        controls.append(dict(state_id=sid,scene_id=c['scene_id'],angle=angle,
            reference_error=compare_flow(f['flow'],d['flow_'+str(angle)],d['common_mask']),
            lpips=measure(c['output_path'],c['source_path'])['lpips']))
    noise=[]
    for arm,sid,angle in sorted({(c.get('arm','native'),c['state_id'],c['target_yaw_deg']) for c in cases}):
        cs=[bygroup[(arm,sid,rep,angle)] for rep in range(3)]
        for x,y in itertools.combinations(cs,2):
            if records[x['key']]['status']!= 'ok' or records[y['key']]['status']!='ok':continue
            px=run/(x['key']+'.png');py=run/(y['key']+'.png')
            fx=engine.get(x['start_image'],px)['flow'];fy=engine.get(y['start_image'],py)['flow']
            noise.append(dict(arm=arm,state_id=sid,angle=angle,repeats=[x['repeat'],y['repeat']],
                lpips=measure(px,py)['lpips'],flow_difference_px=compare_flow(fx,fy,data[sid]['common_mask'])['mean_epe_px']))
    summary=dict(status='passed',cohort=a.cohort,model=a.model,overall=aggregate(rows),by_arm={},
        independent_scene_count=2 if a.cohort=='confirmation' else 1,
        population_confidence_interval=None,reason='Only2 confirmation scenes; seed repeats are not independent scenes',
        codec_mean_lpips=mean([c['lpips'] for c in controls]),unknown_is_failure=False)
    for arm in sorted({r['arm'] for r in rows}):
        ar=[r for r in rows if r['arm']==arm];block=aggregate(ar)
        block['by_scene']={sid:aggregate([r for r in ar if r['scene_id']==sid]) for sid in sorted({r['scene_id'] for r in ar})}
        block['by_position']={sid:aggregate([r for r in ar if r['state_id']==sid]) for sid in sorted({r['state_id'] for r in ar})}
        block['by_angle']={str(d):aggregate([r for r in ar if r['angle']==d]) for d in ANGLES}
        block['scene_macro_action_epe_px']=mean([v['action_epe_px'] for v in block['by_scene'].values()])
        block['same_action_seed_lpips']=mean([n['lpips'] for n in noise if n['arm']==arm])
        block['same_action_seed_flow_px']=mean([n['flow_difference_px'] for n in noise if n['arm']==arm])
        summary['by_arm'][arm]=block
    atomic(dest/'rows.json',rows);atomic(dest/'controls.json',dict(codec=controls,same_action_seed_noise=noise))
    atomic(dest/'summary.json',summary)
    # Fixed repeat0 contact sheets, all angles and positions, without outcome selection.
    from PIL import ImageDraw
    for arm in summary['by_arm']:
        for sid in refs['references']:
            selected=[c for c in cases if c['state_id']==sid and c['repeat']==0 and c.get('arm','native')==arm]
            if not selected:continue
            sheet=Image.new('RGB',(len(ANGLES)*256,2*278),'white');draw=ImageDraw.Draw(sheet)
            for i,angle in enumerate(ANGLES):
                c=next(c for c in selected if c['target_yaw_deg']==angle)
                sheet.paste(Image.fromarray(engine.image(c['target_image'])),(i*256,22));draw.text((i*256+4,3),f'GT {angle:+d} deg',fill='black')
                if records[c['key']]['status']=='ok':sheet.paste(Image.fromarray(engine.image(run/(c['key']+'.png'))),(i*256,300))
                draw.text((i*256+4,281),a.model+' '+arm,fill='black')
            path=dest/f'{arm}_{sid}_repeat0.png';sheet.save(path)
    atomic(dest/'complete.json',dict(status='passed',records=len(rows),rows_sha256=digest(dest/'rows.json'),
        summary_sha256=digest(dest/'summary.json'),controls_sha256=digest(dest/'controls.json'),world_model_updates=0))



if __name__=='__main__':main()
