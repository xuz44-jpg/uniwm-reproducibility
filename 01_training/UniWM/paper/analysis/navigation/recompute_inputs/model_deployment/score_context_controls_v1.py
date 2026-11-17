"""Preserve frozen flow gates; report context/weight effects descriptively."""
import argparse
from collections import defaultdict
from itertools import combinations
from pathlib import Path
import sys
import numpy as np
import torch
from PIL import Image,ImageDraw
from suite_common import read, digest, atomic, frozen
from context_controls_common import root, aggregate_direction
from habitat_flow_common import compare_flow, direction_label, direction_counts, vector_stats
from habitat_v2_metrics import pixel_metrics


def avg(xs):
    xs=[x for x in xs if x is not None];return float(np.mean(xs)) if xs else None


def aggregate(rows, cohort):
    r=dict(records=len(rows),successful=sum(x['status']=='ok' for x in rows),
        lpips_to_actual_future=avg([x.get('reference_metrics',{}).get('lpips') for x in rows]),
        copy_current_lpips_to_future=avg([x.get('copy_current_metrics',{}).get('lpips') for x in rows]),
        ssim=avg([x.get('reference_metrics',{}).get('ssim') for x in rows]))
    if cohort=='habitat':
        nz=[x for x in rows if x['angle']]
        r.update(direction=aggregate_direction(direction_counts([x.get('direction',dict(label='unknown')) for x in nz])),
            action_epe_px=avg([x.get('effect',{}).get('mean_epe_px') for x in nz if x.get('reference_gate')]),
            gain=avg([x.get('effect',{}).get('gain_projection') for x in nz if x.get('reference_gate')]),
            same_as_zero=sum(x.get('same_as_zero',False) for x in nz),
            zero_drift_px=avg([x.get('zero_drift',{}).get('mean_magnitude') for x in rows if not x['angle']]),
            action_vs_zero_lpips=avg([x.get('action_vs_zero_lpips') for x in nz]))
    return r


def source_key(row):
    value=row.get('source')
    if value is None:return None
    return '/'.join(map(str,value)) if isinstance(value,list) else str(value)


def detailed_aggregate(rows,cohort):
    """Keep angles/positions visible, with scenes and native source groups."""
    r=aggregate(rows,cohort)
    r['by_position']={sid:aggregate([x for x in rows if x['state_id']==sid],cohort)
        for sid in sorted({x['state_id'] for x in rows})}
    r['by_angle']={str(angle):aggregate([x for x in rows if x['angle']==angle],cohort)
        for angle in sorted({x['angle'] for x in rows if x['angle'] is not None})}
    r['by_scene']={}
    for scene in sorted({x['scene_id'] for x in rows if x.get('scene_id') is not None}):
        rs=[x for x in rows if x.get('scene_id')==scene];b=aggregate(rs,cohort)
        b['by_position']={sid:aggregate([x for x in rs if x['state_id']==sid],cohort)
            for sid in sorted({x['state_id'] for x in rs})}
        b['by_angle']={str(angle):aggregate([x for x in rs if x['angle']==angle],cohort)
            for angle in sorted({x['angle'] for x in rs if x['angle'] is not None})}
        r['by_scene'][scene]=b
    r['by_source']={source:aggregate([x for x in rows if source_key(x)==source],cohort)
        for source in sorted({source_key(x) for x in rows if source_key(x) is not None})}
    return r


def position_metrics(rows,cohort):
    """One descriptive value per position, never one sample per query frame."""
    a=aggregate(rows,cohort)
    def complete_mean(selected,getter):
        values=[getter(x) for x in selected]
        # Missing queries cannot silently change the set averaged in one arm.
        return avg(values) if values and all(x is not None for x in values) else None
    result=dict(lpips_to_actual_future=complete_mean(rows,lambda x:x.get('reference_metrics',{}).get('lpips')),
        copy_current_lpips_to_future=complete_mean(rows,lambda x:x.get('copy_current_metrics',{}).get('lpips')),
        ssim=complete_mean(rows,lambda x:x.get('reference_metrics',{}).get('ssim')))
    result['generation_success_rate']=a['successful']/a['records'] if a['records'] else None
    if cohort=='habitat':
        nz=[x for x in rows if x['angle']];zeros=[x for x in rows if not x['angle']]
        result.update(action_epe_px=complete_mean(nz,lambda x:x.get('effect',{}).get('mean_epe_px') if x.get('reference_gate') else None),
            gain=complete_mean(nz,lambda x:x.get('effect',{}).get('gain_projection') if x.get('reference_gate') else None),
            zero_drift_px=complete_mean(zeros,lambda x:x.get('zero_drift',{}).get('mean_magnitude')),
            action_vs_zero_lpips=complete_mean(nz,lambda x:x.get('action_vs_zero_lpips')))
        d=a['direction'];total=d['total']
        for label in ['correct','reverse','near_zero','unknown']:
            result[label+'_rate']=d[label]/total if total else None
        result['near_zero_or_unknown_rate']=d['pooled_near_zero_unknown']/total if total else None
        result['acceptance_not_passed_rate']=d['acceptance_not_passed']/total if total else None
        # Full-denominator lower bound; unresolved pairs are not asserted different.
        result['verified_same_as_zero_rate']=a['same_as_zero']/total if total else None
    return result


def paired_position_summary(positive,negative,cohort):
    """Paired arm differences; locations are units of averaging, with no test/CI."""
    pos={sid:[x for x in positive if x['state_id']==sid] for sid in {x['state_id'] for x in positive}}
    neg={sid:[x for x in negative if x['state_id']==sid] for sid in {x['state_id'] for x in negative}}
    assert set(pos)==set(neg),'Paired controls must use exactly the same locations'
    positions={}
    for sid in sorted(pos):
        assert len(pos[sid])==len(neg[sid])
        assert {x['angle'] for x in pos[sid]}=={x['angle'] for x in neg[sid]}
        a=position_metrics(pos[sid],cohort);b=position_metrics(neg[sid],cohort)
        assert set(a)==set(b)
        scenes={x.get('scene_id') for x in pos[sid]+neg[sid]}
        sources={source_key(x) for x in pos[sid]+neg[sid]}
        assert len(scenes)==len(sources)==1
        positions[sid]=dict(scene_id=next(iter(scenes)),source=next(iter(sources)),
            query_records_per_arm=len(pos[sid]),positive=a,negative=b,
            difference={k:a[k]-b[k] if a[k] is not None and b[k] is not None else None for k in a})

    def describe(values):
        result=dict(positions=len(values),metrics={})
        names=list(next(iter(values.values()))['difference']) if values else []
        for name in names:
            valid={sid:v for sid,v in values.items() if v['difference'][name] is not None}
            diffs=[v['difference'][name] for v in valid.values()]
            result['metrics'][name]=dict(paired_positions=len(valid),
                missing_positions=sorted(set(values)-set(valid)),
                positive_mean=avg([v['positive'][name] for v in valid.values()]),
                negative_mean=avg([v['negative'][name] for v in valid.values()]),
                mean_difference=avg(diffs),median_difference=float(np.median(diffs)) if diffs else None,
                min_difference=float(min(diffs)) if diffs else None,
                max_difference=float(max(diffs)) if diffs else None)
        return result

    result=describe(positions)
    result.update(unit='position; query angles aggregated within each position before differences',
        difference='positive arm minus negative arm',inferential_tests=False,confidence_intervals=False,
        missing_metric_policy='a position metric requires every applicable query in both arms; retain missing position IDs and never compare differently filtered angle averages',
        per_position=positions,
        by_scene={scene:describe({sid:v for sid,v in positions.items() if v['scene_id']==scene})
            for scene in sorted({v['scene_id'] for v in positions.values() if v['scene_id'] is not None})},
        by_source={source:describe({sid:v for sid,v in positions.items() if v['source']==source})
            for source in sorted({v['source'] for v in positions.values() if v['source'] is not None})})
    return result


def paired_contrasts(rows,cohort,model):
    groups=defaultdict(list)
    for row in rows:groups[(row['weight_arm'],row['context_arm'])].append(row)
    result=dict(weight={},context={},interpretation='descriptive paired location effects; no significance test, seed/frame independence, or unique-cause claim')

    def contrast(positive,negative,factor):
        p,n=groups[positive],groups[negative]
        r=paired_position_summary(p,n,cohort)
        r.update(positive_arm='/'.join(positive),negative_arm='/'.join(negative),changed_factor=factor,
            by_angle={str(angle):paired_position_summary([x for x in p if x['angle']==angle],
                [x for x in n if x['angle']==angle],cohort)
                for angle in sorted({x['angle'] for x in p if x['angle'] is not None})})
        return r

    contexts=sorted({x['context_arm'] for x in rows})
    for context in contexts:
        if ('adapted',context) in groups and ('pre_habitat',context) in groups:
            result['weight'][context]=contrast(('adapted',context),('pre_habitat',context),'checkpoint adaptation')
    for weight in sorted({x['weight_arm'] for x in rows}):
        for first,second in combinations(contexts,2):
            if (weight,first) not in groups or (weight,second) not in groups:continue
            if cohort=='native':continue
            if model=='uniwm':
                s0,g0=first.split('_g',1);s1,g1=second.split('_g',1)
                if s0!=s1 and g0!=g1:continue  # Do not label a two-factor change a single-factor control.
                factor='goal context, fixed start' if s0==s1 else 'start image and matching start pose, fixed goal'
            else:factor='observed past history, fixed current view'
            result['context'][f'{weight}/{second}__minus__{first}']=contrast((weight,second),(weight,first),factor)
    return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--study',type=Path,required=True);ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--cohort',choices=['native','habitat'],required=True);ap.add_argument('--model',choices=['nwm','raenwm','uniwm'],required=True)
    a=ap.parse_args();out=root(a.study);run=out/a.cohort/a.model;done=read(run/'complete.json')
    m=read(out/f'{a.cohort}_{a.model}.json');cases=m['cases'];assert done['status']=='passed'
    dest=run/'analysis';dest.mkdir(exist_ok=True)
    frozen(dest/'config.json',dict(generation_complete_sha256=digest(run/'complete.json'),
        protocol_sha256=digest(out/'protocol.json'),script_sha256=digest(__file__),
        primary_anchor='current, never changed start image',original_masks_and_thresholds_retained=True,
        native='actual future only, offline dataset has no alternative-action truth',unknown_is_observed_failure=False))
    import lpips
    from probe_habitat_flow_v1 import FlowStore
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    (out/'flow').mkdir(exist_ok=True)
    flow=FlowStore(out/'flow');metric=lpips.LPIPS(net='alex',verbose=False).cuda().eval();cache={};images={}
    if a.cohort=='native' and a.model!='uniwm':
        sys.path.insert(0,str(a.root/'repos'/a.model));import misc
    def image(path,raw=False):
        k=(str(path),raw)
        if k not in images:
            if raw and a.cohort=='native' and a.model!='uniwm':
                with Image.open(path) as im:x=misc.transform(im.convert('RGB'))*.5+.5
                images[k]=np.asarray(Image.fromarray((x.permute(1,2,0).numpy()*255).round().clip(0,255).astype('uint8')).resize((256,256)))
            else:images[k]=flow.image(path)
        return images[k]
    def measure(x,y,raw_y=False,raw_x=False):
        k=(str(x),str(y),raw_y,raw_x)
        if k not in cache:
            ix,iy=image(x,raw_x),image(y,raw_y)
            with torch.inference_mode():v=float(metric(flow.tensor(ix),flow.tensor(iy)))
            cache[k]=dict(lpips=v,**pixel_metrics(ix,iy))
        return cache[k]
    caldir=a.study/'v2/paper_confirmation_v1/flow/references';data={}
    cal=read(caldir/'summary.json') if a.cohort=='habitat' else None
    bygroup={(c['weight_arm'],c['context_arm'],c['state_id'],c.get('target_yaw_deg')):c for c in cases}
    rows=[]
    for c in cases:
        path=run/(c['key']+'.json');assert digest(path)==done['record_hashes'][c['key']]
        rec=read(path);assert rec['case']==c
        rowpath=dest/'rows'/(c['key']+'.json')
        if rowpath.exists():rows.append(read(rowpath));continue
        row=dict(key=c['key'],state_id=c['state_id'],source=c.get('source'),scene_id=c.get('scene_id'),
            weight_arm=c['weight_arm'],context_arm=c['context_arm'],status=rec['status'],origin=rec['origin'],
            angle=c.get('target_yaw_deg'),record_sha256=digest(path))
        if rec['status']=='ok':
            png=path.with_suffix('.png');assert digest(png)==rec['image_sha256'];source=c['history_paths'][-1]
            assert digest(c['target_image'])==m['target_hashes'][c['target_image']]
            row['reference_metrics']=measure(png,c['target_image'],raw_y=a.cohort=='native')
            row['copy_current_metrics']=measure(source,c['target_image'],raw_y=a.cohort=='native',raw_x=a.cohort=='native')
            if a.cohort=='habitat':
                sid=c['state_id'];angle=c['target_yaw_deg']
                if sid not in data:
                    with np.load(caldir/(sid+'.npz'),allow_pickle=False) as f:data[sid]={k:f[k] for k in f.files}
                d=data[sid];mask=d['common_mask'];gate=cal['states'][sid]['reference_gate_passed'];row['reference_gate']=gate
                pred=flow.get(source,png);truth=d['flow_'+str(angle)]-d['flow_0']
                if angle:
                    zero=bygroup[(c['weight_arm'],c['context_arm'],sid,0)]
                    zp=run/(zero['key']+'.png');zr=read(zp.with_suffix('.json'))
                    if zr['status']=='ok':
                        assert digest(zp)==zr['image_sha256'];f0=flow.get(source,zp);delta=pred['flow']-f0['flow']
                        row.update(effect=compare_flow(delta,truth,mask),
                            direction=direction_label(delta,truth,mask,pred['valid']&f0['valid'],gate),
                            same_as_zero=digest(png)==digest(zp),action_vs_zero_lpips=measure(png,zp)['lpips'])
                    else:row['direction']=dict(label='unknown',reason='zero_parse_failed')
                else:row['zero_drift']=vector_stats(pred['flow']-d['flow_0'],mask)
        else:row['direction']=dict(label='unknown',reason='parse_failed')
        atomic(rowpath,row);rows.append(row)
        atomic(dest/'progress.json',dict(records=len(rows),total=len(cases)))
        print('CONTEXT_SCORE',a.cohort,a.model,len(rows),len(cases),flush=True)
    summary=dict(status='passed',overall=detailed_aggregate(rows,a.cohort),by_arm={},limitations=[
        'exploratory reused scenes/states; one seed; no population CI',
        'goal interventions can change training support; not unique cause',
        'native domain sample overlap with pretraining unknown; UniWM locally reproduced',
        'unknown included in acceptance-not-passed, not reclassified as measured zero'])
    for arm in sorted({(r['weight_arm'],r['context_arm']) for r in rows}):
        rs=[r for r in rows if (r['weight_arm'],r['context_arm'])==arm]
        summary['by_arm']['/'.join(arm)]=detailed_aggregate(rs,a.cohort)
    summary['paired_contrasts']=paired_contrasts(rows,a.cohort,a.model)
    atomic(dest/'rows.json',rows);atomic(dest/'summary.json',summary)
    # Fixed-state contact sheets include every context, never selected by score.
    if a.cohort=='habitat':
        for weight in ['pre_habitat','adapted']:
            arms=sorted({c['context_arm'] for c in cases});width=4*256;height=34+len(arms)*282
            for sid in sorted({c['state_id'] for c in cases}):
                sheet=Image.new('RGB',(width,height),'white');draw=ImageDraw.Draw(sheet)
                draw.text((5,5),f'{a.model} {weight} {sid}; columns current / 0 / +5 / -5',fill='black')
                for i,arm in enumerate(arms):
                    group=[bygroup[(weight,arm,sid,d)] for d in [0,5,-5]];y=34+i*282
                    draw.text((5,y),arm,fill='black');sheet.paste(Image.fromarray(image(group[0]['history_paths'][-1])),(0,y+24))
                    for j,c in enumerate(group,1):
                        p=run/(c['key']+'.png')
                        if p.exists():sheet.paste(Image.fromarray(image(p)),(j*256,y+24))
                sheet.save(dest/f'{weight}_{sid}.png')
    atomic(dest/'complete.json',dict(status='passed',records=len(rows),summary_sha256=digest(dest/'summary.json'),
        rows_sha256=digest(dest/'rows.json'),world_model_updates=0))


if __name__=='__main__':main()
