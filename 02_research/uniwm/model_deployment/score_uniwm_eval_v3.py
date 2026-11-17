"""Direct real-reference image scoring; no learned angle estimator."""
import argparse
from collections import Counter,defaultdict
import itertools
import math
from pathlib import Path
import cv2
import numpy as np
from PIL import Image,ImageDraw
import torch
from suite_common import atomic,digest,frozen,read
from habitat_v2_metrics import pixel_metrics,rank_match
from run_habitat_eval_v2 import verify_record
from uniwm_v3_common import experiment


def mean_metrics(rows):
    good=[r for r in rows if r.get('status')=='ok']
    result=dict(total=len(rows),successful=len(good),unknown=len(rows)-len(good))
    for metric in ['lpips','ssim','psnr_db']:
        vals=[r['correct_reference'][metric] for r in good if r['correct_reference'][metric] is not None]
        result[metric]=float(np.mean(vals)) if vals else None
    result['perfect_pixel_matches']=sum(r['correct_reference']['perfect_pixel_match'] for r in good)
    return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--study',type=Path,required=True);ap.add_argument('--model',required=True)
    ap.add_argument('--stage',choices=['baseline','final'],required=True);a=ap.parse_args();r=a.study
    assert a.model=='uniwm' and a.stage=='final'
    m=read(r/'v2/eval_manifest.json');run=experiment(r)/'evaluation';complete=read(run/'complete.json')
    assert complete['status']=='passed' and complete['records']==72
    refs=read(r/'v2/test_references/manifest.json');controls=read(r/'v2/codec_controls'/a.model/'complete.json')
    out=run/'analysis';out.mkdir(exist_ok=True)
    import lpips
    cv2.setNumThreads(1);torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    metric=lpips.LPIPS(net='alex',verbose=False).cuda().eval()
    weights=[Path(torch.hub.get_dir())/'checkpoints/alexnet-owt-7be5be79.pth',Path(lpips.__file__).parent/'weights/v0.1/alex.pth']
    frozen(out/'config.json',dict(script_sha256=digest(__file__),eval_config_sha256=digest(run/'config.json'),
        manifest_sha256=digest(r/'v2/eval_manifest.json'),generation_record_hashes=complete['input_record_hashes'],
        weights={str(p):digest(p) for p in weights},metric=m['scores'],device='cuda fp32; TF32 disabled',
        pixel_resize='all source/generated RGB to 256x256 bicubic, no alignment',
        psnr_identity='None with perfect_pixel_match=true means positive infinity',physical_angle_accuracy=False))
    arrays={};tensors={};pair_cache={}
    def image(path):
        key=str(path)
        if key not in arrays:
            with Image.open(path) as im:arrays[key]=np.array(im.convert('RGB').resize((256,256),Image.Resampling.BICUBIC),copy=True)
        return arrays[key]
    def tensor(path):
        key=str(path)
        if key not in tensors:tensors[key]=torch.from_numpy(image(path)).permute(2,0,1)[None].cuda().float()/127.5-1
        return tensors[key]
    def measure(p,q):
        key=tuple(sorted([str(p),str(q)]))
        if key not in pair_cache:
            with torch.inference_mode():distance=float(metric(tensor(p),tensor(q)))
            assert np.isfinite(distance)
            pair_cache[key]=dict(lpips=distance,**pixel_metrics(image(p),image(q)))
        return pair_cache[key]
    rows=[];generation={}
    with torch.inference_mode():
        for index,c in enumerate(m['cases']):
            rec=verify_record(run,c);assert rec and digest(run/(c['key']+'.json'))==complete['input_record_hashes'][c['key']]
            row=dict(key=c['key'],state_id=c['state_id'],experiment=c['experiment'],repeat=c['repeat'],seed=c['seed'],status=rec['status'],category=c.get('category'))
            if rec['status']=='ok':
                p=run/(c['key']+'.png');generation[c['key']]=p
                row['correct_reference']=measure(p,c['target_image'])
                row['to_initial']=measure(p,c['start_image'])
                row['copy_initial_control']=measure(c['start_image'],c['target_image'])
                if c['experiment']=='seven':
                    row['target_yaw_deg']=c['target_yaw_deg'];entry=refs['references'][c['state_id']]
                    row['all_reference_distances']=[measure(p,entry['reference_views'][str(angle)]['path']) for angle in m['angles']]
                    rank=rank_match([v['lpips'] for v in row['all_reference_distances']],m['angles'].index(c['target_yaw_deg']),m['scores']['tie_tolerance'])
                    row['reference_match']=rank
                    if rank['status']=='other_reference':
                        actual=m['angles'][rank['closest_indices'][0]];wanted=c['target_yaw_deg']
                        row['mismatch_type']='zero_reference' if actual==0 else 'nonzero_for_zero' if wanted==0 else 'opposite_direction' if actual*wanted<0 else 'magnitude'
            else:row['reason']=rec['error']
            atomic(out/'rows'/(c['key']+'.json'),row);rows.append(row)
            atomic(out/'progress.json',dict(scored=len(rows),total=72,last=c['key']))
        codec_rows=[]
        for control in controls['rows']:
            assert digest(control['output_path'])==control['image_sha256']
            assert digest(control['source_path'])==control['source_sha256']
            entry=refs['references'][control['state_id']]
            codec_rows.append(dict(state_id=control['state_id'],angle_deg=control['angle_deg'],correct_reference=measure(control['output_path'],control['source_path']),
                all_reference_distances=[measure(control['output_path'],entry['reference_views'][str(angle)]['path']) for angle in m['angles']],status='ok'))
        noise=[];action_response=[]
        for tid in refs['references']:
            for angle in m['angles']:
                selected=[c for c in m['cases'] if c['experiment']=='seven' and c['state_id']==tid and c['target_yaw_deg']==angle]
                for first,second in itertools.combinations(selected,2):
                    if first['key'] in generation and second['key'] in generation:
                        noise.append(dict(state_id=tid,angle_deg=angle,repeats=[first['repeat'],second['repeat']],lpips=measure(generation[first['key']],generation[second['key']])['lpips']))
            for repeat in range(3):
                selected=[c for c in m['cases'] if c['experiment']=='seven' and c['state_id']==tid and c['repeat']==repeat]
                zero=next(c for c in selected if c['target_yaw_deg']==0)
                for c in selected:
                    if c['target_yaw_deg'] and c['key'] in generation and zero['key'] in generation:
                        action_response.append(dict(state_id=tid,angle_deg=c['target_yaw_deg'],repeat=repeat,lpips=measure(generation[c['key']],generation[zero['key']])['lpips']))
    atomic(out/'rows.json',rows);atomic(out/'controls.json',dict(codec_reconstructions=codec_rows,same_action_seed_noise=noise,paired_action_vs_zero=action_response))
    seven=[row for row in rows if row['experiment']=='seven'];factual=[row for row in rows if row['experiment']=='factual']
    matches=Counter(row.get('reference_match',{}).get('status','unknown') for row in seven)
    mismatch=Counter(row['mismatch_type'] for row in seven if 'mismatch_type' in row)
    summary=dict(status='passed',model=a.model,stage=a.stage,total_records=72,seven=mean_metrics(seven),factual=mean_metrics(factual),
        seven_reference_matches=dict(matches),seven_mismatch_types=dict(mismatch),codec=mean_metrics(codec_rows),
        factual_categories={cat:mean_metrics([row for row in factual if row['category']==cat]) for cat in ['turn','translation','mixed','static']},
        zero_action=mean_metrics([row for row in seven if row.get('target_yaw_deg')==0]),
        same_action_seed_noise_lpips=float(np.mean([v['lpips'] for v in noise])) if noise else None,
        paired_action_vs_zero_lpips=float(np.mean([v['lpips'] for v in action_response])) if action_response else None,
        scene_count=1,heldout_positions=2,independent_confidence_interval=None,continuous_angle_estimates=False)
    atomic(out/'summary.json',summary)
    for tid,entry in refs['references'].items():
        sheet=Image.new('RGB',(7*256,4*278),'white');draw=ImageDraw.Draw(sheet)
        for column,angle in enumerate(m['angles']):
            source=entry['reference_views'][str(angle)]['path'];sheet.paste(Image.fromarray(image(source)),(column*256,22));draw.text((column*256+4,4),f'GT {angle:+d} deg',fill='black')
            for repeat in range(3):
                c=next(c for c in m['cases'] if c['experiment']=='seven' and c['state_id']==tid and c['target_yaw_deg']==angle and c['repeat']==repeat)
                if c['key'] in generation:sheet.paste(Image.fromarray(image(generation[c['key']])),(column*256,(repeat+1)*278+22))
                draw.text((column*256+4,(repeat+1)*278+4),f'{a.model} {a.stage} seed repeat {repeat}',fill='black')
        sheet.save(out/(tid+'_contact.png'))
    atomic(out/'complete.json',dict(status='passed',summary_sha256=digest(out/'summary.json'),rows_sha256=digest(out/'rows.json'),controls_sha256=digest(out/'controls.json'),records=72))
    print('SCORED',a.model,a.stage,summary['seven'],summary['factual'],flush=True)


if __name__=='__main__':main()
