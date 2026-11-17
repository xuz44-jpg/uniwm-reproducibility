"""Guard-owned RAFT probe of existing Habitat images; no model generation/training."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from suite_common import atomic,digest,frozen,read
from habitat_flow_common import CRITERIA,geometry_flow,texture_mask,compare_flow,vector_stats,direction_label,direction_counts
from uniwm_v3_common import experiment


def save_npz(path,**values):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp')
    with tmp.open('wb') as f:np.savez_compressed(f,**values)
    tmp.replace(path)


def output_root(study):return study/'v2/flow_probe_v1'


def generation_root(study,model,stage):
    if stage=='symmetric_best':assert model=='uniwm';return experiment(study)/'evaluation'
    return study/'v2/evaluation'/model/stage


class FlowStore:
    def __init__(self,out):
        from torchvision.models.optical_flow import raft_large,Raft_Large_Weights
        self.raft=raft_large(weights=Raft_Large_Weights.C_T_SKHT_V2,progress=False).cuda().eval()
        self.out=out/'cache';self.out.mkdir(exist_ok=True);self.arrays={};self.hashes={}
        weight=Path(torch.hub.get_dir())/'checkpoints/raft_large_C_T_SKHT_V2-ff5fadd5.pth'
        assert weight.exists(),weight
        frozen(out/'raft_config.json',dict(script_sha256=digest(__file__),common_sha256=digest(Path(__file__).with_name('habitat_flow_common.py')),
            torch=torch.__version__,weight_sha256=digest(weight),architecture='torchvision RAFT Large C_T_SKHT_V2',
            updates=12,precision='FP32; no autocast; TF32 disabled',resize='Pillow RGB 256x256 BICUBIC, no crop',criteria=CRITERIA))

    def image(self,path):
        key=str(path)
        if key not in self.arrays:
            with Image.open(path) as im:self.arrays[key]=np.array(im.convert('RGB').resize((256,256),Image.Resampling.BICUBIC),copy=True)
        return self.arrays[key]

    def tensor(self,image):return torch.from_numpy(image.copy()).permute(2,0,1)[None].cuda().float()/127.5-1

    def estimate(self,a,b):
        x,y=self.tensor(a),self.tensor(b)
        with torch.inference_mode():
            fw=self.raft(x,y,num_flow_updates=12)[-1][0]
            bw=self.raft(y,x,num_flow_updates=12)[-1]
            assert torch.isfinite(fw).all() and torch.isfinite(bw).all()
            yy,xx=torch.meshgrid(torch.arange(256,device='cuda'),torch.arange(256,device='cuda'),indexing='ij')
            qx=xx+fw[0];qy=yy+fw[1];grid=torch.stack([qx/255*2-1,qy/255*2-1],-1)[None]
            back=F.grid_sample(bw,grid,mode='bilinear',align_corners=True,padding_mode='zeros')[0]
            error=torch.linalg.vector_norm(fw+back,dim=0);mag=torch.linalg.vector_norm(fw,dim=0)
            valid=(qx>=1)&(qx<255)&(qy>=1)&(qy<255)&(error<1+.05*mag)
            return dict(flow=fw.permute(1,2,0).cpu().numpy(),valid=valid.cpu().numpy(),fb_error=error.cpu().numpy())

    def get(self,first,second):
        for p in [first,second]:
            if str(p) not in self.hashes:self.hashes[str(p)]=digest(p)
        key=hashlib.sha256((self.hashes[str(first)]+self.hashes[str(second)]).encode()).hexdigest();path=self.out/(key+'.npz')
        if path.exists():
            with np.load(path,allow_pickle=False) as v:result={k:v[k] for k in v.files}
            assert result['flow'].shape==(256,256,2) and np.isfinite(result['flow']).all()
            return result
        result=self.estimate(self.image(first),self.image(second));save_npz(path,**result);return result


def references(a,out,engine):
    manifest=read(a.study/'v2/eval_manifest.json');refs=read(a.study/'v2/test_references/manifest.json')
    dest=out/'references';dest.mkdir(exist_ok=True)
    frozen(dest/'config.json',dict(manifest_sha256=digest(a.study/'v2/eval_manifest.json'),references_sha256=digest(a.study/'v2/test_references/manifest.json'),
        raft_config_sha256=digest(out/'raft_config.json'),masks='source texture+fixed ROI+analytic visibility+real-reference FB; shared across angles; never generated-success selection',
        real_flow='estimated RAFT on actual rendered futures; pure-rotation analytic projection independently checks estimator',
        known_shifts=[-16,-8,-2,0,2,8,16],criterion=CRITERIA))
    if (dest/'complete.json').exists():
        done=read(dest/'complete.json')
        for p,sha in done['hashes'].items():assert digest(dest/p)==sha
        return done
    states={}
    for tid,entry in refs['references'].items():
        views=entry['reference_views'];source=views['0']['path'];initial=engine.image(source);fixed=texture_mask(initial)
        controls=[]
        for shift in [-16,-8,-2,0,2,8,16]:
            target=cv2.warpAffine(initial,np.float32([[1,0,shift],[0,1,0]]),(256,256),flags=cv2.INTER_NEAREST,borderMode=cv2.BORDER_REPLICATE)
            est=engine.estimate(initial,target);expected=np.zeros_like(est['flow']);expected[...,0]=shift
            cmp=compare_flow(est['flow'],expected,fixed);coverage=float((fixed&est['valid']).sum()/max(1,fixed.sum()))
            passed=(int(fixed.sum())>=400 and coverage>=.5 and cmp['median_epe_px']<=max(.5,.1*abs(shift)))
            controls.append(dict(shift_px=shift,passed=bool(passed),coverage=coverage,comparison=cmp))
        data={};rows=[];masks=[]
        for angle in manifest['angles']:
            ref=views[str(angle)];assert digest(ref['path'])==ref['sha256']
            geometry,visible=geometry_flow(refs['camera'],views['0'],ref)
            est=engine.get(source,ref['path']);base=fixed&visible;mask=base&est['valid'];coverage=float(mask.sum()/max(1,base.sum()))
            cmp=compare_flow(est['flow'],geometry,base);stats=vector_stats(est['flow'],mask)
            passed=(int(base.sum())>=400 and coverage>=.5 and cmp['median_epe_px']<=1. and cmp['p90_epe_px']<=2.)
            if angle==0:passed=passed and stats['median_magnitude'] is not None and stats['median_magnitude']<=.5
            else:passed=passed and stats['median_magnitude'] is not None and stats['median_magnitude']>2. and stats['median_x']*vector_stats(geometry,base)['median_x']>0
            rows.append(dict(angle_deg=angle,passed=bool(passed),reference_coverage=coverage,raft_vs_analytic=cmp,raft_motion=stats))
            data['flow_'+str(angle)]=est['flow'];data['geometry_'+str(angle)]=geometry;data['mask_'+str(angle)]=mask;masks.append(mask)
            atomic(dest/'progress.json',dict(state=tid,angle=angle,states_finished=len(states)))
        common=np.logical_and.reduce(masks);common_coverage=float(common.sum()/max(1,fixed.sum()))
        data['common_mask']=common;save_npz(dest/(tid+'.npz'),**data)
        # Independent small-angle separation uses one common reference mask.
        separations=[]
        for sign in [1,-1]:
            mags=[vector_stats(data['flow_'+str(sign*v)],common)['median_magnitude'] for v in [3,5,7]]
            ok=all(v is not None for v in mags) and mags[1]-mags[0]>1. and mags[2]-mags[1]>1.
            separations.append(dict(sign=sign,magnitudes_px=mags,passed=bool(ok)))
        gate=(all(row['passed'] for row in rows) and all(c['passed'] for c in controls) and all(s['passed'] for s in separations)
            and int(common.sum())>=400 and common_coverage>=.5)
        states[tid]=dict(reference_gate_passed=bool(gate),common_pixels=int(common.sum()),common_coverage=common_coverage,
            fixed_texture_pixels=int(fixed.sum()),angles=rows,synthetic_controls=controls,separation=separations)
        atomic(dest/(tid+'.json'),states[tid]);print('FLOW_REFERENCE',tid,'gate',gate,flush=True)
    atomic(dest/'summary.json',dict(states=states,all_reference_gates_passed=all(s['reference_gate_passed'] for s in states.values()),
        interpretation='Estimator audit; a failed gate is not world-model failure. Thresholds frozen before generated measurements.'))
    done=dict(status='complete',all_reference_gates_passed=all(s['reference_gate_passed'] for s in states.values()),
        hashes={p.name:digest(p) for p in dest.iterdir() if p.suffix in ['.json','.npz'] and p.name not in ['complete.json','progress.json']})
    atomic(dest/'complete.json',done);return done


def score(a,out,engine):
    manifest_path=a.study/'v2/eval_manifest.json';manifest=read(manifest_path);refs=read(a.study/'v2/test_references/manifest.json')
    cal=read(out/'references/summary.json');assert read(out/'references/complete.json')['status']=='complete'
    run=generation_root(a.study,a.model,a.stage);complete=read(run/'complete.json');assert complete['status']=='passed' and complete['records']==72
    for path,sha in manifest['input_hashes'].items():assert digest(path)==sha
    dest=out/(a.model+'_'+a.stage);dest.mkdir(exist_ok=True)
    frozen(dest/'config.json',dict(script_sha256=digest(__file__),manifest_sha256=digest(manifest_path),generation_complete_sha256=digest(run/'complete.json'),
        raft_config_sha256=digest(out/'raft_config.json'),reference_gate_sha256=digest(out/'references/complete.json'),
        model=a.model,stage=a.stage,source_weight_identity=read(run/'config.json')['weight_hashes'],
        primary='same real-current anchor; EPE on real-reference-only mask; no generated-flow mask in primary error',
        action_effect='[F(I0,gen_a)-F(I0,gen_zero)] - [F(I0,ref_a)-F(I0,ref_zero)]; common reference mask',
        amplitude='least-squares flow projection gain; pixel-response diagnostic, not generated angle in degrees',new_model_predictions=0,optimizer_updates=0))
    records={}
    for c in manifest['cases']:
        p=run/(c['key']+'.json');assert digest(p)==complete['input_record_hashes'][c['key']]
        rec=read(p);assert rec['case']==c
        if rec['status']=='ok':assert digest(p.with_suffix('.png'))==rec['image_sha256']
        else:assert rec['status']=='parse_failed'
        records[c['key']]=rec
    if (dest/'complete.json').exists():
        done=read(dest/'complete.json');assert digest(dest/'rows.json')==done['rows_sha256'];assert digest(dest/'summary.json')==done['summary_sha256'];return
    data={}
    for tid in refs['references']:
        with np.load(out/'references'/(tid+'.npz'),allow_pickle=False) as v:data[tid]={k:v[k] for k in v.files}
    rows=[]
    for c in manifest['cases']:
        path=dest/'rows'/(c['key']+'.json')
        if path.exists():rows.append(read(path));continue
        row=dict(key=c['key'],state_id=c['state_id'],experiment=c['experiment'],repeat=c['repeat'],seed=c['seed'],
            target_yaw_deg=c.get('target_yaw_deg'),uniwm_bin_representative=c.get('uniwm_bin_representative') if a.model=='uniwm' else None,
            status=records[c['key']]['status'],direction=dict(label='unknown',reason='parse_failed'))
        if row['status']=='ok':
            current=c['history_paths'][-1];prediction=run/(c['key']+'.png');pred=engine.get(current,prediction);reference=engine.get(current,c['target_image'])
            if c['experiment']=='seven':
                d=data[c['state_id']];angle=c['target_yaw_deg'];mask=d['common_mask'];ref_flow=d['flow_'+str(angle)];gate=cal['states'][c['state_id']]['reference_gate_passed']
                row['reference_gate_passed']=gate;row['analytic_error']=compare_flow(pred['flow'],d['geometry_'+str(angle)],mask)
                if angle:
                    zero=next(z for z in manifest['cases'] if z['experiment']=='seven' and z['state_id']==c['state_id'] and z['repeat']==c['repeat'] and z['target_yaw_deg']==0)
                    zero_path=run/(zero['key']+'.png')
                    if records[zero['key']]['status']=='ok':
                        f0=engine.get(current,zero_path);delta=pred['flow']-f0['flow'];real_delta=ref_flow-d['flow_0']
                        row['action_effect']=compare_flow(delta,real_delta,mask)
                        row['action_direction']=direction_label(delta,real_delta,mask,pred['valid']&f0['valid'],gate)
                        row['image_identical_to_paired_zero']=digest(prediction)==digest(zero_path)
                else:row['zero_drift']=vector_stats(pred['flow']-ref_flow,mask)
                row['direction']=direction_label(pred['flow'],ref_flow,mask,pred['valid'],gate) if angle else dict(label='zero_control')
            else:
                mask=texture_mask(engine.image(current))&reference['valid'];ref_flow=reference['flow'];row['category']=c['category']
                row['reference_gate_passed']=None
            row['reference_only_mask_pixels']=int(mask.sum());row['generated_fb_coverage']=float((mask&pred['valid']).sum()/max(1,mask.sum()))
            row['reference_error']=compare_flow(pred['flow'],ref_flow,mask)
            row['copy_current_control_error']=compare_flow(np.zeros_like(ref_flow),ref_flow,mask)
        atomic(path,row);rows.append(row)
        atomic(dest/'progress.json',dict(scored=len(rows),total=72,last=c['key']))
        print('FLOW_SCORE',a.model,a.stage,len(rows),flush=True)
    codec=[];controls=read(a.study/'v2/codec_controls'/a.model/'complete.json');assert controls['count']==14
    codec_flows={}
    for c in controls['rows']:
        assert digest(c['source_path'])==c['source_sha256'] and digest(c['output_path'])==c['image_sha256']
        tid=c['state_id'];angle=c['angle_deg'];source=refs['references'][tid]['reference_views']['0']['path'];d=data[tid]
        flow=engine.get(source,c['output_path']);codec_flows[(tid,angle)]=flow['flow']
        codec.append(dict(state_id=tid,angle_deg=angle,error=compare_flow(flow['flow'],d['flow_'+str(angle)],d['common_mask']),
            generated_fb_coverage=float((flow['valid']&d['common_mask']).sum()/max(1,d['common_mask'].sum()))))
    for row in codec:
        tid,angle=row['state_id'],row['angle_deg'];d=data[tid]
        if angle:row['action_effect']=compare_flow(codec_flows[(tid,angle)]-codec_flows[(tid,0)],d['flow_'+str(angle)]-d['flow_0'],d['common_mask'])
    noise=[]
    for tid in refs['references']:
        for angle in manifest['angles']:
            cs=[c for c in manifest['cases'] if c['experiment']=='seven' and c['state_id']==tid and c['target_yaw_deg']==angle]
            for i in range(3):
                for j in range(i+1,3):
                    x,y=cs[i],cs[j]
                    if records[x['key']]['status']==records[y['key']]['status']=='ok':
                        fx=engine.get(x['start_image'],run/(x['key']+'.png'))['flow'];fy=engine.get(y['start_image'],run/(y['key']+'.png'))['flow']
                        noise.append(dict(state_id=tid,angle_deg=angle,repeats=[x['repeat'],y['repeat']],mean_flow_difference_px=compare_flow(fx,fy,data[tid]['common_mask'])['mean_epe_px']))
    atomic(dest/'rows.json',rows);atomic(dest/'controls.json',dict(codec=codec,same_action_seed_noise=noise,
        codec_interpretation='one frozen reconstruction per reference; diagnostic control, not a universal irreducible floor; never automatically subtract'))
    seven=[x for x in rows if x['experiment']=='seven' and x['target_yaw_deg']];zero=[x for x in rows if x['experiment']=='seven' and not x['target_yaw_deg']]
    def average(values):
        vals=[v for v in values if v is not None];return float(np.mean(vals)) if vals else None
    summary=dict(model=a.model,stage=a.stage,total=72,successful=sum(x['status']=='ok' for x in rows),
        parse_failed=sum(x['status']!='ok' for x in rows),all_reference_gates_passed=cal['all_reference_gates_passed'],
        absolute_direction=direction_counts([x['direction'] for x in seven]),
        action_effect_direction=direction_counts([x.get('action_direction',dict(label='unknown')) for x in seven]),
        nonzero_identical_to_zero=dict(count=sum(x.get('image_identical_to_paired_zero',False) for x in seven),total=36),
        zero_drift_mean_px=average([x.get('zero_drift',{}).get('mean_magnitude') for x in zero]),
        same_action_seed_noise_mean_px=average([x['mean_flow_difference_px'] for x in noise]),
        nonzero_reference_mean_epe_px=average([x.get('reference_error',{}).get('mean_epe_px') for x in seven]),
        nonzero_action_effect_mean_epe_px=average([x.get('action_effect',{}).get('mean_epe_px') for x in seven]),
        factual_reference_mean_epe_px=average([x.get('reference_error',{}).get('mean_epe_px') for x in rows if x['experiment']=='factual']),
        codec_nonzero_reference_mean_epe_px=average([x['error']['mean_epe_px'] for x in codec if x['angle_deg']]),
        by_angle={},limitations=['one house, two camera positions; seeds are repeats','flow gain is not an angle estimate','unknown is not failure',
            'generated-domain optical flow has no physical ground truth; inspect FB coverage and codec controls','UniWM uses quantized action bins; requested reference degrees and bin representatives retained'])
    for angle in [3,-3,5,-5,7,-7]:
        xs=[x for x in seven if x['target_yaw_deg']==angle]
        summary['by_angle'][str(angle)]=dict(total=len(xs),successful=sum(x['status']=='ok' for x in xs),absolute_direction=direction_counts([x['direction'] for x in xs]),
            action_effect_direction=direction_counts([x.get('action_direction',dict(label='unknown')) for x in xs]),
            reference_epe_px=average([x.get('reference_error',{}).get('mean_epe_px') for x in xs]),
            action_effect_epe_px=average([x.get('action_effect',{}).get('mean_epe_px') for x in xs]),
            action_projection_gain=average([x.get('action_effect',{}).get('gain_projection') for x in xs]),
            median_action_motion_px=average([x.get('action_effect',{}).get('predicted',{}).get('median_magnitude') for x in xs]))
    atomic(dest/'summary.json',summary)
    plot_samples(a,out,dest,run,engine,manifest,data)
    atomic(dest/'complete.json',dict(status='complete',records=72,rows_sha256=digest(dest/'rows.json'),summary_sha256=digest(dest/'summary.json'),
        controls_sha256=digest(dest/'controls.json'),new_world_model_predictions=0,optimizer_updates=0))


def plot_samples(a,out,dest,run,engine,manifest,data):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for tid,d in data.items():
        cs=[c for c in manifest['cases'] if c['experiment']=='seven' and c['state_id']==tid and c['repeat']==0]
        fig,axes=plt.subplots(4,7,figsize=(17.5,10));yy,xx=np.mgrid[32:225:16,32:225:16]
        for j,c in enumerate(cs):
            angle=c['target_yaw_deg'];p=run/(c['key']+'.png');source=c['start_image'];reference=d['flow_'+str(angle)]
            axes[0,j].imshow(engine.image(c['target_image']));axes[0,j].set_title(f'Real {angle:+d} deg')
            if p.exists():
                flow=engine.get(source,p)['flow'];axes[1,j].imshow(engine.image(p));axes[1,j].set_title('Generated seed repeat 0')
                axes[2,j].imshow(engine.image(source),alpha=.5)
                axes[2,j].quiver(xx,yy,reference[yy,xx,0],reference[yy,xx,1],color='lime',angles='xy',scale_units='xy',scale=1)
                axes[2,j].quiver(xx,yy,flow[yy,xx,0],flow[yy,xx,1],color='red',angles='xy',scale_units='xy',scale=1)
                error=np.linalg.norm(flow-reference,axis=-1);masked=np.ma.array(error,mask=~d['common_mask'])
                axes[3,j].imshow(masked,cmap='magma',vmin=0,vmax=20);axes[3,j].set_title('Reference EPE, px (0-20)')
            for i in range(4):axes[i,j].axis('off')
        fig.suptitle(f'{a.model} {a.stage} {tid} | green: real RAFT; red: generated RAFT | fixed repeat 0')
        fig.tight_layout();fig.savefig(dest/(tid+'_flow.png'),dpi=110);plt.close(fig)


def summarize(a,out):
    keys=[(m,s) for m in ['nwm','raenwm','uniwm'] for s in ['baseline','final']]
    if a.include_v3:keys.append(('uniwm','symmetric_best'))
    reports={}
    for m,s in keys:
        d=out/(m+'_'+s);done=read(d/'complete.json');assert done['status']=='complete'
        assert digest(d/'summary.json')==done['summary_sha256'];reports[m+'_'+s]=read(d/'summary.json')
    suffix='all' if a.include_v3 else 'existing'
    atomic(out/('summary_'+suffix+'.json'),dict(status='complete',stages=reports,reference_calibration=read(out/'references/summary.json'),
        new_world_model_predictions=0,optimizer_updates=0))
    lines=['# Habitat 光流探针','', '同一实际当前帧作共同锚点；主误差掩码仅由真实参考决定。数值单位为256×256像素，不是角度。','',
        '| 模型/阶段 | 动作效应：正确/反向/近零/未知（36） | 动作效应 EPE px | 参考 EPE px | 零动作漂移 px |',
        '|---|---|---:|---:|---:|']
    for key,s in reports.items():
        d=s['action_effect_direction'];fmt=lambda v:'NA' if v is None else f'{v:.3f}'
        lines.append(f"| {key} | {d['correct']}/{d['reverse']}/{d['near_zero']}/{d['unknown']} | {fmt(s['nonzero_action_effect_mean_epe_px'])} | {fmt(s['nonzero_reference_mean_epe_px'])} | {fmt(s['zero_drift_mean_px'])} |")
    lines+=['','真实参考验收：'+str(read(out/'references/complete.json')['all_reference_gates_passed']),
        '未过验收的状态方向保留未知。生成域仍可能出现光流失真，结合有效覆盖、codec控制和固定seed图解释。',
        '3°/5°/7°分别按左右列出投影增益、EPE与运动幅度，见各阶段 summary.json；投影增益不是转角比例的物理证明。',
        '一栋房屋、两个位置；三次seed不是独立场景。旧微调存在采集偏置；新UniWM同时修复数据与预处理，不能单独归因。']
    (out/('RESULTS_'+suffix+'.md')).write_text('\n'.join(lines)+'\n')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--study',type=Path,required=True);ap.add_argument('--phase',choices=['references','score','summary'],required=True)
    ap.add_argument('--model',choices=['nwm','raenwm','uniwm']);ap.add_argument('--stage',choices=['baseline','final','symmetric_best']);ap.add_argument('--include-v3',action='store_true');a=ap.parse_args()
    out=output_root(a.study);out.mkdir(exist_ok=True)
    if a.phase=='summary':summarize(a,out);return
    torch.set_num_threads(2);cv2.setNumThreads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    engine=FlowStore(out)
    if a.phase=='references':references(a,out,engine)
    else:assert a.model and a.stage;score(a,out,engine)


if __name__=='__main__':main()
