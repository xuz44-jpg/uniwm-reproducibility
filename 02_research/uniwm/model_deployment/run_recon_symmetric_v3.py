"""Paired frozen/final inference and independent frozen-codec controls."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image
import torch
from suite_common import atomic,digest,frozen,read
from train_habitat_model import CHECKPOINTS
from recon_symmetric_v3_common import experiment


def tensor_hash(x):
    t=x.detach().contiguous().cpu()
    return hashlib.sha256(str((str(t.dtype),list(t.shape))).encode()+t.view(torch.uint8).numpy().tobytes()).hexdigest()


def training_path(study,model):
    return experiment(study)/'training'/model


def put_png(path,image):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp');image.save(tmp,format='PNG');tmp.replace(path)


def verify_record(out,case):
    p=out/(case['key']+'.json')
    if not p.exists():return None
    record=read(p);assert record['case']==case
    if record['status']=='ok':assert digest(p.with_suffix('.png'))==record['image_sha256']
    else:assert record['status']=='parse_failed'
    for path,sha in record.get('input_hashes',{}).items():assert digest(path)==sha
    return record


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--study',type=Path,required=True);ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--model',choices=CHECKPOINTS,required=True);ap.add_argument('--stage',choices=['baseline','final'],required=True)
    a=ap.parse_args();assert a.model in ['nwm','raenwm'] and a.stage=='final';r=a.study;m=read(r/'v2/eval_manifest.json');out=experiment(r)/'evaluation'/a.model;out.mkdir(parents=True,exist_ok=True)
    cases=m['cases'];assert len(cases)==72
    for p,sha in m['input_hashes'].items():assert digest(p)==sha
    final=training_path(r,a.model);weight=CHECKPOINTS[a.model];weight_key='released_ema' if a.model!='uniwm' else 'existing_adapter'
    if a.stage=='final':
        report=read(final/'complete.json');assert report['status']=='passed' and report['test_used_for_selection'] is False
        assert digest(final/'best_trainable.pt')==report['checkpoint_sha256']
        weight=str(final/'final_adapter') if a.model=='uniwm' else str(final/'best_trainable.pt');weight_key='final_adapter' if a.model=='uniwm' else 'trainable_validation_best'
    weight_files=([Path(weight)] if a.model!='uniwm' else [Path(weight)/p for p in ['adapter_model.safetensors','adapter_config.json','tokenizer.json']])
    frozen(out/'config.json',dict(manifest_sha256=digest(r/'v2/eval_manifest.json'),script_sha256=digest(__file__),
        backend_sha256=digest(Path(__file__).with_name('suite_backends.py')),model=a.model,stage=a.stage,
        weight=weight,weight_key=weight_key,weight_hashes={str(p):digest(p) for p in weight_files},
        precision='UniWM fp16 with native overflow guards; NWM/RAE bf16 autocast with fp32 parameters',
        target_images_in_inputs=False,history_rule='stationary prefix in seven; observed prefix in factual',
        future_goal_image_in_inputs=False))
    records={c['key']:verify_record(out,c) for c in cases}
    if all(records.values()) and (out/'complete.json').exists():print('EVAL_ALREADY_COMPLETE',a.model,a.stage);return
    from suite_backends import ReConBackend,UniWMBackend,seed_all
    if a.model=='uniwm':backend=UniWMBackend(dict(checkpoint=weight),a.root)
    else:
        backend=ReConBackend(dict(model=a.model,checkpoint=CHECKPOINTS[a.model]),a.root)
        if a.stage=='final':
            state=torch.load(final/'best_trainable.pt',map_location='cpu',weights_only=False)
            assert state['step']==report['step'] and state['model_name']==a.model
            expected={n for n,p in backend.model.named_parameters() if p.requires_grad}
            assert set(state['trainable'])==expected
            missing,unexpected=backend.model.load_state_dict(state['trainable'],strict=False)
            assert not unexpected and not set(missing)&expected
            for name,value in state['trainable'].items():assert torch.equal(dict(backend.model.named_parameters())[name].detach().cpu(),value),name
            atomic(out/'loaded_parameters.json',dict(status='passed',checkpoint=weight,step=report['step'],trainable_tensors_verified=len(state['trainable']),all_tensors_match=True))
            del state;gc.collect()
        backend.model.eval();backend.codec.eval().requires_grad_(False)
    torch.set_num_threads(2)
    # Reconstruction controls are frozen in baseline and reused after tuning.
    control_root=r/'v2/codec_controls'/a.model;control_root.mkdir(parents=True,exist_ok=True)
    refs=read(r/'v2/test_references/manifest.json')
    with torch.inference_mode():
        if a.stage=='baseline':
            control_rows=[]
            for i,(tid,entry) in enumerate(sorted(refs['references'].items())):
                for j,angle in enumerate(m['angles']):
                    ref=entry['reference_views'][str(angle)];key=f'{tid}/{angle:+d}';p=control_root/(key+'.png');meta=control_root/(key+'.json')
                    if meta.exists():
                        row=read(meta);assert row['source_sha256']==ref['sha256'] and digest(p)==row['image_sha256'];control_rows.append(row);continue
                    seed_all(20261023+i*101+j)
                    if a.model=='uniwm':im=backend.sanitize(Path(ref['path']),False)
                    else:
                        with Image.open(ref['path']) as raw:x=backend.misc.transform(raw.convert('RGB'))[None].cuda()
                        with torch.autocast('cuda',dtype=torch.bfloat16):
                            if a.model=='nwm':z=backend.codec.encode(x).latent_dist.sample()*.18215;pixels=backend.codec.decode(z/.18215).sample*.5+.5
                            else:z=backend.codec.encode(x*.5+.5);pixels=backend.codec.decode(z)
                        assert torch.isfinite(pixels).all();array=(pixels[0].float().clamp(0,1).permute(1,2,0).cpu().numpy()*255).round().astype('uint8');im=Image.fromarray(array)
                    put_png(p,im);row=dict(key=key,state_id=tid,angle_deg=angle,source_path=ref['path'],source_sha256=ref['sha256'],
                        output_path=str(p),image_sha256=digest(p),seed=20261023+i*101+j,world_model_prediction=False)
                    atomic(meta,row);control_rows.append(row)
            atomic(control_root/'complete.json',dict(status='passed',rows=control_rows,count=14))
        else:assert read(control_root/'complete.json')['count']==14
        for index,c in enumerate(cases):
            if records[c['key']]:continue
            allowed=set(c['history_paths']+[c['start_image'],c['goal_image']])
            def resolve(path):
                assert path in allowed,'A future or unregistered frame reached inference'
                assert digest(path)==m['input_hashes'][path];return Path(path)
            capture={};hook=None;original_generate=None
            if a.model!='uniwm':
                def before(module,args,kw):
                    if not capture:
                        capture.update(conditioning_sha256=tensor_hash(kw['x_cond']),initial_noise_sha256=tensor_hash(args[0]),
                            observed_action=kw['y'].detach().float().cpu().tolist(),observed_rel_t=kw['rel_t'].detach().float().cpu().tolist())
                hook=backend.model.register_forward_pre_hook(before,with_kwargs=True)
            else:
                original_generate=backend.model.generate
                action_ids=set(backend.processor.tokenizer.convert_tokens_to_ids(c['action_tokens']))
                def generate(*args,**kw):
                    ids=kw['input_ids'].clone();mask=torch.zeros_like(ids,dtype=torch.bool)
                    for token in action_ids:mask|=(ids==token)
                    assert int(mask.sum())==3
                    ids[mask]=-1
                    capture.update(non_action_input_ids_sha256=tensor_hash(ids),conditioning_pixels_sha256=tensor_hash(kw['pixel_values']))
                    return original_generate(*args,**kw)
                backend.model.generate=generate
            try:
                im,feedback,info=backend.predict(c,resolve)
                p=out/(c['key']+'.png');put_png(p,im);info.pop('_anchor',None)
                if a.model!='uniwm':
                    expected=backend.encode_action(c['action'])
                    assert np.allclose(capture['observed_action'],[expected],rtol=0,atol=1e-7)
                    assert abs(capture['observed_rel_t'][0]-1/128)<1e-9
                assert capture
                record=dict(status='ok',case=c,image_sha256=digest(p),**info,**capture)
                del feedback
            except ValueError as e:
                if a.model!='uniwm' or str(e) not in ['Expected one generated image segment','Incomplete generated image']:raise
                record=dict(status='parse_failed',case=c,error=str(e),input_hashes={p:m['input_hashes'][p] for p in allowed},**capture)
            finally:
                if hook is not None:hook.remove()
                if original_generate is not None:backend.model.generate=original_generate
            atomic(out/(c['key']+'.json'),record);records[c['key']]=record
            atomic(out/'progress.json',dict(stage='generation',complete=sum(v is not None for v in records.values()),total=72,
                successful=sum(v is not None and v['status']=='ok' for v in records.values()),last=c['key']))
            print('EVAL',a.model,a.stage,index+1,record['status'],c['key'],flush=True)
    # Actual conditioning, and the first sampler noise, must match siblings.
    for tid in refs['references']:
        for rep in range(3):
            group=[records[c['key']] for c in cases if c['experiment']=='seven' and c['state_id']==tid and c['repeat']==rep]
            fields=['non_action_input_ids_sha256','conditioning_pixels_sha256'] if a.model=='uniwm' else ['conditioning_sha256','initial_noise_sha256']
            for field in fields:assert len({rec[field] for rec in group})==1,(tid,rep,field)
    successful=sum(rec['status']=='ok' for rec in records.values());assert successful>0
    atomic(out/'complete.json',dict(status='passed',records=72,successful=successful,parse_failed=72-successful,
        config_sha256=digest(out/'config.json'),input_record_hashes={k:digest(out/(k+'.json')) for k in records},
        paired_sibling_conditioning_passed=True,codec_control_count=14,world_model_optimizer_steps=report['step'],optimizer_steps_completed=report['optimizer_steps_completed'],weight_selection='minimum fixed-noise validation loss; unchanged72-case test'))


if __name__=='__main__':main()
