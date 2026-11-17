"""Action-scale extension; old252 records reused and only144 new predictions per model."""
import argparse
import copy
import gc
import hashlib
from pathlib import Path
import shutil
import numpy as np
from PIL import Image
import torch
from suite_common import read, digest, atomic, frozen
from angle_scale_common import root, selected_weights, ARMS, NEW_ANGLES


def thash(t):
    t=t.detach().contiguous().cpu()
    return hashlib.sha256(str((str(t.dtype),list(t.shape))).encode()+t.view(torch.uint8).numpy().tobytes()).hexdigest()


def put_png(path,image):
    path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix('.tmp')
    image.save(temp,format='PNG');temp.replace(path)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--study',type=Path,required=True)
    ap.add_argument('--root',type=Path,required=True);ap.add_argument('--model',choices=['nwm','raenwm','uniwm'],required=True)
    ap.add_argument('--cohort',choices=['confirmation'],required=True)
    ap.add_argument('--smoke',action='store_true');a=ap.parse_args()
    out=root(a.study);assert read(out/'render_complete.json')['status']=='passed'
    assert read(out/'execution_freeze.json')['status']=='frozen'
    for path,sha in read(out/'execution_freeze.json')['source_hashes'].items():assert digest(path)==sha
    p=read(out/'protocol.json');weights=selected_weights(a.study)
    for name,value in weights.items():assert value['sha256']==p['selected_checkpoints'][name]['sha256']
    if a.cohort=='confirmation':
        manifest=read(out/'confirmation_manifest.json');cases=manifest['cases'];assert len(cases)==396
    else:
        assert a.model=='uniwm';original=read(a.study/'v2/eval_manifest.json')
        goals=read(out/'mechanism_goals.json')['goals'];cases=[]
        for arm,spec in ARMS.items():
            for source in original['cases']:
                if source['experiment']!='seven':continue
                c=copy.deepcopy(source);c['original_key']=c['key'];c['key']=arm+'/'+c['key'];c['arm']=arm
                c['scene_id']='development_mp3d';c['goal_image']=goals[c['state_id']][spec['goal']]['path'];cases.append(c)
        assert len(cases)==252
        manifest=dict(cases=cases,input_hashes={path:digest(path) for c in cases for path in c['history_paths']+[c['start_image'],c['goal_image']]})
        frozen(out/'mechanism_manifest.json',manifest)
    for path,sha in manifest['input_hashes'].items():assert digest(path)==sha
    dest=out/a.cohort/a.model;dest.mkdir(parents=True,exist_ok=True)
    frozen(dest/'config.json',dict(protocol_sha256=digest(out/'protocol.json'),execution_sha256=digest(out/'execution_freeze.json'),
        script_sha256=digest(__file__),model=a.model,cohort=a.cohort,weights=weights[a.model],records=len(cases),
        updates=0,native_sampling_retained=a.cohort=='confirmation',input_hashes=manifest['input_hashes']))
    records={}
    for c in cases:
        path=dest/(c['key']+'.json')
        if path.exists():
            d=read(path);assert d['case']==c
            if d['status']=='ok':assert digest(path.with_suffix('.png'))==d['image_sha256']
            else:assert d['status']=='parse_failed'
            records[c['key']]=d
    if a.smoke and (dest/'smoke.json').exists():
        assert read(dest/'smoke.json')['status']=='passed';return
    if len(records)==len(cases) and (dest/'complete.json').exists():return
    from suite_backends import ReConBackend,UniWMBackend,seed_all
    from train_habitat_model import CHECKPOINTS
    if a.model=='uniwm':
        backend=UniWMBackend(dict(checkpoint=weights[a.model]['path']),a.root)
        from uniwm.extra_tokens import unwrap
        hf=unwrap(backend.model);core=hf.model;tok=backend.processor.tokenizer
        assert backend.audit['input_embedding_matches_checkpoint'] and backend.audit['active_output_head_matches_checkpoint']
        from safetensors import safe_open
        count=0
        with safe_open(str(Path(weights[a.model]['path'])/'adapter_model.safetensors'),framework='pt',device='cpu') as saved:
            for name,param in backend.model.named_parameters():
                if '.lora_A.default.' not in name and '.lora_B.default.' not in name:continue
                key=name.replace('.modules_to_save.default.','.')
                for part in ['lora_A','lora_B']:key=key.replace(f'.{part}.default.',f'.{part}.')
                assert torch.equal(param.detach().cpu(),saved.get_tensor(key).to(param.dtype));count+=1
        assert count==256
        atomic(dest/'loaded_parameters.json',dict(status='passed',lora_tensors=count,head_matches=True,input_embedding_matches=True,weight=weights[a.model]))
    else:
        backend=ReConBackend(dict(model=a.model,checkpoint=CHECKPOINTS[a.model]),a.root)
        state=torch.load(weights[a.model]['path'],map_location='cpu',weights_only=False)
        expected={n for n,p0 in backend.model.named_parameters() if p0.requires_grad}
        assert set(state['trainable'])==expected and state['step']==weights[a.model]['step']
        missing,unexpected=backend.model.load_state_dict(state['trainable'],strict=False)
        assert not unexpected and not set(missing)&expected
        for name,value in state['trainable'].items():assert torch.equal(dict(backend.model.named_parameters())[name].detach().cpu(),value)
        atomic(dest/'loaded_parameters.json',dict(status='passed',trainable_tensors=len(expected),weight=weights[a.model]))
        del state;gc.collect()
    backend.model.eval().requires_grad_(False);torch.set_num_threads(2)

    # Frozen-codec controls for every new reference, not a claimed noise floor.
    if a.cohort=='confirmation':
        controls=[];refs=read(out/'references_manifest.json')['references']
        prior_rows=read(a.study/'v2/paper_confirmation_v1/codec_controls'/a.model/'complete.json')['rows']
        prior={(r['state_id'],r['angle_deg']):r for r in prior_rows}
        for si,(sid,entry) in enumerate(refs.items()):
            for angle,ref in entry['reference_views'].items():
                if (sid,int(angle)) in prior:
                    d=prior[(sid,int(angle))];assert digest(d['output_path'])==d['image_sha256']
                    assert d['source_sha256']==ref['sha256'];controls.append(d);continue
                path=out/'codec_controls'/a.model/sid/f'yaw_{int(angle):+d}.png';meta=path.with_suffix('.json')
                if meta.exists():d=read(meta);assert digest(path)==d['image_sha256'];controls.append(d);continue
                seed_all(20261027+si*101+int(angle))
                with torch.inference_mode():
                    if a.model=='uniwm':im=backend.sanitize(Path(ref['path']),False)
                    else:
                        with Image.open(ref['path']) as raw:x=backend.misc.transform(raw.convert('RGB'))[None].cuda()
                        with torch.autocast('cuda',dtype=torch.bfloat16):
                            if a.model=='nwm':z=backend.codec.encode(x).latent_dist.sample()*.18215;pixels=backend.codec.decode(z/.18215).sample*.5+.5
                            else:z=backend.codec.encode(x*.5+.5);pixels=backend.codec.decode(z)
                        assert torch.isfinite(pixels).all()
                        im=Image.fromarray((pixels[0].float().clamp(0,1).permute(1,2,0).cpu().numpy()*255).round().astype('uint8'))
                put_png(path,im);d=dict(state_id=sid,scene_id=entry['scene_id'],angle_deg=int(angle),source_path=ref['path'],
                    source_sha256=ref['sha256'],output_path=str(path),image_sha256=digest(path));atomic(meta,d);controls.append(d)
        atomic(out/'codec_controls'/a.model/'complete.json',dict(status='passed',count=len(controls),rows=controls))

    for index,c in enumerate(cases):
        if a.smoke and index>=11:break
        if c['key'] in records:continue
        capture={};hooks=[];original_generate=None;original_multinomial=None
        allowed=set(c['history_paths']+[c['start_image'],c['goal_image']])
        def resolve(path):
            assert path in allowed,'Unregistered/future image attempted as input'
            assert digest(path)==manifest['input_hashes'][path];return Path(path)
        if a.model!='uniwm':
            def before(module,args,kw):
                if not capture:capture.update(conditioning_sha256=thash(kw['x_cond']),initial_noise_sha256=thash(args[0]),
                    observed_action=kw['y'].detach().float().cpu().tolist(),observed_rel_t=kw['rel_t'].detach().float().cpu().tolist())
            hooks.append(backend.model.register_forward_pre_hook(before,with_kwargs=True))
        else:
            spec=ARMS[c.get('arm','native')]
            backend.sampling.update(top_k=spec['top_k'],top_p=spec['top_p'])
            original_generate=backend.model.generate;original_multinomial=torch.multinomial
            action_ids=tok.convert_tokens_to_ids(c['action_tokens']);prefix={};supports=[]
            def embedding_before(module,args):
                if 'codes' in prefix:return
                ids=args[0].detach();starts=(ids[0]==backend.model.config.boi_token_id).nonzero().flatten().tolist()
                if len(starts)!=3:return
                assert all(int((ids[0]==value).sum())==1 for value in action_ids)
                other=ids.clone()
                for value in action_ids:other[other==value]=-1
                prefix['codes']=[ids[0,b+1:b+785].cpu().numpy() for b in starts]
                capture.update(post_vq_non_action_sha256=thash(other),post_vq_input_sha256=thash(ids),action_ids=action_ids)
            def generated(*args,**kw):
                capture['conditioning_pixels_sha256']=thash(kw['pixel_values'])
                result=original_generate(*args,**kw)
                seq=result[0];seq=seq[0] if seq.ndim==2 else seq
                starts=(seq==backend.model.config.boi_token_id).nonzero().flatten()
                if len(starts)!=1:return result
                b=int(starts[0]);generated_codes=seq[b+1:b+785].detach().cpu().numpy()
                if len(generated_codes)!=784:return result
                assert 'codes' in prefix
                capture['generated_image_token_ids']=generated_codes.tolist()
                capture['copy_tokens']={name:dict(equal=int((generated_codes==codes).sum()),total=784)
                    for name,codes in zip(['start','goal','current'],prefix['codes'])}
                return result
            def multinomial(probs,*args,**kw):
                if probs.ndim==2 and probs.shape[0]==1 and probs.shape[1]==len(tok):supports.append(int((probs[0]>0).sum()))
                return original_multinomial(probs,*args,**kw)
            hooks.append(core.embed_tokens.register_forward_pre_hook(embedding_before))
            backend.model.generate=generated;torch.multinomial=multinomial
        path=dest/(c['key']+'.png')
        try:
            with torch.inference_mode():im,feedback,info=backend.predict(c,resolve)
            info.pop('_anchor',None);del feedback
            put_png(path,im)
            if a.model!='uniwm':
                assert np.allclose(capture['observed_action'],[backend.encode_action(c['action'])],rtol=0,atol=1e-7)
                assert abs(capture['observed_rel_t'][0]-1/128)<1e-9
            else:
                capture['sampling_support']=dict(steps=len(supports),single_candidate_steps=sum(x==1 for x in supports),
                    min_support=min(supports) if supports else None,max_support=max(supports) if supports else None)
                capture['sampling']=dict(backend.sampling)
            record=dict(status='ok',case=c,image_sha256=digest(path),**info,**capture)
            if c.get('arm')=='native':
                old=a.study/'v2/uniwm_symmetric_aligned_v3/evaluation'/(c['original_key']+'.png')
                record['original_prediction_sha256']=digest(old);record['native_exact_replay']=digest(path)==digest(old)
                atomic(path.with_suffix('.json'),record)
                assert record['native_exact_replay'],'Native control failed exact reproducibility'
        except ValueError as error:
            if a.model!='uniwm' or str(error) not in ['Expected one generated image segment','Incomplete generated image']:raise
            record=dict(status='parse_failed',case=c,error=str(error),**capture)
        finally:
            for hook in hooks:hook.remove()
            if original_generate is not None:backend.model.generate=original_generate
            if original_multinomial is not None:torch.multinomial=original_multinomial
        atomic(path.with_suffix('.json'),record);records[c['key']]=record
        atomic(dest/'progress.json',dict(completed=len(records),total=len(cases),successful=sum(x['status']=='ok' for x in records.values()),last=c['key']))
        print('GENERATION',a.cohort,a.model,len(records),len(cases),c['key'],record['status'],flush=True)
    if a.smoke:
        selected=[records[c['key']] for c in cases[:11] if c['target_yaw_deg'] in NEW_ANGLES]
        assert len(selected)==4 and any(c['status']=='ok' for c in selected),'All four new smoke outputs failed parsing'
        atomic(dest/'smoke.json',dict(status='passed',new_records=4,reused_records=7,successful=sum(c['status']=='ok' for c in selected),
            records_in_total_budget=True,optimizer_updates=0));return
    groups={}
    for c in cases:groups.setdefault((c.get('arm','native'),c['state_id'],c['repeat']),[]).append(records[c['key']])
    for group in groups.values():
        ok=[x for x in group if x['status']=='ok'];assert len(group)==11
        fields=['post_vq_non_action_sha256','conditioning_pixels_sha256'] if a.model=='uniwm' else ['conditioning_sha256','initial_noise_sha256']
        for field in fields:assert len({x[field] for x in ok})<=1,(field,group[0]['case']['key'])
    atomic(dest/'complete.json',dict(status='passed',records=len(records),successful=sum(x['status']=='ok' for x in records.values()),
        parse_failed=sum(x['status']!='ok' for x in records.values()),paired_inputs_passed=True,
        config_sha256=digest(dest/'config.json'),record_hashes={k:digest(dest/(k+'.json')) for k in records},optimizer_updates=0))


if __name__=='__main__':main()
