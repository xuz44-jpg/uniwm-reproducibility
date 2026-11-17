"""Trace existing frozen cases through native generation. No optimizer or fitting.

Budget: first two forwards for the original 2x7 repeat0 cases; replay 0,+7,-7
at position0 and 0 at position1; shared-prefix diagnostics on those same cases.
Only diagnostic files are written. Completed predictions/weights remain intact.
"""
import argparse
import gc
import hashlib
import inspect
import io
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from safetensors import safe_open
from suite_common import atomic,digest,frozen,read
from suite_backends import UniWMBackend


def thash(t):
    t=t.detach().contiguous().cpu()
    return hashlib.sha256(str((str(t.dtype),list(t.shape))).encode()+t.view(torch.uint8).numpy().tobytes()).hexdigest()


def export_key(name):
    key=name.replace('.modules_to_save.default.','.')
    for part in ['lora_A','lora_B']:key=key.replace(f'.{part}.default.',f'.{part}.')
    return key


class PrefillCaptured(Exception):pass


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--study',type=Path,required=True);ap.add_argument('--root',type=Path,required=True);a=ap.parse_args()
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    run=a.study/'v2/uniwm_symmetric_aligned_v3';out=run/'loading_action_audit_v1';out.mkdir(exist_ok=True)
    if (out/'gpu_complete.json').exists():
        assert read(out/'gpu_complete.json')['script_sha256']==digest(__file__);print('AUDIT_ALREADY_COMPLETE');return
    assert read(out/'cpu_complete.json')['status']=='passed'
    ckpt=run/'training/final_adapter';eval_cfg=read(run/'evaluation/config.json');assert eval_cfg['weight']==str(ckpt)
    assert digest(ckpt/'adapter_model.safetensors')==eval_cfg['weight_hashes'][str(ckpt/'adapter_model.safetensors')]
    frozen(out/'gpu_config.json',dict(script_sha256=digest(__file__),checkpoint=str(ckpt),checkpoint_sha256=digest(ckpt/'adapter_model.safetensors'),
        manifest_sha256=digest(a.study/'v2/eval_manifest.json'),prefix_cases=14,complete_replays=4,
        original_positions=2,repeat=0,optimizer_updates=0,weights_modified=False,scored_as_new_science=False))
    backend=UniWMBackend(dict(checkpoint=str(ckpt)),a.root);model=backend.model
    model.eval().requires_grad_(False)
    from uniwm.extra_tokens import unwrap
    from transformers.generation.logits_process import TemperatureLogitsWarper,TopPLogitsWarper
    hf=unwrap(model);core=hf.model;tok=backend.processor.tokenizer
    loaded={}
    with safe_open(str(ckpt/'adapter_model.safetensors'),framework='pt',device='cpu') as saved:
        for name,param in model.named_parameters():
            if '.lora_A.default.' not in name and '.lora_B.default.' not in name:continue
            expected=saved.get_tensor(export_key(name)).to(param.dtype);observed=param.detach().cpu()
            assert torch.equal(expected,observed),name
            loaded[name]=dict(matched=True,dtype=str(param.dtype),sha256=thash(observed))
    assert len(loaded)==256
    lora_modules={name:dict(disabled=bool(layer.disable_adapters),active=list(layer.active_adapters),scaling=dict(layer.scaling))
        for name,layer in model.named_modules() if hasattr(layer,'lora_A') and hasattr(layer,'disable_adapters')}
    assert len(lora_modules)==128 and all(not x['disabled'] and x['active']==['default'] for x in lora_modules.values())
    atomic(out/'loaded_parameters.json',dict(status='passed',embedding_and_head=backend.audit,lora_tensors=loaded,lora_modules=lora_modules,
        input_embedding_matches_saved=backend.audit['input_embedding_matches_checkpoint'],head_matches_saved=backend.audit['active_output_head_matches_checkpoint'],
        full_trainable_tensors_verified=257,active_adapters=model.active_adapters,
        native_class_source={inspect.getfile(type(hf)):digest(inspect.getfile(type(hf))),inspect.getfile(type(core)):digest(inspect.getfile(type(core)))},optimizer_updates=0))
    print('LOADED_PARAMETERS_VERIFIED',257,flush=True)
    cases=read(a.study/'v2/eval_manifest.json')['cases']
    cases=[c for c in cases if c['experiment']=='seven' and c['repeat']==0]
    assert len(cases)==14
    image_ids=torch.tensor(core.vocabulary_mapping.image_token_ids,device='cuda',dtype=torch.long)
    internal={};prefix_reports=[];replay_reports=[]

    def trace(c,full):
        row={};internal_row={};calls=[];samples=[];raw_logits={};hooks=[]
        action_ids=tok.convert_tokens_to_ids(c['action_tokens'])
        original_generate=model.generate;original_multinomial=torch.multinomial
        def generated(*args,**kw):
            ids=kw['input_ids'].detach().clone();pos=[]
            for value in action_ids:
                where=(ids[0]==value).nonzero().flatten();assert len(where)==1;pos.append(int(where[0]))
            assert pos==sorted(pos)
            row.update(action_ids=action_ids,action_positions=pos,input_length=ids.shape[1],input_ids_sha256=thash(ids),
                attention_mask_all_ones=bool(torch.all(kw['attention_mask']==1)),sampling=dict(backend.sampling))
            internal_row['outer_ids']=ids.cpu();row['input_action_text']=c['action_text']
            result=original_generate(*args,**kw)
            gen=result[0];gen=gen[0] if gen.ndim==2 else gen
            internal_row['generated_ids']=gen.detach().cpu()
            return result
        def forward_before(module,args,kw):
            if calls:return
            ids=kw['input_ids'];cache=kw.get('past_key_values');cp=kw.get('cache_position');positions=kw.get('position_ids')
            row.update(first_forward_input_length=ids.shape[1],first_forward_action_ids=[int(ids[0,p]) for p in row['action_positions']],
                initial_cache_length=int(cache.get_seq_length()) if cache is not None else 0,
                cache_type=type(cache).__name__ if cache is not None else None,use_cache=kw.get('use_cache'),
                cache_position_start=int(cp[0]) if cp is not None else None,cache_position_end=int(cp[-1]) if cp is not None else None,
                action_position_ids=[int(positions[0,p]) for p in row['action_positions']] if positions is not None else None)
            assert row['initial_cache_length']==0 and row['cache_position_start']==0
            assert row['first_forward_action_ids']==action_ids
        def embedding_before(module,args):
            if 'network_prefix' in internal_row:return
            ids=args[0].detach().cpu();pos=row['action_positions']
            assert ids.shape[1]==row['input_length'] and ids[0,pos].tolist()==action_ids
            other=ids.clone();other[0,pos]=-1
            row.update(post_vq_action_ids=ids[0,pos].tolist(),post_vq_non_action_sha256=thash(other),post_vq_input_sha256=thash(ids),
                action_embedding_norms=module.weight[action_ids].detach().float().norm(dim=-1).cpu().tolist())
            starts=(ids[0]==model.config.boi_token_id).nonzero().flatten().tolist();assert len(starts)==3
            internal_row['network_prefix']=ids;internal_row['current_image_codes']=ids[0,starts[-1]+1:starts[-1]+1+784].clone()
            assert len(internal_row['current_image_codes'])==784
        def layer_before(module,args,kw):
            if 'action_visible_to_last_query' in row:return
            mask=kw.get('attention_mask')
            if mask is None:
                visible=[p<row['input_length'] for p in row['action_positions']]
            else:
                values=mask[0,0,-1,row['action_positions']].detach().cpu()
                visible=values.tolist() if values.dtype==torch.bool else (values==0).tolist()
                row['actual_action_mask_bias']=values.tolist()
            row.update(action_visible_to_last_query=visible,layer_mask_shape=list(mask.shape) if mask is not None else None)
            assert all(visible)
        def after_forward(module,args,kw,result):
            index=len(calls);ids=kw['input_ids'];calls.append(dict(index=index,input_length=ids.shape[1],last_input_token=int(ids[0,-1])))
            if index in [0,1,17,101,401,701,784]:raw_logits[str(index)]=result.logits[0,-1].detach().float().cpu().numpy()
            if not full and index==1:raise PrefillCaptured()
        def multinomial(probs,*args,**kw):
            result=original_multinomial(probs,*args,**kw)
            if probs.ndim==2 and probs.shape[0]==1 and probs.shape[1]==len(tok):
                samples.append(dict(index=len(samples),support=int((probs[0]>0).sum()),maximum_probability=float(probs[0].max()),
                    argmax_id=int(probs[0].argmax()),chosen_id=int(result.flatten()[0])))
            return result
        hooks.append(hf.register_forward_pre_hook(forward_before,with_kwargs=True))
        hooks.append(core.embed_tokens.register_forward_pre_hook(embedding_before))
        hooks.append(core.layers[0].register_forward_pre_hook(layer_before,with_kwargs=True))
        hooks.append(hf.register_forward_hook(after_forward,with_kwargs=True))
        model.generate=generated;torch.multinomial=multinomial
        try:
            allowed=set(c['history_paths']+[c['start_image'],c['goal_image']])
            def resolve(path):assert path in allowed;return Path(path)
            with torch.inference_mode():im,_,info=backend.predict(c,resolve)
            assert full
            png=io.BytesIO();im.save(png,format='PNG');png_sha=hashlib.sha256(png.getvalue()).hexdigest()
            original=run/'evaluation'/(c['key']+'.png')
            assert png_sha==digest(original),'Native deterministic replay did not reproduce completed PNG'
            dest=out/'replays'/(c['key']+'.png');dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(png.getvalue())
            gen=internal_row['generated_ids'];b=(gen==model.config.boi_token_id).nonzero().flatten();assert len(b)==1
            b=int(b[0]);codes=gen[b+1:b+785];assert len(codes)==784
            image_samples=samples[b+1:b+785];assert len(image_samples)==784
            row.update(reproduces_completed_png=True,image_sha256=png_sha,generated_image_start=b,image_tokens=784,
                generated_ids_sha256=thash(gen),copy_current_image_codes=int((codes==internal_row['current_image_codes']).sum()),
                copy_current_image_codes_total=784,image_sampling_support_one=sum(s['support']==1 for s in image_samples),
                image_sampling_maximum_probability_mean=float(np.mean([s['maximum_probability'] for s in image_samples])),
                image_sampling_support_median=float(np.median([s['support'] for s in image_samples])),sampling_steps=samples)
            internal_row['image_codes']=codes
        except PrefillCaptured:
            assert not full;row['stopped_after_two_forwards']=True
        finally:
            model.generate=original_generate;torch.multinomial=original_multinomial
            for hook in hooks:hook.remove()
        assert len(calls)>=2 and row['post_vq_action_ids']==action_ids and row['attention_mask_all_ones']
        assert row['input_length']+804<=4096
        row.update(key=c['key'],full_replay=full,forward_calls=len(calls),first_two_calls=calls[:2],optimizer_updates=0)
        dest=out/('replay_traces' if full else 'prefix_traces')/(c['key']+'.json');atomic(dest,row)
        npz=dest.with_suffix('.npz');npz.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(npz,network_prefix=internal_row['network_prefix'].numpy(),
            generated_ids=internal_row.get('generated_ids',torch.empty(0,dtype=torch.long)).numpy(),**{'logits_'+k:v for k,v in raw_logits.items()})
        print('TRACE',c['key'],'full' if full else 'two-forward',row.get('copy_current_image_codes'),row.get('image_sampling_support_one'),flush=True)
        return row,internal_row

    for c in cases:
        dest=out/'prefix_traces'/(c['key']+'.json');npz=dest.with_suffix('.npz')
        if dest.exists() and npz.exists():
            row=read(dest)
            with np.load(npz) as z:ir=dict(network_prefix=torch.from_numpy(z['network_prefix'].copy()))
        else:row,ir=trace(c,False)
        prefix_reports.append(row);internal[c['key']]=ir
        atomic(out/'progress.json',dict(phase='prefix',complete=len(prefix_reports),total=14))
    for tid in ['room09_traj00','room09_traj01']:
        group=[r for r in prefix_reports if '/'+tid+'/' in r['key']]
        assert len(group)==7 and len(set(r['post_vq_non_action_sha256'] for r in group))==1
        assert len(set(r['post_vq_input_sha256'] for r in group))==7
    for tid,angle in [('room09_traj00',0),('room09_traj00',7),('room09_traj00',-7),('room09_traj01',0)]:
        c=next(c for c in cases if c['state_id']==tid and c['target_yaw_deg']==angle)
        dest=out/'replay_traces'/(c['key']+'.json');npz=dest.with_suffix('.npz')
        if dest.exists() and npz.exists():
            row=read(dest)
            with np.load(npz) as z:ir=dict(network_prefix=torch.from_numpy(z['network_prefix'].copy()),generated_ids=torch.from_numpy(z['generated_ids'].copy()))
        else:row,ir=trace(c,True)
        replay_reports.append(row);internal[c['key']]=ir
        atomic(out/'progress.json',dict(phase='native_replays',complete=len(replay_reports),total=4))

    # Shared generated prefix: test whether swapping just the action changes
    # logits, rather than comparing two trajectories that have already diverged.
    offsets=[0,16,100,400,700,783];teacher=[]
    for tid in ['room09_traj00','room09_traj01']:
        zero=next(c for c in cases if c['state_id']==tid and c['target_yaw_deg']==0);gen=internal[zero['key']]['generated_ids']
        b=int((gen==model.config.boi_token_id).nonzero().flatten()[0]);wanted=[b+1+i for i in offsets]
        zero_logp=None;zero_raw=None
        for c in sorted([c for c in cases if c['state_id']==tid],key=lambda c:(c['target_yaw_deg']!=0,c['target_yaw_deg'])):
            dest=out/'shared_prefix'/(c['key']+'.json')
            ids=internal[c['key']]['network_prefix'].cuda();full=torch.cat([ids,gen[None,:max(wanted)+1].cuda()],dim=1)
            select=torch.zeros_like(full,dtype=torch.bool);select[0,[ids.shape[1]+j-1 for j in wanted]]=True
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
                raw=model(input_ids=full,attention_mask=torch.ones_like(full),loss_positions=select,use_cache=False).logits.float()
            assert raw.shape[0]==len(offsets) and torch.isfinite(raw).all()
            values=raw[:,image_ids];lp=torch.log_softmax(values,dim=-1)
            if zero_logp is None:zero_logp=lp.clone();zero_raw=values.clone()
            mask=torch.ones_like(raw,dtype=torch.bool);mask[:,image_ids]=False;scores=raw.masked_fill(mask,torch.finfo(raw.dtype).min)
            scores=TemperatureLogitsWarper(.7)(full,scores);scores=TopPLogitsWarper(.9)(full,scores);probs=torch.softmax(scores,dim=-1)
            rec=dict(key=c['key'],image_offsets=offsets,mean_abs_image_logit_change_from_zero=float((values-zero_raw).abs().mean()),
                max_abs_image_logit_change_from_zero=float((values-zero_raw).abs().max()),image_distribution_kl_to_zero=float((lp.exp()*(lp-zero_logp)).sum(-1).mean()),
                same_image_argmax_as_zero=int((values.argmax(-1)==zero_raw.argmax(-1)).sum()),
                processed_sampling_support=(probs>0).sum(-1).cpu().tolist(),processed_sampling_max_probability=probs.max(-1).values.cpu().tolist(),
                target_generated_zero_code_probability=probs.gather(-1,gen[wanted,None].cuda()).flatten().cpu().tolist(),
                actual_action_tokens=tok.convert_tokens_to_ids(c['action_tokens']),optimizer_updates=0,shared_future_prefix=True)
            atomic(dest,rec);teacher.append(rec);print('SHARED_PREFIX',c['key'],rec['mean_abs_image_logit_change_from_zero'],rec['processed_sampling_support'],flush=True)
            del full,raw,values,lp,scores,probs;gc.collect()
    atomic(out/'gpu_complete.json',dict(status='passed',script_sha256=digest(__file__),checkpoint=str(ckpt),checkpoint_sha256=digest(ckpt/'adapter_model.safetensors'),
        all_257_trainables_match_export=True,all_128_lora_modules_enabled=True,all_action_tokens_reach_embedding=True,
        no_action_masking=True,no_cross_action_kv_cache_reuse=True,prefix_case_count=14,native_replay_count=4,
        all_replays_reproduce_completed_png=True,prefix_traces=prefix_reports,native_replays=replay_reports,shared_prefix=teacher,
        optimizer_updates=0,weights_modified=False,original_results_modified=False,new_test_positions=0))
    print('GENERATION_AUDIT_COMPLETE',flush=True)


if __name__=='__main__':main()
