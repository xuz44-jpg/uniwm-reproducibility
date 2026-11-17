"""Resumable, guard-owned 2,000-step adaptation with each native model loss.

Five smoke steps are part of the same 2,000 logical optimizer steps. A speech
preemption discards only work after the most recent atomic checkpoint. No
validation/test pixels are used for optimization or model selection.
"""
import argparse
import copy
import gc
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from types import SimpleNamespace
import numpy as np
from PIL import Image
import torch
from suite_common import atomic,digest,frozen,read


CHECKPOINTS=dict(nwm='external/navigation-world-models/models/nwm/cdit_xl_ego4d_200000.pth.tar',
    raenwm='external/navigation-world-models/models/raenwm/raenwm_b.pth.tar',
    uniwm='external/uniwm-action-pilot/checkpoints/1x200')


def cpu_tree(value):
    if isinstance(value,torch.Tensor):return value.detach().cpu()
    if isinstance(value,dict):return {k:cpu_tree(v) for k,v in value.items()}
    if isinstance(value,list):return [cpu_tree(v) for v in value]
    if isinstance(value,tuple):return tuple(cpu_tree(v) for v in value)
    return value


def save_atomic(path,state):
    tmp=path.with_suffix('.tmp')
    with tmp.open('wb') as f:
        torch.save(state,f);f.flush();os.fsync(f.fileno())
    tmp.replace(path)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model',choices=CHECKPOINTS,required=True)
    ap.add_argument('--root',type=Path,required=True);ap.add_argument('--prepared',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True);ap.add_argument('--stop-at',type=int,default=2000)
    a=ap.parse_args();assert a.stop_at in [5,2000]
    a.out.mkdir(parents=True,exist_ok=True);rows=read(a.prepared/'train.json')
    assert len(rows)==240 and all(r['split']=='train' for r in rows)
    for p,sha in read(a.prepared/'input_image_hashes.json').items():
        if any(p in r['history_paths']+[r['target_image']] for r in rows):assert digest(p)==sha
    seed=20261023;random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    baseline=CHECKPOINTS[a.model]
    baseline_files=([Path(baseline)] if a.model!='uniwm' else
        [Path(baseline)/n for n in ['adapter_model.safetensors','adapter_config.json','tokenizer.json']])
    frozen(a.out/'training_config.json',dict(model=a.model,training_script_sha256=digest(__file__),
        prepared_protocol_sha256=digest(a.prepared/'protocol.json'),train_rows_sha256=digest(a.prepared/'train.json'),
        baseline_hashes={str(p):digest(p) for p in baseline_files},
        backend_sha256=digest(Path(__file__).with_name('suite_backends.py')),
        baseline=baseline,max_steps=2000,smoke_steps_included=5,batch_size=1,learning_rate=1e-5,
        warmup_steps=100,precision='bf16 autocast; fp32 trainable weights',gradient_clip_norm=1.,
        optimizer='AdamW betas .9/.999, weight_decay=0, foreach=False',seed=seed,
        save_every=100,ema='none; evaluate raw final step-2000 weights',
        future_input_policy='only training target enters loss; start image reused for UniWM goal',
        resume='atomic weights+optimizer+RNG; unsaved updates may be recomputed after speech preemption'))
    pending=a.out/'latest.pt';resume=torch.load(pending,map_location='cpu',weights_only=False) if pending.exists() else None
    start=resume['step'] if resume else 0
    if start>=a.stop_at:
        assert (a.out/('complete.json' if start==2000 else 'smoke.json')).exists();print('ALREADY_COMPLETE',start);return
    cache={};tokenized=[];codec=None
    if a.model in ['nwm','raenwm']:
        from suite_backends import ReConBackend
        manifest=dict(model=a.model,checkpoint=baseline)
        backend=ReConBackend(manifest,a.root);model=backend.model;codec=backend.codec
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        codec.eval().requires_grad_(False)
        model.gradient_checkpointing=(a.model=='nwm')
        if a.model=='raenwm':
            repo=a.root/'repos/raenwm';sys.path.insert(0,str(repo/'RAE/src'))
            from RAE.src.stage2.transport.transport import Transport,ModelType,PathType,WeightType
            transport=Transport(model_type=ModelType.VELOCITY,path_type=PathType.LINEAR,loss_type=WeightType.VELOCITY,
                time_dist_type='uniform',time_dist_shift=math.sqrt(codec.latent_dim*16*16/4096),train_eps=1e-3,sample_eps=1e-3)
        else:
            from diffusion import create_diffusion
            diffusion=create_diffusion('')
        # Frozen encoder cache preserves NWM posterior randomness: moments, not
        # one sampled latent. Sampling occurs afresh on every training step.
        paths=sorted({p for r in rows for p in r['history_paths']+[r['target_image']]})
        for i,p in enumerate(paths):
            with Image.open(p) as im:x=backend.misc.transform(im.convert('RGB'))[None].cuda()
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                if a.model=='nwm':
                    dist=codec.encode(x).latent_dist;cache[p]=(dist.mean.cpu(),dist.std.cpu())
                else:cache[p]=codec.encode(x*.5+.5).cpu()
            if i%32==0:atomic(a.out/'progress.json',dict(stage='encode_train_images',encoded=i+1,total=len(paths),optimizer_step=start))
        del x
        # Preserve native frozen Fourier frequencies / positional embeddings.
    else:
        repo=a.root/'repos/uniwm';sys.path.insert(0,str(repo))
        from scripts.load_model import load_model
        from scripts.prompt_builder import build_viz_prompt
        from scripts.action_utils import action_to_text
        args=SimpleNamespace(model='anole',model_ckpt=None,init_adapter=baseline,image_seq_length=784,precision='bf16',
            attn_impl='sdpa',base_model='external/uniwm-action-pilot/models/anole',use_memory_bank_inference=False,
            do_task_level_eval=False,do_train=True,do_single_step_eval=False,do_rollout_eval=False,do_action_probe=False,
            lora_r=16,lora_alpha=32,lora_dropout=.05,lora_targets='q_proj,k_proj,v_proj,o_proj',lm_head_mode='full',
            fp32_mlp_layers='auto',lora_fp32=True)
        mp=load_model(args,None);model=mp['model'];processor=mp['processor']
        from verify_uniwm_embeddings import compare_matrix
        from safetensors import safe_open
        from uniwm.extra_tokens import unwrap
        hf=unwrap(model)
        with safe_open(str(Path(baseline)/'adapter_model.safetensors'),framework='pt',device='cpu') as saved:
            embedding_check=compare_matrix(hf.model.embed_tokens.weight,saved.get_slice('base_model.model.model.embed_tokens.weight'))
            head_module=hf.lm_head
            head_weight=head_module.modules_to_save[head_module.active_adapter].weight
            head_check=compare_matrix(head_weight,saved.get_slice('base_model.model.lm_head.weight'))
        assert embedding_check['matched'] and head_check['matched']
        atomic(a.out/'baseline_embedding_audit.json',dict(status='passed',input_table=embedding_check,output_table=head_check,
            convention='saved tensors cast to effective bf16/fp32 training dtype; no comparison of unlike precisions',initialization_source=baseline))
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        model.enable_input_require_grads();model.config.use_cache=False
        paths=sorted({p for r in rows for p in [r['start_image'],r['current_image'],r['target_image']]})
        for i,p in enumerate(paths):
            with Image.open(p) as im:image=im.convert('RGB').resize((256,256))
            values=processor(text='<image>',images=image,return_tensors='pt')['pixel_values'].cuda().float()
            with torch.no_grad():cache[p]=model.model.model.get_image_tokens(values).reshape(-1).cpu()
            if i%32==0:atomic(a.out/'progress.json',dict(stage='encode_train_images',encoded=i+1,total=len(paths),optimizer_step=start))
        label_processor=copy.deepcopy(processor);label_processor.tokenizer.padding_side='right'
        for r in rows:
            assert action_to_text(r['action'])==r['action_text']
            ids=processor.tokenizer.convert_tokens_to_ids(r['action_tokens'])
            assert all(v!=processor.tokenizer.unk_token_id and v<len(processor.tokenizer) for v in ids)
            sp=r['start_pose'];pose=f'Starting Point Coordinate: x={sp[0]:.3f}, y={sp[1]:.3f}, yaw={sp[2]:.3f}\n'
            prompt=build_viz_prompt(r['action_text'],pose)
            ip=[r['start_image'],r['goal_image'],r['current_image']]
            images=[Image.open(p).convert('RGB').resize((256,256)) for p in ip]
            inputs=processor([prompt],images=images,padding=False,truncation=False,return_tensors='pt')['input_ids']
            input_mask=inputs==model.config.image_token_id;codes=torch.cat([cache[p] for p in ip])
            assert int(input_mask.sum())==len(codes);inputs[input_mask]=codes
            im=Image.open(r['target_image']).convert('RGB').resize((256,256))
            labels=label_processor(['<image>'],images=[im],padding=False,truncation=False,return_tensors='pt')['input_ids'][:,1:]
            label_mask=labels==model.config.image_token_id;assert int(label_mask.sum())==len(cache[r['target_image']]);labels[label_mask]=cache[r['target_image']]
            all_ids=torch.cat([inputs,labels],1);assert all_ids.shape[1]<=4096
            loss_labels=torch.cat([torch.full_like(inputs,-100),labels],1)
            tokenized.append((all_ids,loss_labels))
        del values,cache,images
        processor.save_pretrained(a.out/'processor')
    for p in model.parameters():
        if p.requires_grad:p.data=p.data.float()
    params={n:p for n,p in model.named_parameters() if p.requires_grad};assert params
    optimizer=torch.optim.AdamW(list(params.values()),lr=1e-5,betas=(.9,.999),weight_decay=0.,foreach=False)
    if resume:
        assert set(resume['trainable'])==set(params)
        with torch.no_grad():
            for n,p in params.items():p.copy_(resume['trainable'][n])
        optimizer.load_state_dict(resume['optimizer'])
        random.setstate(resume['rng_python']);np.random.set_state(resume['rng_numpy']);torch.set_rng_state(resume['rng_cpu']);torch.cuda.set_rng_state_all(resume['rng_cuda'])
        del resume;gc.collect()
    probe_names=[n for n in params if 'lora_B' in n] or list(params)[:3]
    probes={n:params[n].detach().cpu().clone() for n in probe_names[:3]}
    model.train()
    if codec is not None:codec.eval()
    losses=[];started=time.monotonic();last_save=start
    def checkpoint(step):
        state=dict(step=step,model_name=a.model,trainable={n:p.detach().cpu() for n,p in params.items()},
            optimizer=cpu_tree(optimizer.state_dict()),rng_python=random.getstate(),rng_numpy=np.random.get_state(),
            rng_cpu=torch.get_rng_state(),rng_cuda=torch.cuda.get_rng_state_all(),training_config_sha256=digest(a.out/'training_config.json'))
        save_atomic(pending,state)
        atomic(a.out/'checkpoint.json',dict(step=step,bytes=pending.stat().st_size,sha256=digest(pending)))
    for step0 in range(start,a.stop_at):
        # Fixed, source-independent permutation for each epoch; resume indexes
        # depend on logical step, not on how often the worker was interrupted.
        epoch,index=divmod(step0,len(rows));perm=np.random.default_rng(seed+epoch).permutation(len(rows));idx=int(perm[index]);r=rows[idx]
        step=step0+1;scale=min(1.,step/100.)*.5*(1+math.cos(math.pi*max(0,step-100)/1900))
        for group in optimizer.param_groups:group['lr']=1e-5*scale
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            if a.model in ['nwm','raenwm']:
                latents=[]
                for p in r['history_paths']+[r['target_image']]:
                    if a.model=='nwm':
                        mu,std=cache[p];mu=mu.cuda();std=std.cuda();z=(mu+std*torch.randn_like(mu))*.18215
                    else:z=cache[p].cuda()
                    latents.append(z)
                x_cond=torch.stack(latents[:4],dim=1);target=latents[4]
                y=torch.tensor(r['encoded_actions'][a.model],device='cuda',dtype=torch.float32)[None]
                rel_t=torch.tensor([1/128],device='cuda',dtype=torch.float32)
                kwargs=dict(y=y,x_cond=x_cond,rel_t=rel_t)
                if a.model=='nwm':
                    t=torch.randint(0,diffusion.num_timesteps,(1,),device='cuda')
                    loss=diffusion.training_losses(model,target,t,kwargs)['loss'].mean()
                else:loss=transport.training_losses(model,target,kwargs)['loss'].mean()
            else:
                input_ids,labels=tokenized[idx];input_ids=input_ids.cuda();labels=labels.cuda()
                shifted=labels[:,1:];keep=shifted!=-100
                positions=torch.cat([keep,torch.zeros_like(keep[:,:1])],1)
                output=model(input_ids=input_ids,attention_mask=torch.ones_like(input_ids),loss_positions=positions,use_cache=False)
                logits=output.logits.float();targets=shifted[keep]
                log_probs=torch.nn.functional.log_softmax(logits,dim=-1)
                nll=-log_probs.gather(-1,targets[:,None]).squeeze(-1)
                loss=(.9*nll-.1*log_probs.mean(-1)).mean()
        assert torch.isfinite(loss).item(),('nonfinite_loss',step)
        loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(list(params.values()),1.,error_if_nonfinite=True)
        optimizer.step()
        record=dict(step=step,loss=float(loss.detach()),gradient_norm=float(norm),sample_id=r['sample_id'],lr=optimizer.param_groups[0]['lr'],
            elapsed_seconds=time.monotonic()-started,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
        with (a.out/'train_log.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
        atomic(a.out/'progress.json',dict(stage='training',optimizer_step=step,target_steps=2000,last=record))
        print('TRAIN',json.dumps(record),flush=True);losses.append(record['loss'])
        if step%100==0 or step==a.stop_at:
            checkpoint(step);last_save=step
    changed={n:bool(not torch.equal(probes[n],params[n].detach().cpu())) for n in probes}
    assert all(changed.values()),changed
    report=dict(status='passed',step=a.stop_at,target_steps=2000,probe_parameters_changed=changed,
        optimizer_steps_min=min(float(v['step']) for v in optimizer.state.values() if 'step' in v),
        optimizer_steps_max=max(float(v['step']) for v in optimizer.state.values() if 'step' in v),
        finite_gradients=True,last_loss=losses[-1],steps_per_second=(a.stop_at-start)/(time.monotonic()-started),
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),trainable_parameters=sum(p.numel() for p in params.values()),
        checkpoint_sha256=digest(pending),checkpoint_bytes=pending.stat().st_size,
        baseline=baseline,smoke_included_in_total=True)
    assert report['optimizer_steps_min']==a.stop_at
    if a.model=='uniwm' and a.stop_at==2000:
        model.save_pretrained(a.out/'final_adapter',save_embedding_layers=True)
        processor.save_pretrained(a.out/'final_adapter')
    atomic(a.out/('smoke.json' if a.stop_at==5 else 'complete.json'),report)
    print('TRAIN_STAGE_PASSED',json.dumps(report),flush=True)


if __name__=='__main__':main()
