"""Speech-guard owned symmetric adaptation with fixed-noise validation stopping."""
import argparse
import gc
import math
from pathlib import Path
import random
import shutil
import sys
import time
from collections import defaultdict
import numpy as np
from PIL import Image
import torch
from suite_common import atomic,digest,frozen,read
from suite_backends import ReConBackend,seed_all
from train_habitat_model import CHECKPOINTS,cpu_tree,save_atomic
from uniwm_v3_common import stopping_update
from recon_symmetric_v3_common import experiment,SEED


def rng_state():return dict(python=random.getstate(),numpy=np.random.get_state(),cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all())
def restore_rng(s):
    random.setstate(s['python']);np.random.set_state(s['numpy']);torch.set_rng_state(s['cpu']);torch.cuda.set_rng_state_all(s['cuda'])


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--study',type=Path,required=True);ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--model',choices=['nwm','raenwm'],required=True);ap.add_argument('--phase',choices=['smoke','train'],required=True);a=ap.parse_args()
    run=experiment(a.study);out=run/'training'/a.model;out.mkdir(parents=True,exist_ok=True)
    data=read(run/'data_complete.json');assert data['status']=='passed'
    rows={s:read(run/(s+'.json')) for s in ['train','validation']};assert len(rows['train'])==336 and len(rows['validation'])==42
    assert all(x['room_id']!=9 and x['split']==s for s,items in rows.items() for x in items)
    if a.phase=='train':
        for m in ['raenwm','nwm']:assert read(run/'training'/m/'smoke.json')['status']=='passed'
    baseline=CHECKPOINTS[a.model]
    config=dict(model=a.model,script_sha256=digest(__file__),protocol_sha256=digest(run/'protocol.json'),backend_sha256=digest(Path(__file__).with_name('suite_backends.py')),
        baseline=baseline,baseline_sha256=digest(baseline),data_rows_sha256={s:digest(run/(s+'.json')) for s in rows},
        seed=SEED,batch_size=1,lr=1e-5,warmup100=True,after_warmup='constant',max_steps_budget=2000,
        optimizer='AdamW betas .9/.999 weight_decay0 foreachFalse',clip_norm=1.,precision='BF16 autocast and FP32 trainable weights',
        native_loss='diffusion.training_losses' if a.model=='nwm' else 'Transport.training_losses',
        validation_rows=42,validation_noise_seeds_per_row=2,validation_seed_base=87261026,validation_interval=100,validation_rng_restore=True,
        stopping=dict(min_steps=300,relative_delta=.005,patience=3),selection='minimum actual fixed-noise validation loss, including baseline0; no test selection',
        checkpoint='atomic latest weights+optimizer+RNG at smoke5/every100, separate best model',ema='released EMA initialization, evaluate raw validation-best model')
    frozen(out/'training_config.json',config)
    if (out/'complete.json').exists():
        done=read(out/'complete.json');assert digest(out/'best_trainable.pt')==done['checkpoint_sha256'];print('ALREADY_COMPLETE',a.model,done['step']);return
    assert shutil.disk_usage(out).free>24*1024**3,'root disk space below checkpoint margin'
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;seed_all(SEED)
    backend=ReConBackend(dict(model=a.model,checkpoint=baseline),a.root);model=backend.model;codec=backend.codec
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;codec.eval().requires_grad_(False)
    model.gradient_checkpointing=(a.model=='nwm')
    for items in rows.values():
        for row in items:assert np.allclose(backend.encode_action(row['action']),row['encoded_actions'][a.model],atol=1e-10,rtol=0)
    paths=sorted({p for items in rows.values() for row in items for p in row['history_paths']+[row['target_image']]})
    hashes=read(run/'input_image_hashes.json')
    for p in paths:assert digest(p)==hashes[p]
    cache_path=out/'latent_cache.pt';cache_meta=out/'latent_cache_complete.json'
    if cache_meta.exists():
        assert digest(cache_path)==read(cache_meta)['sha256'];cache=torch.load(cache_path,map_location='cpu',weights_only=False);assert sorted(cache)==paths
    else:
        cache={}
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            for index,p in enumerate(paths):
                with Image.open(p) as im:x=backend.misc.transform(im.convert('RGB'))[None].cuda()
                if a.model=='nwm':
                    dist=codec.encode(x).latent_dist;cache[p]=(dist.mean.cpu(),dist.std.cpu())
                else:cache[p]=codec.encode(x*.5+.5).cpu()
                if index%32==0:atomic(out/'progress.json',dict(stage='encode_train_validation',complete=index+1,total=len(paths),optimizer_step=0))
        save_atomic(cache_path,cache);atomic(cache_meta,dict(status='passed',images=len(paths),sha256=digest(cache_path),recipe='NWM posterior moments, sampled per call; RAE deterministic latent; frozen native codec'))
    if a.model=='nwm':
        from diffusion import create_diffusion
        diffusion=create_diffusion('')
    else:
        sys.path.insert(0,str(a.root/'repos/raenwm/RAE/src'))
        from RAE.src.stage2.transport.transport import Transport,ModelType,PathType,WeightType
        transport=Transport(model_type=ModelType.VELOCITY,path_type=PathType.LINEAR,loss_type=WeightType.VELOCITY,
            time_dist_type='uniform',time_dist_shift=math.sqrt(codec.latent_dim*16*16/4096),train_eps=1e-3,sample_eps=1e-3)
    params={n:p for n,p in model.named_parameters() if p.requires_grad};assert params
    for p in params.values():p.data=p.data.float()
    probe_names=list(params)[:3];probes={n:params[n].detach().cpu().clone() for n in probe_names}
    optimizer=torch.optim.AdamW(list(params.values()),lr=1e-5,betas=(.9,.999),weight_decay=0.,foreach=False)
    selection={};start=0;stopped=False;latest=out/'latest.pt';best=out/'best_trainable.pt';seed_all(SEED)
    if latest.exists():
        state=torch.load(latest,map_location='cpu',weights_only=False)
        assert state['training_config_sha256']==digest(out/'training_config.json') and set(state['trainable'])==set(params)
        with torch.no_grad():
            for n,p in params.items():p.copy_(state['trainable'][n])
        optimizer.load_state_dict(state['optimizer']);start=state['step'];selection=state['selection'];stopped=state['stopped'];restore_rng(state['rng']);del state;gc.collect()
    if a.phase=='smoke' and start>=5 and (out/'smoke.json').exists():print('SMOKE_ALREADY_COMPLETE',a.model);return

    def loss_for(row):
        latents=[]
        for p in row['history_paths']+[row['target_image']]:
            if a.model=='nwm':
                mean,std=cache[p];mean,std=mean.cuda(),std.cuda();z=(mean+std*torch.randn_like(mean))*.18215
            else:z=cache[p].cuda()
            latents.append(z)
        cond=torch.stack(latents[:4],dim=1);target=latents[4];y=torch.tensor(row['encoded_actions'][a.model],device='cuda',dtype=torch.float32)[None]
        kwargs=dict(y=y,x_cond=cond,rel_t=torch.tensor([1/128],device='cuda',dtype=torch.float32))
        with torch.autocast('cuda',dtype=torch.bfloat16):
            if a.model=='nwm':
                t=torch.randint(0,diffusion.num_timesteps,(1,),device='cuda');loss=diffusion.training_losses(model,target,t,kwargs)['loss'].mean()
            else:loss=transport.training_losses(model,target,kwargs)['loss'].mean()
        assert torch.isfinite(loss).item();return loss

    def validate(step):
        nonlocal selection
        saved=rng_state();model.eval();values=[];groups=defaultdict(list);individual=[];repeat_error=None
        try:
            with torch.no_grad():
                for index,row in enumerate(rows['validation']):
                    direction='left' if row['action'][2]>.01 else 'right' if row['action'][2]<-.01 else 'zero'
                    for repeat in range(2):
                        seed=87261026+index*101+repeat;seed_all(seed);value=float(loss_for(row));values.append(value);groups[direction].append(value)
                        individual.append(dict(sample_id=row['sample_id'],repeat=repeat,seed=seed,loss=value))
                if step==0:
                    seed_all(87261026);repeat_error=abs(float(loss_for(rows['validation'][0]))-values[0])
                    assert repeat_error<=1e-6*(1+abs(values[0])),('fixed_noise_validation_not_reproducible',repeat_error)
        finally:restore_rng(saved);model.train();codec.eval()
        after=rng_state();assert torch.equal(saved['cpu'],after['cpu']) and all(torch.equal(x,y) for x,y in zip(saved['cuda'],after['cuda']))
        value=float(np.mean(values));prior=selection.get('best_loss',float('inf'));selection=stopping_update(selection,value,step)
        if value<prior:save_atomic(best,dict(step=step,model_name=a.model,validation_loss=value,trainable={n:p.detach().cpu() for n,p in params.items()},training_config_sha256=digest(out/'training_config.json')))
        record=dict(step=step,validation_loss=value,rows=42,noise_repeats=2,forward_calls=84,extra_reproducibility_forward=1 if step==0 else 0,
            repeated_first_loss_abs_error=repeat_error,training_rng_restored=True,by_direction={k:dict(count=len(v),loss=float(np.mean(v))) for k,v in groups.items()},**selection)
        atomic(out/'validation'/f'{step:04d}.json',dict(summary=record,rows=individual))
        with (out/'validation_log.jsonl').open('a') as f:f.write(__import__('json').dumps(record)+'\n')
        atomic(out/'validation_latest.json',record);print('VALIDATION',a.model,record,flush=True)

    def checkpoint(step):
        save_atomic(latest,dict(step=step,model_name=a.model,trainable={n:p.detach().cpu() for n,p in params.items()},optimizer=cpu_tree(optimizer.state_dict()),
            rng=rng_state(),selection=selection,stopped=stopped,training_config_sha256=digest(out/'training_config.json')))
        atomic(out/'checkpoint.json',dict(step=step,bytes=latest.stat().st_size,sha256=digest(latest),selection=selection,stopped=stopped))
    if not selection:validate(0)
    last=start;limit=5 if a.phase=='smoke' else 2000;started=time.monotonic();model.train();codec.eval()
    if not stopped:
        for step0 in range(start,limit):
            epoch,index=divmod(step0,len(rows['train']));idx=int(np.random.default_rng(SEED+epoch).permutation(len(rows['train']))[index]);row=rows['train'][idx];step=step0+1
            optimizer.zero_grad(set_to_none=True)
            for g in optimizer.param_groups:g['lr']=1e-5*min(1.,step/100.)
            loss=loss_for(row);loss.backward();norm=torch.nn.utils.clip_grad_norm_(list(params.values()),1.,error_if_nonfinite=True);optimizer.step();last=step
            record=dict(step=step,loss=float(loss.detach()),gradient_norm=float(norm),sample_id=row['sample_id'],lr=optimizer.param_groups[0]['lr'],elapsed_seconds=time.monotonic()-started,peak_allocated_bytes=torch.cuda.max_memory_allocated())
            with (out/'train_log.jsonl').open('a') as f:f.write(__import__('json').dumps(record)+'\n')
            atomic(out/'progress.json',dict(stage='training',optimizer_step=step,max_steps_budget=2000,last=record,selection=selection));print('TRAIN',a.model,record,flush=True)
            if step%100==0:validate(step);stopped=selection['stop']
            if step%100==0 or step==limit:checkpoint(step)
            if stopped:break
    changed={n:not torch.equal(p,params[n].detach().cpu()) for n,p in probes.items()};assert all(changed.values()),changed
    steps=[int(s['step']) for s in optimizer.state.values() if 'step' in s];assert min(steps)==max(steps)==last
    if a.phase=='smoke':
        atomic(out/'smoke.json',dict(status='passed',optimizer_steps_completed=5,probe_parameters_changed=changed,finite_gradients=True,checkpoint_sha256=digest(latest),initial_validation=selection));print('SMOKE_PASSED',a.model,flush=True);return
    assert stopped or last==2000
    chosen=torch.load(best,map_location='cpu',weights_only=False);assert chosen['step']==selection['best_step'];del chosen;gc.collect()
    report=dict(status='passed',model=a.model,step=selection['best_step'],optimizer_steps_completed=last,stop_reason='validation_plateau' if stopped else 'budget_cap_not_convergence',
        selection=selection,checkpoint_sha256=digest(best),latest_resume_sha256=digest(latest),initialization=baseline,train_pairs=336,validation_pairs=42,test_used_for_selection=False,
        optimizer_steps_min=min(steps),optimizer_steps_max=max(steps),probe_parameters_changed=changed,finite_gradients=True)
    atomic(out/'complete.json',report);atomic(out/'progress.json',dict(stage='complete',**report));print('SYMMETRIC_TRAIN_COMPLETE',report,flush=True)


if __name__=='__main__':main()
