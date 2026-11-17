"""Fresh 1x200 fine-tuning: native input cache, symmetric data, validation stop."""
import argparse
import gc
import json
import math
from pathlib import Path
import random
import shutil
import sys
import time
from types import SimpleNamespace
import numpy as np
import torch
from suite_common import atomic,digest,frozen,read
from train_habitat_v2_uniwm_r2 import cpu_tree,save_atomic
from uniwm_v3_common import experiment,BASELINE,SEED,stopping_update


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--study',type=Path,required=True);ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--phase',choices=['smoke','train'],required=True);a=ap.parse_args()
    run=experiment(a.study);out=run/'training';out.mkdir(exist_ok=True)
    prepared=run/'preprocessing';prep=read(prepared/'complete.json');assert prep['status']=='passed'
    assert read(prepared/'parity.json')['native_prefix_mismatches']==0
    if a.phase=='train':assert read(out/'smoke.json')['status']=='passed'
    cfg=dict(script_sha256=digest(__file__),common_sha256=digest(Path(__file__).with_name('uniwm_v3_common.py')),
        baseline=BASELINE,baseline_hashes={n:digest(Path(BASELINE)/n) for n in ['adapter_model.safetensors','adapter_config.json','tokenizer.json']},
        training_cache_sha256=prep['cache_hashes']['train'],validation_cache_sha256=prep['cache_hashes']['validation'],
        protocol_sha256=digest(run/'protocol.json'),backend_sha256=digest(Path(__file__).with_name('suite_backends.py')),
        master_restore_sha256=digest(Path(__file__).with_name('restore_uniwm_trainable_masters.py')),
        seed=SEED,batch_size=1,learning_rate=1e-5,warmup_steps=100,schedule='100-step linear warmup then constant; no forced final step',
        optimizer='AdamW betas .9/.999 weight_decay=0 foreach=False',gradient_clip=1.,label_smoothing=.1,
        precision='BF16 autocast; FP32 trainable masters; frozen base and codec; exact native inference input tokens',
        validation_interval=100,min_steps_before_plateau_stop=300,relative_significant_improvement=.005,patience=3,
        max_steps_budget=2000,best_selection='minimum mean validation label-smoothed CE across all 42 rows, including step0',
        loss_threshold='no absolute CE target; lower validation loss and plateau define convergence',
        checkpoint_policy='resume weights+optimizer+RNG every100 and smoke5; separate best trainable weights; atomic files',
        data=dict(train=336,validation=42,test_used_for_selection=0),initialization='fresh original 1x200, never old habitat step2000')
    frozen(out/'training_config.json',cfg)
    if (out/'complete.json').exists():
        done=read(out/'complete.json');assert digest(out/'final_adapter/adapter_model.safetensors')==done['adapter_sha256']
        print('ALREADY_COMPLETE',done['step']);return
    for split in ['train','validation']:assert digest(prepared/(split+'.pt'))==prep['cache_hashes'][split]
    train_cache=torch.load(prepared/'train.pt',map_location='cpu',weights_only=False)
    val_cache=torch.load(prepared/'validation.pt',map_location='cpu',weights_only=False)
    train_rows=read(run/'train.json');val_rows=read(run/'validation.json')
    for cache,rows,split in [(train_cache,train_rows,'train'),(val_cache,val_rows,'validation')]:
        assert cache['split']==split and cache['rows_sha256']==digest(run/(split+'.json'))
        assert [x['sample_id'] for x in cache['samples']]==[r['sample_id'] for r in rows]
        assert all(r['split']==split and r['room_id']!=9 for r in rows)
    assert not set(r['sample_id'] for r in train_rows)&set(r['sample_id'] for r in val_rows)
    random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    sys.path.insert(0,str(a.root/'repos/uniwm'))
    from scripts.load_model import load_model
    from restore_uniwm_trainable_masters import restore_masters
    from verify_uniwm_embeddings import compare_matrix
    from safetensors import safe_open
    from uniwm.extra_tokens import unwrap
    args=SimpleNamespace(model='anole',model_ckpt=None,init_adapter=BASELINE,image_seq_length=784,precision='bf16',
        attn_impl='sdpa',base_model='external/uniwm-action-pilot/models/anole',use_memory_bank_inference=False,
        do_task_level_eval=False,do_train=True,do_single_step_eval=False,do_rollout_eval=False,do_action_probe=False,
        lora_r=16,lora_alpha=32,lora_dropout=.05,lora_targets='q_proj,k_proj,v_proj,o_proj',lm_head_mode='full',
        fp32_mlp_layers='auto',lora_fp32=True)
    mp=load_model(args,None);model=mp['model'];processor=mp['processor'];hf=unwrap(model);hf.model.vqmodel.eval()
    restore=restore_masters(model,BASELINE);atomic(out/'fp32_master_restore_audit.json',restore)
    with safe_open(str(Path(BASELINE)/'adapter_model.safetensors'),framework='pt',device='cpu') as saved:
        emb=compare_matrix(hf.model.embed_tokens.weight,saved.get_slice('base_model.model.model.embed_tokens.weight'))
        head=hf.lm_head.modules_to_save[hf.lm_head.active_adapter].weight
        head_check=compare_matrix(head,saved.get_slice('base_model.model.lm_head.weight'))
    assert emb['matched'] and head_check['matched']
    atomic(out/'baseline_embedding_audit.json',dict(status='passed',input_table=emb,output_head=head_check,source=BASELINE))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    model.enable_input_require_grads();model.config.use_cache=False
    params={n:p for n,p in model.named_parameters() if p.requires_grad}
    for p in params.values():p.data=p.data.float()
    assert len(params)==257
    probe_names=[n for n in params if 'lora_B' in n][:3]
    probes={n:params[n].detach().cpu().clone() for n in probe_names}
    optimizer=torch.optim.AdamW(list(params.values()),lr=1e-5,betas=(.9,.999),weight_decay=0.,foreach=False)
    pending=out/'latest.pt';best_path=out/'best_trainable.pt';start=0;selection={};stopped=False
    if pending.exists():
        resume=torch.load(pending,map_location='cpu',weights_only=False)
        assert resume['training_config_sha256']==digest(out/'training_config.json') and set(resume['trainable'])==set(params)
        with torch.no_grad():
            for n,p in params.items():p.copy_(resume['trainable'][n])
        optimizer.load_state_dict(resume['optimizer']);start=resume['step'];selection=resume['selection'];stopped=resume['stopped']
        random.setstate(resume['rng_python']);np.random.set_state(resume['rng_numpy']);torch.set_rng_state(resume['rng_cpu']);torch.cuda.set_rng_state_all(resume['rng_cuda'])
        del resume;gc.collect()
    # Atomic optimizer saves require ~7GB transiently; old experiments are never removed.
    assert shutil.disk_usage(out).free>8*1024**3,('insufficient_root_disk',shutil.disk_usage(out).free)
    processor.save_pretrained(out/'processor');model.train();hf.model.vqmodel.eval()
    started=time.monotonic();last_step=start

    def loss_for(sample):
        input_ids=sample['input_ids'].cuda();labels=sample['labels'].cuda()
        shifted=labels[:,1:];keep=shifted!=-100;positions=torch.cat([keep,torch.zeros_like(keep[:,:1])],1)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            output=model(input_ids=input_ids,attention_mask=torch.ones_like(input_ids),loss_positions=positions,use_cache=False)
            logits=output.logits.float();targets=shifted[keep]
            log_probs=torch.nn.functional.log_softmax(logits,dim=-1)
            nll=-log_probs.gather(-1,targets[:,None]).squeeze(-1)
            loss=(.9*nll-.1*log_probs.mean(-1)).mean()
        assert torch.isfinite(loss).item()
        return loss,nll.detach().mean()

    def validate(step):
        nonlocal selection
        model.eval();totals=[];nlls=[];parts={'left':[],'right':[],'zero':[]}
        with torch.no_grad():
            for sample,row in zip(val_cache['samples'],val_rows):
                loss,nll=loss_for(sample);value=float(loss);totals.append(value);nlls.append(float(nll))
                parts['left' if row['action'][2]>.01 else 'right' if row['action'][2]<-.01 else 'zero'].append(value)
        value=float(np.mean(totals));previous_best=selection.get('best_loss',float('inf'))
        selection=stopping_update(selection,value,step)
        if value<previous_best:
            save_atomic(best_path,dict(step=step,validation_loss=value,trainable={n:p.detach().cpu() for n,p in params.items()},
                training_config_sha256=digest(out/'training_config.json')))
        report=dict(step=step,validation_loss=value,validation_nll=float(np.mean(nlls)),count=len(totals),
            by_direction={k:dict(count=len(v),loss=float(np.mean(v)) if v else None) for k,v in parts.items()},**selection)
        with (out/'validation_log.jsonl').open('a') as f:f.write(json.dumps(report)+'\n')
        atomic(out/'validation_latest.json',report);print('VALIDATION',json.dumps(report),flush=True)
        model.train();hf.model.vqmodel.eval()

    def checkpoint(step,is_stopped=False):
        state=dict(step=step,model_name='uniwm',trainable={n:p.detach().cpu() for n,p in params.items()},optimizer=cpu_tree(optimizer.state_dict()),
            rng_python=random.getstate(),rng_numpy=np.random.get_state(),rng_cpu=torch.get_rng_state(),rng_cuda=torch.cuda.get_rng_state_all(),
            training_config_sha256=digest(out/'training_config.json'),selection=selection,stopped=is_stopped)
        save_atomic(pending,state)
        atomic(out/'checkpoint.json',dict(step=step,bytes=pending.stat().st_size,sha256=digest(pending),selection=selection,stopped=is_stopped))

    if not selection:
        validate(0)
        atomic(out/'progress.json',dict(stage='baseline_validation_complete',optimizer_step=0,selection=selection))
    limit=5 if a.phase=='smoke' else 2000
    if not stopped:
        for step0 in range(start,limit):
            epoch,index=divmod(step0,len(train_rows));perm=np.random.default_rng(SEED+epoch).permutation(len(train_rows));idx=int(perm[index])
            step=step0+1;optimizer.zero_grad(set_to_none=True)
            for group in optimizer.param_groups:group['lr']=1e-5*min(1.,step/100.)
            loss,nll=loss_for(train_cache['samples'][idx]);loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(list(params.values()),1.,error_if_nonfinite=True);optimizer.step();last_step=step
            rec=dict(step=step,loss=float(loss.detach()),nll=float(nll),gradient_norm=float(norm),lr=optimizer.param_groups[0]['lr'],
                sample_id=train_rows[idx]['sample_id'],elapsed_seconds=time.monotonic()-started,peak_allocated_bytes=torch.cuda.max_memory_allocated())
            with (out/'train_log.jsonl').open('a') as f:f.write(json.dumps(rec)+'\n')
            atomic(out/'progress.json',dict(stage='training',optimizer_step=step,max_steps_budget=2000,last=rec,selection=selection))
            print('TRAIN',json.dumps(rec),flush=True)
            if step%100==0:
                validate(step);stopped=selection['stop']
            if step%100==0 or step==limit:checkpoint(step,stopped)
            if stopped:break
    changed={n:not torch.equal(probes[n],params[n].detach().cpu()) for n in probes};assert all(changed.values())
    optimizer_steps=[int(v['step']) for v in optimizer.state.values() if 'step' in v]
    assert min(optimizer_steps)==max(optimizer_steps)==last_step
    if a.phase=='smoke':
        assert last_step==5
        atomic(out/'smoke.json',dict(status='passed',step=5,finite_gradients=True,baseline_validation_count=42,
            parameters_changed=changed,optimizer_steps=5,checkpoint_sha256=digest(pending),initial_validation=selection))
        return
    assert stopped or last_step==2000
    # Export the best validation checkpoint, not automatically the last update.
    best=torch.load(best_path,map_location='cpu',weights_only=False)
    assert best['training_config_sha256']==digest(out/'training_config.json') and best['step']==selection['best_step']
    with torch.no_grad():
        for n,p in params.items():p.copy_(best['trainable'][n])
    del best;gc.collect()
    model.save_pretrained(out/'final_adapter',save_embedding_layers=True);processor.save_pretrained(out/'final_adapter')
    report=dict(status='passed',step=selection['best_step'],optimizer_steps_completed=last_step,selection=selection,
        stop_reason='validation_plateau' if stopped else 'budget_cap_not_convergence',max_steps_budget=2000,
        optimizer_steps_min=min(optimizer_steps),optimizer_steps_max=max(optimizer_steps),parameters_changed=changed,
        baseline=BASELINE,adapter_sha256=digest(out/'final_adapter/adapter_model.safetensors'),
        best_trainable_sha256=digest(best_path),latest_resume_checkpoint_sha256=digest(pending),
        finite_gradients=True,test_used_for_selection=False,train_rows=336,validation_rows=42,
        limitation='combined preprocessing+data+stopping change; cannot isolate causal contribution; one house')
    atomic(out/'complete.json',report);atomic(out/'progress.json',dict(stage='complete',**report))
    print('UNIWM_TRAIN_COMPLETE',json.dumps(report),flush=True)


if __name__=='__main__':main()
