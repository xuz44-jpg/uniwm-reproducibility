"""Resumable frozen-weight inference with actual-input and reuse provenance."""
import argparse
from collections import defaultdict
import gc
from pathlib import Path
import shutil
import numpy as np
import torch
from suite_common import read, digest, atomic, frozen
from context_controls_common import root


def thash(x):
    import hashlib
    return hashlib.sha256(x.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--study',type=Path,required=True)
    ap.add_argument('--root',type=Path,required=True);ap.add_argument('--model',choices=['nwm','raenwm','uniwm'],required=True)
    ap.add_argument('--cohort',choices=['native','habitat'],required=True);ap.add_argument('--smoke',action='store_true')
    a=ap.parse_args();out=root(a.study);freeze=read(out/'execution_freeze.json')
    for path,sha in freeze['source_hashes'].items():assert digest(path)==sha,path
    assert digest(out/'protocol.json')==freeze['protocol_sha256']
    m=read(out/f'{a.cohort}_{a.model}.json');cases=m['cases'];weights=read(out/'weights.json')[a.model]
    dest=out/a.cohort/a.model;dest.mkdir(parents=True,exist_ok=True)
    frozen(dest/'config.json',dict(manifest_sha256=digest(out/f'{a.cohort}_{a.model}.json'),
        execution_sha256=digest(out/'execution_freeze.json'),weights=weights,updates=0,source='native frozen backend'))
    records={}
    for c in cases:
        path=dest/(c['key']+'.json')
        if path.exists():
            r=read(path);assert r['case']==c
            if r['status']=='ok':assert digest(path.with_suffix('.png'))==r['image_sha256']
            records[c['key']]=r
    from suite_backends import ReConBackend,UniWMBackend
    from train_habitat_model import CHECKPOINTS
    backend=None
    for arm in ['pre_habitat','adapted']:
        arm_cases=[c for c in cases if c['weight_arm']==arm]
        if a.smoke:
            # Exercise every input arm at the first position, zero and both signs.
            sid=arm_cases[0]['state_id'];arm_cases=[c for c in arm_cases if c['state_id']==sid]
        pending=[c for c in arm_cases if c['key'] not in records]
        if any(c['reuse'] is None for c in pending):
            if backend is not None:del backend;gc.collect();torch.cuda.empty_cache()
            weight=weights[arm];p=Path(weight['path']);identity=p/'adapter_model.safetensors' if a.model=='uniwm' else p
            assert digest(identity)==weight['sha256']
            if a.model=='uniwm':
                backend=UniWMBackend(dict(checkpoint=str(p)),a.root)
                assert backend.audit['input_embedding_matches_checkpoint'] and backend.audit['active_output_head_matches_checkpoint']
                from safetensors import safe_open
                checked=0
                with safe_open(str(identity),framework='pt',device='cpu') as saved:
                    for name,param in backend.model.named_parameters():
                        if '.lora_A.default.' not in name and '.lora_B.default.' not in name:continue
                        key=name.replace('.modules_to_save.default.','.')
                        for part in ['lora_A','lora_B']:key=key.replace(f'.{part}.default.',f'.{part}.')
                        assert torch.equal(param.detach().cpu(),saved.get_tensor(key).to(param.dtype));checked+=1
                assert checked==256
                del param
                backend.sampling.update(top_k=50,top_p=.9,temperature=.7)
                audit=dict(status='passed',embedding_audit=backend.audit,lora_tensors_checked=checked,weight=weight)
            else:
                backend=ReConBackend(dict(model=a.model,checkpoint=CHECKPOINTS[a.model]),a.root)
                if arm=='adapted':
                    state=torch.load(p,map_location='cpu',weights_only=False)
                    actual=dict(backend.model.named_parameters());expected={n for n,v in actual.items() if v.requires_grad}
                    assert set(state['trainable'])==expected
                    missing,unexpected=backend.model.load_state_dict(state['trainable'],strict=False)
                    assert not unexpected and not set(missing)&expected
                    for n,v in state['trainable'].items():assert torch.equal(actual[n].detach().cpu(),v)
                    audit=dict(status='passed',trainable_tensors=len(expected),weight=weight);del state
                else:audit=dict(status='passed',strict_released_ema_load=True,weight=weight)
            backend.model.eval().requires_grad_(False);torch.set_num_threads(2)
            atomic(dest/f'loaded_{arm}.json',audit)
        for c in pending:
            path=dest/(c['key']+'.png');path.parent.mkdir(parents=True,exist_ok=True)
            capture={};hook=None;input_hashes={};reuse=c['reuse'];original_generate=None
            if reuse:
                assert digest(reuse['path'])==reuse['sha256']
                if reuse.get('record_path'):assert digest(reuse['record_path'])==reuse['record_sha256']
                tmp=path.with_suffix('.tmp');shutil.copyfile(reuse['path'],tmp);tmp.replace(path)
                r=dict(status='ok',case=c,image_sha256=digest(path),origin='reused',reuse_provenance=reuse)
            else:
                allowed=set(c['history_paths']+[c['start_image'],c['goal_image']])-{None}
                def resolve(p):
                    assert p in allowed,'Unregistered image or target passed to model'
                    assert digest(p)==m['input_hashes'][p];input_hashes[p]=digest(p);return Path(p)
                if a.model!='uniwm':
                    def before(module,args,kw):
                        if not capture:capture.update(conditioning_sha256=thash(kw['x_cond']),initial_noise_sha256=thash(args[0]),
                            observed_action=kw['y'].detach().float().cpu().tolist(),observed_rel_t=kw['rel_t'].detach().float().cpu().tolist())
                    hook=backend.model.register_forward_pre_hook(before,with_kwargs=True)
                else:
                    from uniwm.extra_tokens import unwrap
                    import re
                    ids=backend.processor.tokenizer.convert_tokens_to_ids(re.findall(r'<[^>]+>',c['action_text']))
                    def embedded(module,args):
                        if capture:return
                        x=args[0].detach()
                        if int((x[0]==backend.model.config.boi_token_id).sum())!=3:return
                        assert all(int((x[0]==v).sum())==1 for v in ids)
                        clean=x.clone()
                        for v in ids:clean[clean==v]=-1
                        capture.update(post_vq_non_action_sha256=thash(clean),actual_action_ids=ids,
                            actual_post_vq_sha256=thash(x))
                        starts=(x[0]==backend.model.config.boi_token_id).nonzero().flatten().tolist()
                        capture['input_image_token_ids']={role:x[0,b+1:b+785].cpu().tolist()
                            for role,b in zip(['start','goal','current'],starts)}
                    hook=unwrap(backend.model).model.embed_tokens.register_forward_pre_hook(embedded)
                    original_generate=backend.model.generate
                    def generated(*args,**kw):
                        result=original_generate(*args,**kw)
                        row=result[0];row=row[0] if row.ndim==2 else row
                        starts=(row==backend.model.config.boi_token_id).nonzero().flatten()
                        if len(starts)==1:
                            b=int(starts[0]);codes=row[b+1:b+785].detach().cpu().tolist()
                            if len(codes)==784 and 'input_image_token_ids' in capture:
                                capture['generated_image_token_ids']=codes
                                capture['copy_tokens']={role:dict(equal=sum(a==b for a,b in zip(codes,ref)),total=784)
                                    for role,ref in capture['input_image_token_ids'].items()}
                        return result
                    backend.model.generate=generated
                try:
                    with torch.inference_mode():im,feedback,info=backend.predict(c,resolve)
                    info.pop('_anchor',None);del feedback
                    tmp=path.with_suffix('.tmp');im.save(tmp,format='PNG');tmp.replace(path)
                    if a.model!='uniwm':
                        assert np.allclose(capture['observed_action'],[backend.encode_action(c['action'])],rtol=0,atol=1e-7)
                        assert abs(capture['observed_rel_t'][0]-c['horizon_frames']/128)<1e-9
                    r=dict(status='ok',case=c,image_sha256=digest(path),origin='new',actual_input_hashes=input_hashes,**info,**capture)
                    if a.model=='uniwm':
                        assert 'post_vq_non_action_sha256' in capture
                        r['actual_sampling']=dict(backend.sampling)
                except ValueError as e:
                    if a.model!='uniwm' or str(e) not in ['Expected one generated image segment','Incomplete generated image']:raise
                    r=dict(status='parse_failed',case=c,error=str(e),origin='new',actual_input_hashes=input_hashes)
                finally:
                    if hook is not None:hook.remove()
                    if original_generate is not None:backend.model.generate=original_generate
                    original_generate=None
            atomic(path.with_suffix('.json'),r);records[c['key']]=r
            atomic(dest/'progress.json',dict(records=len(records),total=len(cases),last=c['key'],
                successful=sum(r['status']=='ok' for r in records.values()),new=sum(r['origin']=='new' for r in records.values())))
            print('CONTEXT_GENERATION',a.cohort,a.model,len(records),len(cases),c['key'],r['status'],flush=True)
        if backend is not None:del backend;backend=None;gc.collect();torch.cuda.empty_cache()
    if a.smoke:
        smoke_cases=[c for arm in ['pre_habitat','adapted'] for c in cases
                     if c['weight_arm']==arm and c['state_id']==next(x for x in cases if x['weight_arm']==arm)['state_id']]
        assert all(records[c['key']]['status']=='ok' for c in smoke_cases),'Technical smoke parse failure; retain result and stop'
        atomic(dest/'smoke.json',dict(status='passed',completed=len(records),checked_cases=len(smoke_cases),
            every_arm_successful=True,records_in_full_budget=True,updates=0));return
    assert len(records)==len(cases)
    groups=defaultdict(list)
    if a.cohort=='habitat':
        for r in records.values():
            c=r['case']
            if r['origin']=='new' and r['status']=='ok':groups[(c['weight_arm'],c['context_arm'],c['state_id'])].append(r)
        for g in groups.values():
            keys=['post_vq_non_action_sha256'] if a.model=='uniwm' else ['conditioning_sha256','initial_noise_sha256']
            for k in keys:assert len({r[k] for r in g})==1
    atomic(dest/'complete.json',dict(status='passed',records=len(records),new=sum(r['origin']=='new' for r in records.values()),
        reused=sum(r['origin']=='reused' for r in records.values()),parse_failed=sum(r['status']!='ok' for r in records.values()),
        record_hashes={k:digest(dest/(k+'.json')) for k in records},world_model_updates=0))


if __name__=='__main__':main()
