"""Cache exact native inference inputs, without running image generation."""
import argparse
import copy
from pathlib import Path
import torch
from PIL import Image
from suite_common import atomic,digest,frozen,read
from suite_backends import UniWMBackend,pil
from train_habitat_v2_uniwm_r2 import save_atomic
from uniwm_v3_common import experiment,BASELINE


class Captured(Exception):
    pass


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--study',type=Path,required=True);ap.add_argument('--root',type=Path,required=True);a=ap.parse_args()
    out=experiment(a.study);data=read(out/'data_complete.json');assert data['status']=='passed'
    dest=out/'preprocessing';dest.mkdir(exist_ok=True)
    cfg=dict(script_sha256=digest(__file__),backend_sha256=digest(Path(__file__).with_name('suite_backends.py')),
        train_rows_sha256=digest(out/'train.json'),validation_rows_sha256=digest(out/'validation.json'),baseline=BASELINE,
        recipe='native FP16 backend: sanitize current through FP32 VQ, uint8 decoded PIL; batch encode start/goal/current',
        optimization_split='train only; validation cache used only in no-grad loss',new_world_model_predictions=0)
    frozen(dest/'config.json',cfg)
    if (dest/'complete.json').exists():
        done=read(dest/'complete.json')
        for s,sha in done['cache_hashes'].items():assert digest(dest/(s+'.pt'))==sha
        print('PREPROCESS_ALREADY_COMPLETE');return
    for p,sha in read(out/'input_image_hashes.json').items():assert digest(p)==sha
    torch.set_num_threads(2)
    backend=UniWMBackend(dict(checkpoint=BASELINE),a.root);model=backend.model;processor=backend.processor
    model.eval().requires_grad_(False)
    atomic(dest/'native_embedding_audit.json',backend.audit)
    label_processor=copy.deepcopy(processor);label_processor.tokenizer.padding_side='right'
    raw={};summary={};parity=[];from_start={}
    with torch.inference_mode():
        for split,expected in [('train',336),('validation',42)]:
            rows=read(out/(split+'.json'));assert len(rows)==expected
            encoded=[]
            for i,r in enumerate(rows):
                assert r['split']==split and r['room_id']!=9
                assert backend.action_text(r['action'])==r['action_text']
                capture={};original=model.generate
                def capture_generate(*args,**kwargs):
                    ids=kwargs['input_ids'].clone();pixels=kwargs['pixel_values']
                    # This is called within the unchanged native FP16 inference context.
                    codes=model.model.model.get_image_tokens(pixels).reshape(-1).long()
                    mask=ids==model.config.image_token_id
                    assert int(mask.sum())==len(codes)==3*784
                    ids[mask]=codes
                    capture.update(ids=ids.cpu(),codes=codes.cpu())
                    raise Captured()
                case=dict(r,history_paths=[r['current_image']],memory_entries=[],memory_mode='none',seed=17)
                allowed={r[k] for k in ['current_image','start_image','goal_image']}
                def resolve(path):assert path in allowed;return Path(path)
                model.generate=capture_generate
                try:
                    backend.predict(case,resolve)
                    raise AssertionError('Generation interception did not run')
                except Captured:pass
                finally:model.generate=original
                assert capture
                # Independent assembly must reproduce intercepted native prefix exactly.
                sp=r['start_pose'];pose=f'Starting Point Coordinate: x={sp[0]:.3f}, y={sp[1]:.3f}, yaw={sp[2]:.3f}\n'
                images=[pil(Path(r['start_image']),(256,256)),pil(Path(r['goal_image']),(256,256)),backend.sanitize(Path(r['current_image']),False)]
                check=processor(text=[backend.prompt(r['action_text'],pose)],images=images,padding=False,truncation=False,return_tensors='pt').to(model.device)
                other=check['input_ids'];mask=other==model.config.image_token_id
                other[mask]=model.model.model.get_image_tokens(check['pixel_values']).reshape(-1).long()
                assert torch.equal(other.cpu(),capture['ids']),(r['sample_id'],'native input mismatch')
                target=r['target_image']
                if target not in raw:
                    values=processor(text='<image>',images=pil(Path(target),(256,256)),return_tensors='pt')['pixel_values'].cuda().float()
                    raw[target]=model.model.model.get_image_tokens(values).reshape(-1).long().cpu()
                labels=label_processor(['<image>'],images=[pil(Path(target),(256,256))],padding=False,truncation=False,return_tensors='pt')['input_ids'][:,1:]
                label_mask=labels==model.config.image_token_id;assert int(label_mask.sum())==784
                labels[label_mask]=raw[target]
                inputs=capture['ids'];all_ids=torch.cat([inputs,labels],1).long()
                loss_labels=torch.cat([torch.full_like(inputs,-100),labels],1).long()
                assert all_ids.shape[1]<=4096 and int((loss_labels!=-100).sum())==labels.numel()
                ids=processor.tokenizer.convert_tokens_to_ids(r['action_tokens']);assert len(set(ids))==3
                action_mask=torch.zeros_like(inputs,dtype=torch.bool)
                for tok in ids:action_mask|=inputs==tok
                assert int(action_mask.sum())==3
                non_action=inputs.clone();non_action[action_mask]=-1
                if abs(r['action'][0])+abs(r['action'][1])<1e-9 and abs(r['action'][2])>.01 and abs(float(__import__('habitat_study_common').wrap(r['current_pose'][2]-sp[2])))<1e-9:
                    group=(split,r.get('source_trajectory_id',r['trajectory_id']))
                    if group in from_start:assert torch.equal(from_start[group],non_action)
                    else:from_start[group]=non_action
                encoded.append(dict(sample_id=r['sample_id'],input_ids=all_ids,labels=loss_labels,prefix_length=inputs.shape[1]))
                parity.append(dict(sample_id=r['sample_id'],split=split,prefix_token_mismatches=0,input_tokens=int(inputs.numel()),target_tokens=int(labels.numel())))
                atomic(dest/'progress.json',dict(stage='native_input_cache',split=split,complete=i+1,total=expected,new_generated_images=0))
                if (i+1)%24==0:print('NATIVE_CACHE',split,i+1,expected,flush=True)
            save_atomic(dest/(split+'.pt'),dict(split=split,rows_sha256=digest(out/(split+'.json')),samples=encoded))
            summary[split]=len(encoded)
    assert len(from_start)==18
    atomic(dest/'parity.json',dict(status='passed',rows=parity,native_prefix_mismatches=0,same_state_action_sibling_groups=18,
        method='intercept unchanged backend.predict before model.generate; independent prompt+codec assembly compares every prefix token',
        new_world_model_predictions=0,optimizer_updates=0,test_images_in_cache=0))
    atomic(dest/'complete.json',dict(status='passed',counts=summary,cache_hashes={s:digest(dest/(s+'.pt')) for s in summary},
        parity_sha256=digest(dest/'parity.json'),config_sha256=digest(dest/'config.json'),new_generated_images=0,optimizer_updates=0))


if __name__=='__main__':main()
