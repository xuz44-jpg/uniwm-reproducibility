"""Freeze the 72 cases/model/stage and the source RGB/pose hashes."""
import argparse
import math
from pathlib import Path
import numpy as np
from suite_common import atomic,digest,frozen,read
from habitat_study_common import case_name


ANGLES=[0,3,-3,5,-5,7,-7]
MODELS=['raenwm','nwm','uniwm']


def action_text(action):
    tokens=[f'<{name}_{"pos" if v>=0 else "neg"}_bin_{math.floor(abs(v)/.01):02d}>' for name,v in zip(['dx','dy','dyaw'],action)]
    return f'Move by dx: {tokens[0]}, dy: {tokens[1]}, dyaw: {tokens[2]}',tokens


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--study',type=Path,required=True);a=ap.parse_args();r=a.study
    out=r/'v2';out.mkdir(exist_ok=True)
    refs=read(out/'test_references/manifest.json');data=read(r/'collection_v1/dataset.json')
    assert refs['status']=='passed' and refs['new_rgb_count']==6 and refs['reused_rgb_count']==8
    cases=[];inputs={}
    for i,(tid,entry) in enumerate(sorted(refs['references'].items())):
        source=entry['reference_views']['0']['path']
        for deg in ANGLES:
            for repeat in range(3):
                action=[0.,0.,math.radians(deg)];text,tokens=action_text(action)
                view=entry['reference_views'][str(deg)];assert digest(view['path'])==view['sha256']
                cases.append(dict(key=f'seven/{tid}/r{repeat}/{case_name(deg)}',experiment='seven',state_id=tid,
                    history='teacher',history_paths=[source]*4,horizon_frames=1,repeat=repeat,seed=20261023+i*100003+repeat*1009,
                    action=action,action_text=text,action_tokens=tokens,target_yaw_deg=deg,target_image=view['path'],
                    target_sha256=view['sha256'],start_image=source,goal_image=source,start_pose=entry['start_pose'],
                    query_pose=entry['start_pose'],current_index=0,memory_entries=[],memory_mode='none',
                    uniwm_bin_representative=[(1 if v>=0 else -1)*math.floor(abs(v)/.01)*.01 for v in action]))
    test=read(r/'prepared_v1/test.json');assert len(test)==30
    for i,row in enumerate(test):
        dx,dy,yaw=row['action'];translation=np.linalg.norm([dx,dy])>1e-6;turn=abs(yaw)>1e-6
        category='mixed' if translation and turn else 'translation' if translation else 'turn' if turn else 'static'
        cases.append(dict(key=f'factual/{row["sample_id"]}',experiment='factual',state_id=row['trajectory_id'],
            history='teacher',history_paths=row['history_paths'],horizon_frames=1,repeat=0,seed=20261023+200003+i,
            action=row['action'],action_text=row['action_text'],action_tokens=row['action_tokens'],category=category,
            target_image=row['target_image'],target_sha256=digest(row['target_image']),start_image=row['start_image'],goal_image=row['goal_image'],
            start_pose=row['start_pose'],query_pose=row['current_pose'],current_index=row['current_index'],memory_entries=[],memory_mode='none'))
    assert len(cases)==72 and len({c['key'] for c in cases})==72
    for c in cases:
        assert c['goal_image']==c['start_image']
        for p in c['history_paths']+[c['start_image'],c['goal_image'],c['target_image']]:inputs[p]=digest(p)
    train=read(r/'prepared_v1/train.json');train_paths={p for c in train for p in c['history_paths']+[c['target_image']]}
    assert not train_paths&set(inputs)
    manifest=dict(version='habitat_v2',cases=cases,input_hashes=inputs,angles=ANGLES,reference_manifest_sha256=digest(out/'test_references/manifest.json'),
        dataset_sha256=digest(r/'collection_v1/dataset.json'),training_rows_sha256=digest(r/'prepared_v1/train.json'),
        training_protocol_sha256=digest(r/'prepared_v1/protocol.json'),camera=data['camera'],expected_cases_per_model_stage=72,
        seven_cases=42,factual_cases=30,models=MODELS,stages=['baseline','final'],world_model_generation_budget=432,
        scores=dict(primary='LPIPS AlexNet v0.1 at 256x256, RGB bicubic; no image alignment',ssim='RGB Gaussian11 sigma1.5, valid centers',
            psnr='RGB mean squared error, data_range=1',tie_tolerance=1e-6),
        constraints=['all three baseline evaluations finish before any v2 optimizer update',
            'no future target is resolved by inference','one house, two held-out starts in one room',
            'three inference seeds are repeats, not independent scenes','nearest reference is a seven-class image match, not continuous angle accuracy'])
    frozen(out/'eval_manifest.json',manifest)
    atomic(out/'preparation.json',dict(status='passed',cases=72,seven=42,factual=30,train_test_image_overlap=0,
        source_training_images_unchanged=320,estimator_training_steps=0))
    print('EVAL_MANIFEST_PASSED',72,flush=True)


if __name__=='__main__':main()
