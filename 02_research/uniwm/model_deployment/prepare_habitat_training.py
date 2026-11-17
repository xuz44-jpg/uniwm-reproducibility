"""Validate actual poses and export a common, room-separated action dataset."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import pickle
import numpy as np
import yaml
from suite_common import atomic,digest,frozen,read
from habitat_study_common import relative_action,proposed_pose


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--collection',type=Path,required=True)
    ap.add_argument('--plan',type=Path,required=True);ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    d=read(a.collection/'dataset.json');plan=read(a.plan)
    assert d['plan_sha256']==digest(a.plan) and len(d['trajectories'])==20
    cfg={m:yaml.safe_load((a.root/'repos'/m/'config/data_config.yaml').read_text()) for m in ['nwm','raenwm']}
    frozen(a.out/'protocol.json',dict(plan_sha256=digest(a.plan),dataset_sha256=digest(a.collection/'dataset.json'),
        script_sha256=digest(__file__),action_config_sha256={m:digest(a.root/'repos'/m/'config/data_config.yaml') for m in cfg},
        horizon_frames=1,context_frames=4,early_context='repeat first observed frame for missing history; no future frames',
        uniwm_goal='starting frame reused as fixed goal; no actual future goal image in input',
        uniwm_input='native start/goal/current prompt with body-frame action bins of width .01',
        nwm_raenwm_input='native four-frame history and endpoint delta; meters/.25 then each native action_stats normalization',
        splits='16 training/2 validation/2 test trajectories, separated by starting room',
        optimizer_steps_per_model=2000,learning_rates=dict(nwm=1e-5,raenwm=1e-5,uniwm=1e-5),
        native_losses=dict(nwm='native diffusion.training_losses',raenwm='native Transport.training_losses',
            uniwm='native supervised-position label-smoothed image-token CE; epsilon=.1; no perceptual auxiliary loss'),
        baseline_weight_sources=dict(nwm='released ema',raenwm='released ema',uniwm='existing 1x200 adapter'),
        evaluation_weight_selection='raw final model at optimizer step 2000; no test-based checkpoint selection',
        fine_tuning='full CDiT weights for NWM/RAE; existing LoRA plus full lm_head for UniWM; frozen codecs'))
    samples=[];images={};all_actions=[];room_sets={s:set() for s in ['train','validation','test']}
    for tr in d['trajectories']:
        tid=tr['trajectory_id'];frames=tr['frames'];assert len(frames)==16
        room_sets[tr['split']].add(tr['room_id'])
        folder=a.collection/'trajectories'/tid
        for i,f in enumerate(frames):
            p=folder/f'{i:03d}.png';assert digest(p)==f['image_sha256'];images[str(p)]=f['image_sha256']
            if i:
                action=relative_action(frames[i-1]['planar_pose'],f['planar_pose'])
                assert np.allclose(action,f['actual_body_action'],atol=1e-10,rtol=0)
                assert np.allclose(proposed_pose(frames[i-1]['planar_pose'],action),f['planar_pose'],atol=1e-9,rtol=0)
                all_actions.append(action)
        # A lightweight native-style pose export; RGB PNG paths are explicit.
        native=a.out/'pose_exports'/tid;native.mkdir(parents=True,exist_ok=True)
        with (native/'traj_data.pkl').open('wb') as f:
            pickle.dump(dict(position=np.array([f['planar_pose'][:2] for f in frames]),
                yaw=np.array([f['planar_pose'][2] for f in frames]),
                camera_world_xyz=np.array([f['camera_position_world'] for f in frames]),
                camera_c2w=np.array([f['camera_rotation_c2w'] for f in frames]),
                action_body=np.array([f['actual_body_action'] for f in frames[1:]]),
                image_paths=[str(folder/f'{i:03d}.png') for i in range(16)]),f)
        for t in range(15):
            action=relative_action(frames[t]['planar_pose'],frames[t+1]['planar_pose'])
            paths=[str(folder/f'{max(0,j):03d}.png') for j in range(t-3,t+1)]
            tokens=[]
            for value,prefix in zip(action,['dx','dy','dyaw']):
                sign='pos' if value>=0 else 'neg';idx=math.floor(abs(value)/.01)
                tokens.append(f'<{prefix}_{sign}_bin_{idx:02d}>')
            text=f'Move by dx: {tokens[0]}, dy: {tokens[1]}, dyaw: {tokens[2]}'
            encoded={}
            for m,c in cfg.items():
                xy=np.array(action[:2])/float(c['recon']['metric_waypoint_spacing'])
                lo=np.array(c['action_stats']['min']);hi=np.array(c['action_stats']['max'])
                code=2*(xy-lo)/(hi-lo)-1
                # Independent reverse conversion and a 90-degree-frame check.
                recovered=(code+1)*(hi-lo)/2+lo
                assert np.allclose(recovered*.25,action[:2],atol=1e-10)
                encoded[m]=[*code.tolist(),action[2]]
            samples.append(dict(sample_id=f'{tid}_t{t:02d}',trajectory_id=tid,room_id=tr['room_id'],split=tr['split'],
                current_index=t,target_index=t+1,horizon_frames=1,history_paths=paths,
                start_image=str(folder/'000.png'),goal_image=str(folder/'000.png'),
                current_image=str(folder/f'{t:03d}.png'),target_image=str(folder/f'{t+1:03d}.png'),
                current_pose=frames[t]['planar_pose'],target_pose=frames[t+1]['planar_pose'],
                start_pose=frames[0]['planar_pose'],action=action,action_text=text,action_tokens=tokens,
                encoded_actions=encoded,translation_constrained=frames[t+1]['translation_constrained']))
    assert not room_sets['train']&room_sets['validation'] and not room_sets['train']&room_sets['test'] and not room_sets['validation']&room_sets['test']
    assert len(images)==320 and len(samples)==300
    for split,expected in [('train',240),('validation',30),('test',30)]:
        rows=[r for r in samples if r['split']==split];assert len(rows)==expected
        atomic(a.out/(split+'.json'),rows)
    atomic(a.out/'input_image_hashes.json',images)
    v=np.array(all_actions)
    atomic(a.out/'dataset_validation.json',dict(status='passed',images_verified=320,trajectories=20,
        split_samples=dict(Counter(r['split'] for r in samples)),split_rooms={k:sorted(v) for k,v in room_sets.items()},
        action_min=v.min(0).tolist(),action_max=v.max(0).tolist(),actual_action_roundtrip_pairs=300,
        translation_constrained=sum(bool(r['translation_constrained']) for r in samples),
        nonzero_translation_pairs=int((np.linalg.norm(v[:,:2],axis=1)>1e-4).sum()),
        camera=d['camera'],world_model_optimizer_steps=0,world_model_predictions=0,
        limitations=d['limitations']))
    print('DATASET_VALIDATED',len(images),len(samples),flush=True)


if __name__=='__main__':main()
