"""Add true rendered right-first trajectories at existing train/val starts only."""
import argparse
import math
from pathlib import Path
import numpy as np
from PIL import Image
from suite_common import atomic, digest, frozen, read
from habitat_study_common import relative_action, wrap
from uniwm_v3_common import experiment, action_tokens, action_text


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--study',type=Path,required=True);a=ap.parse_args()
    r=a.study;out=experiment(r);out.mkdir(parents=True,exist_ok=True)
    data=read(r/'collection_v1/dataset.json');camera=data['camera'];old_plan=read(r/'plan.json')
    for p,sha in old_plan['source_hashes'].items():assert digest(r/'assets'/p)==sha
    frozen(out/'render_config.json',dict(script_sha256=digest(__file__),dataset_sha256=digest(r/'collection_v1/dataset.json'),
        original_prepared_hashes={s:digest(r/'prepared_v1'/(s+'.json')) for s in ['train','validation','test']},
        splits=['train','validation'],new_positions=0,sequence_deg=[0,-3,0,-5,0,-7,0],
        new_rgb_count=108,new_depth_count=108,test_modified=False,world_model_optimizer_steps=0))
    import habitat_sim
    import quaternion
    from habitat_sim.utils.common import quat_from_angle_axis
    cfg=habitat_sim.SimulatorConfiguration();cfg.scene_dataset_config_file=str(r/'assets/mp3d.scene_dataset_config.json')
    cfg.scene_id=str(r/'assets/17DRP5sb8fy/17DRP5sb8fy.glb');cfg.enable_physics=False;cfg.gpu_device_id=0
    specs=[]
    for name,kind in [('color',habitat_sim.SensorType.COLOR),('depth',habitat_sim.SensorType.DEPTH)]:
        s=habitat_sim.CameraSensorSpec();s.uuid=name;s.sensor_type=kind;s.sensor_subtype=habitat_sim.SensorSubType.PINHOLE
        s.resolution=[camera['height'],camera['width']];s.hfov=camera['hfov_deg'];s.position=[0.,0.,0.];s.near=.01;s.far=100.;specs.append(s)
    ac=habitat_sim.agent.AgentConfiguration();ac.sensor_specifications=specs
    sim=habitat_sim.Simulator(habitat_sim.Configuration(cfg,[ac]));trajectories=[];extra={'train':[],'validation':[]};hashes={};controls=[]
    try:
        agent=sim.get_agent(0)
        P=np.asarray(sim._sensors['color']._sensor_object.render_camera.projection_matrix,dtype=float)
        assert np.allclose(P,camera['projection_matrix'],atol=1e-8,rtol=0)
        for tr in data['trajectories']:
            if tr['split']=='test':continue
            assert tr['split'] in extra and tr['room_id']!=9
            tid=tr['trajectory_id'];base=tr['frames'][0];pose0=base['planar_pose'];position=base['camera_position_world']
            folder=out/'collection'/tid;folder.mkdir(parents=True,exist_ok=True)
            start=r/'collection_v1/trajectories'/tid/'000.png'
            assert digest(start)==base['image_sha256']
            frames=[];paths=[str(start)];hashes[str(start)]=digest(start)
            for i,deg in enumerate([0,-3,0,-5,0,-7,0]):
                pose=[*pose0[:2],float(wrap(pose0[2]+math.radians(deg)))]
                state=habitat_sim.AgentState();state.position=position
                state.rotation=quat_from_angle_axis(pose[2]-math.pi/2,np.array([0.,1.,0.]))
                agent.set_state(state,reset_sensors=True);obs=sim.get_sensor_observations()
                actual=agent.get_state().sensor_states['color'];R=quaternion.as_rotation_matrix(actual.rotation)
                forward=R@np.array([0.,0.,-1.]);yaw=math.atan2(-forward[2],forward[0])
                assert np.linalg.norm(np.asarray(actual.position)-position)<1e-5
                assert abs(float(wrap(yaw-pose[2])))<1e-5
                rgb=np.asarray(obs['color'])[...,:3].copy();depth=np.asarray(obs['depth']).copy()
                if deg==0:
                    assert np.array_equal(rgb,np.asarray(Image.open(start).convert('RGB'))),(tid,i,'zero rerender mismatch')
                    controls.append(dict(trajectory_id=tid,frame=i,exact_zero_match=True))
                if i:
                    p=folder/f'{i:03d}.png';dp=folder/f'{i:03d}.depth.npy'
                    if p.exists():assert np.array_equal(np.asarray(Image.open(p)),rgb)
                    else:
                        tmp=p.with_suffix('.tmp');Image.fromarray(rgb).save(tmp,format='PNG');tmp.replace(p)
                    if dp.exists():assert np.array_equal(np.load(dp),depth)
                    else:
                        with dp.with_suffix('.tmp').open('wb') as f:np.save(f,depth)
                        dp.with_suffix('.tmp').replace(dp)
                    paths.append(str(p));hashes[str(p)]=digest(p)
                else:p=start;dp=r/'collection_v1/trajectories'/tid/'000.depth.npy'
                # Canonical exact zero prevents signed near-zero action aliases in new labels.
                action=None if not i else relative_action(frames[-1]['planar_pose'],pose)
                if action is not None:action=[0. if abs(v)<1e-10 else v for v in action]
                row=dict(frame=i,split=tr['split'],room_id=tr['room_id'],trajectory_id=tid+'_right_first',
                    source_trajectory_id=tid,planar_pose=pose,camera_position_world=np.asarray(actual.position).tolist(),
                    camera_rotation_c2w=R.tolist(),actual_body_action=action,image_path=str(p),image_sha256=digest(p),
                    depth_path=str(dp),depth_sha256=digest(dp),commanded_yaw_offset_deg=deg)
                if i:atomic(folder/f'{i:03d}.json',row)
                frames.append(row)
            for t in range(6):
                act=frames[t+1]['actual_body_action'];tokens=action_tokens(act)
                sample=dict(sample_id=f'{tid}_mirror_t{t:02d}',trajectory_id=tid+'_right_first',source_trajectory_id=tid,
                    room_id=tr['room_id'],split=tr['split'],current_index=t,target_index=t+1,horizon_frames=1,
                    history_paths=[paths[max(0,j)] for j in range(t-3,t+1)],start_image=str(start),goal_image=str(start),
                    current_image=paths[t],target_image=paths[t+1],current_pose=frames[t]['planar_pose'],target_pose=frames[t+1]['planar_pose'],
                    start_pose=pose0,action=act,action_tokens=tokens,action_text=action_text(act),translation_constrained=False,
                    source='same-position right-first Habitat render; not an image mirror')
                # Check the counterpart from the old left-first sequence explicitly.
                old_index=t+1;original=next(x for x in read(r/'prepared_v1'/(tr['split']+'.json')) if x['sample_id']==f'{tid}_t{old_index:02d}')
                assert np.allclose(act,-np.asarray(original['action']),atol=1e-9,rtol=0)
                assert abs(float(wrap(sample['current_pose'][2]-pose0[2]))+float(wrap(original['current_pose'][2]-pose0[2])))<1e-9
                extra[tr['split']].append(sample)
            trajectories.append(dict(source=tid,split=tr['split'],room_id=tr['room_id'],frames=frames))
            atomic(out/'render_progress.json',dict(completed_trajectories=len(trajectories),total=18,new_rgb=len(list((out/'collection').glob('*/*.png')))))
            print('SYMMETRIC_TRAJECTORY',tid,flush=True)
        assert len(trajectories)==18 and len(controls)==72
        assert len(list((out/'collection').glob('*/*.png')))==108
        for s,expected in [('train',336),('validation',42)]:
            old=read(r/'prepared_v1'/(s+'.json'));rows=old+extra[s];assert len(rows)==expected
            for row in rows:
                assert row['room_id']!=9
                for p in [row['start_image'],row['goal_image'],row['current_image'],row['target_image']]:hashes[p]=digest(p)
            atomic(out/(s+'.json'),rows)
        # Test manifest is referenced and hashed, never copied into fitting data.
        atomic(out/'input_image_hashes.json',hashes)
        frozen(out/'protocol.json',dict(version=3,camera=camera,train_pairs=336,validation_pairs=42,
            original_pairs_retained=True,new_right_first_pairs=108,test_manifest_sha256=digest(r/'v2/eval_manifest.json'),
            original_test_rows_sha256=digest(r/'prepared_v1/test.json'),collection=trajectories,
            zero_rerender_controls=controls,original_data_sha256=digest(r/'collection_v1/dataset.json'),
            preprocessing='native UniWMBackend.sanitize(current,False); native start/goal/current batch encoding',
            selection='42 heldout room8 validation pairs only; room9 test never selects a checkpoint',
            changes=['current preprocessing aligned to inference','right-first symmetric trajectories','validation early stopping'],
            attribution='combined repair run; cannot attribute improvement to an individual change'))
        atomic(out/'data_complete.json',dict(status='passed',new_rgb=108,new_depth=108,train_pairs=336,validation_pairs=42,
            pure_yaw_left_train=96,pure_yaw_right_train=96,from_start_each_sign_magnitude=16,
            exact_zero_rerender_controls=72,train_sha256=digest(out/'train.json'),validation_sha256=digest(out/'validation.json'),
            protocol_sha256=digest(out/'protocol.json'),input_hashes_sha256=digest(out/'input_image_hashes.json')))
    finally:sim.close()


if __name__=='__main__':main()
