"""Render a frozen geometric sample; never inspect model predictions or train."""
import argparse
import math
from pathlib import Path
import numpy as np
from PIL import Image
from suite_common import read, digest, atomic, frozen
from confirmation_common import root, case, ANGLES
from habitat_study_common import wrap


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--study',type=Path,required=True);a=ap.parse_args()
    out=root(a.study);p=read(out/'protocol.json')
    frozen(out/'render_config.json',dict(protocol_sha256=digest(out/'protocol.json'),
        script_sha256=digest(__file__),common_sha256=digest(Path(__file__).with_name('confirmation_common.py'))))
    import habitat_sim
    import quaternion
    from habitat_sim.utils.common import quat_from_angle_axis
    camera=read(a.study/'collection_v1/camera.json')
    references={};cases=[];selection={};controls=[]

    def simulator(scene, dataset=None):
        cfg=habitat_sim.SimulatorConfiguration();cfg.scene_id=str(scene);cfg.enable_physics=False;cfg.gpu_device_id=0
        if dataset:cfg.scene_dataset_config_file=str(dataset)
        specs=[]
        for name,kind in [('color',habitat_sim.SensorType.COLOR),('depth',habitat_sim.SensorType.DEPTH)]:
            spec=habitat_sim.CameraSensorSpec();spec.uuid=name;spec.sensor_type=kind
            spec.resolution=[camera['height'],camera['width']];spec.hfov=camera['hfov_deg']
            spec.position=[0.,0.,0.];spec.near=.01;spec.far=100.;specs.append(spec)
        agent=habitat_sim.agent.AgentConfiguration();agent.sensor_specifications=specs
        return habitat_sim.Simulator(habitat_sim.Configuration(cfg,[agent]))

    def render(sim, folder, angle, pose0, position):
        pose=[*pose0[:2],float(wrap(pose0[2]+math.radians(angle)))]
        agent=sim.get_agent(0);state=habitat_sim.AgentState();state.position=position
        state.rotation=quat_from_angle_axis(pose[2]-math.pi/2,np.array([0.,1.,0.]))
        agent.set_state(state,reset_sensors=True);obs=sim.get_sensor_observations()
        actual=agent.get_state().sensor_states['color'];R=quaternion.as_rotation_matrix(actual.rotation)
        forward=R@np.array([0.,0.,-1.]);yaw=math.atan2(-forward[2],forward[0])
        assert np.linalg.norm(np.asarray(actual.position)-position)<1e-5
        assert abs(float(wrap(yaw-pose[2])))<1e-5
        rgb=np.asarray(obs['color'])[...,:3].copy();depth=np.asarray(obs['depth']).copy()
        folder.mkdir(parents=True,exist_ok=True);path=folder/f'yaw_{angle:+d}.png';dp=path.with_suffix('.depth.npy')
        if path.exists():assert np.array_equal(np.asarray(Image.open(path)),rgb)
        else:
            temp=path.with_suffix('.tmp');Image.fromarray(rgb).save(temp,format='PNG');temp.replace(path)
        if dp.exists():assert np.array_equal(np.load(dp,allow_pickle=False),depth)
        else:
            with dp.with_suffix('.tmp').open('wb') as f:np.save(f,depth)
            dp.with_suffix('.tmp').replace(dp)
        row=dict(path=str(path),sha256=digest(path),depth_path=str(dp),depth_sha256=digest(dp),
            yaw_deg_left_positive=angle,planar_pose=pose,camera_position_world=np.asarray(actual.position).tolist(),
            camera_rotation_c2w=R.tolist())
        frozen(path.with_suffix('.json'),row);return row

    for index,scene in enumerate(p['confirmation']['scenes']):
        for filename,sha in scene['files'].items():assert digest(out/'assets'/filename)==sha
        sim=simulator(out/'assets'/scene['glb'])
        try:
            assert sim.pathfinder.is_loaded or sim.pathfinder.load_nav_mesh(str(out/'assets'/scene['navmesh']))
            sim.seed(p['confirmation']['selection_seed']+index)
            P=np.asarray(sim._sensors['color']._sensor_object.render_camera.projection_matrix,dtype=float)
            assert np.allclose(P,camera['projection_matrix'],atol=1e-8,rtol=0)
            sp=out/'selection'/f'{scene["id"]}.json'
            if sp.exists():selected=read(sp)
            else:
                chosen=[];attempts=0;rng=np.random.default_rng(p['confirmation']['selection_seed']+index)
                while len(chosen)<p['confirmation']['positions_per_scene']:
                    attempts+=1;assert attempts<=20000,'Not enough geometrically separated navigable points'
                    point=np.asarray(sim.pathfinder.get_random_navigable_point(),dtype=float)
                    if not np.isfinite(point).all():continue
                    if any(np.linalg.norm(point-np.asarray(x['floor_world_xyz']))<1.25 for x in chosen):continue
                    yaw=float(rng.uniform(-math.pi,math.pi));position=point.copy();position[1]+=1.5
                    chosen.append(dict(state_id=f'{scene["id"]}_p{len(chosen):02d}',floor_world_xyz=point.tolist(),
                        position_world=position.tolist(),pose=[float(point[0]),float(-point[2]),yaw]))
                selected=dict(scene=scene['id'],seed=p['confirmation']['selection_seed']+index,
                    attempts=attempts,points=chosen,image_or_model_selection=False)
                frozen(sp,selected)
            selection[scene['id']]=selected
            for entry in selected['points']:
                sid=entry['state_id'];views={str(d):render(sim,out/'references'/sid,d,entry['pose'],entry['position_world']) for d in ANGLES}
                repeated=render(sim,out/'references'/sid,0,entry['pose'],entry['position_world'])
                assert repeated['sha256']==views['0']['sha256']
                references[sid]=dict(scene_id=scene['id'],start_pose=entry['pose'],reference_views=views)
                for rep in range(3):cases.extend(case(sid,views,entry['pose'],rep,scene['id']))
                atomic(out/'render_progress.json',dict(confirmation_positions=len(references),total=12))
                print('REFERENCE_POSITION',sid,flush=True)
        finally:sim.close()
    frozen(out/'references_manifest.json',dict(camera=camera,references=references,selections=selection,
        protocol_sha256=digest(out/'protocol.json'),model_predictions_seen=False))
    frozen(out/'confirmation_manifest.json',dict(cases=cases,angles=ANGLES,
        input_hashes={c['start_image']:digest(c['start_image']) for c in cases},
        references_sha256=digest(out/'references_manifest.json'),protocol_sha256=digest(out/'protocol.json')))

    # The mechanism experiment is explicitly exploratory on the old two positions.
    old=read(a.study/'v2/test_references/manifest.json');goals={}
    sim=simulator(a.study/'assets/17DRP5sb8fy/17DRP5sb8fy.glb',a.study/'assets/mp3d.scene_dataset_config.json')
    try:
        for sid,entry in old['references'].items():
            v=entry['reference_views']['0'];folder=out/'mechanism_goals'/sid
            zero=render(sim,folder,0,entry['start_pose'],v['camera_position_world'])
            assert np.array_equal(np.asarray(Image.open(zero['path'])),np.asarray(Image.open(v['path'])))
            goals[sid]={'zero':v,
                'left30':render(sim,folder,30,entry['start_pose'],v['camera_position_world']),
                'right30':render(sim,folder,-30,entry['start_pose'],v['camera_position_world'])}
            controls.append(dict(state_id=sid,old_zero_exactly_reproduced=True))
    finally:sim.close()
    frozen(out/'mechanism_goals.json',dict(goals=goals,controls=controls,action_specific_future_in_inputs=False))
    atomic(out/'render_complete.json',dict(status='passed',confirmation_positions=12,confirmation_rgb=84,
        confirmation_depth=84,mechanism_goal_rgb=6,zero_repeat_exact=True,world_model_updates=0,
        confirmation_manifest_sha256=digest(out/'confirmation_manifest.json'),
        references_manifest_sha256=digest(out/'references_manifest.json'),goals_sha256=digest(out/'mechanism_goals.json')))


if __name__=='__main__':main()
