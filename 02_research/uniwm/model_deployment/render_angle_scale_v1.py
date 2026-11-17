"""Render only new angles at the twelve previously frozen camera centers."""
import argparse
import math
from pathlib import Path
import numpy as np
from PIL import Image
from suite_common import read,digest,atomic,frozen
from angle_scale_common import root,ANGLES,NEW_ANGLES,expanded_cases
from habitat_study_common import wrap


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--study',type=Path,required=True);a=ap.parse_args()
    out=root(a.study);old=a.study/'v2/paper_confirmation_v1'
    p=read(out/'protocol.json');assert read(out/'prepared.json')['status']=='passed'
    freeze=read(out/'execution_freeze.json')
    for path,sha in freeze['source_hashes'].items():assert digest(path)==sha
    frozen(out/'render_config.json',dict(protocol_sha256=digest(out/'protocol.json'),execution_sha256=digest(out/'execution_freeze.json')))
    import habitat_sim
    import quaternion
    from habitat_sim.utils.common import quat_from_angle_axis
    original=read(old/'references_manifest.json');camera=original['camera'];references={};checks=[]
    for scene in p['scenes']:
        for name,sha in scene['files'].items():assert digest(old/'assets'/name)==sha
        cfg=habitat_sim.SimulatorConfiguration();cfg.scene_id=str(old/'assets'/scene['glb']);cfg.enable_physics=False;cfg.gpu_device_id=0
        specs=[]
        for name,kind in [('color',habitat_sim.SensorType.COLOR),('depth',habitat_sim.SensorType.DEPTH)]:
            spec=habitat_sim.CameraSensorSpec();spec.uuid=name;spec.sensor_type=kind
            spec.resolution=[camera['height'],camera['width']];spec.hfov=camera['hfov_deg'];spec.position=[0.,0.,0.]
            spec.near=.01;spec.far=100.;specs.append(spec)
        agentcfg=habitat_sim.agent.AgentConfiguration();agentcfg.sensor_specifications=specs
        sim=habitat_sim.Simulator(habitat_sim.Configuration(cfg,[agentcfg]))
        try:
            P=np.asarray(sim._sensors['color']._sensor_object.render_camera.projection_matrix,dtype=float)
            assert np.allclose(P,camera['projection_matrix'],atol=1e-8,rtol=0)
            for sid,entry in original['references'].items():
                if entry['scene_id']!=scene['id']:continue
                v0=entry['reference_views']['0'];position=np.array(v0['camera_position_world']);pose0=entry['start_pose']
                views=dict(entry['reference_views'])
                for angle in [0]+NEW_ANGLES:
                    pose=[*pose0[:2],float(wrap(pose0[2]+math.radians(angle)))]
                    agent=sim.get_agent(0);state=habitat_sim.AgentState();state.position=position
                    state.rotation=quat_from_angle_axis(pose[2]-math.pi/2,np.array([0.,1.,0.]))
                    agent.set_state(state,reset_sensors=True);obs=sim.get_sensor_observations()
                    actual=agent.get_state().sensor_states['color'];R=quaternion.as_rotation_matrix(actual.rotation)
                    forward=R@np.array([0.,0.,-1.]);yaw=math.atan2(-forward[2],forward[0])
                    assert np.linalg.norm(np.asarray(actual.position)-position)<1e-5
                    assert abs(float(wrap(yaw-pose[2])))<1e-5
                    rgb=np.asarray(obs['color'])[...,:3].copy();depth=np.asarray(obs['depth']).copy()
                    if angle==0:
                        assert np.array_equal(rgb,np.asarray(Image.open(v0['path'])))
                        assert np.array_equal(depth,np.load(v0['depth_path'],allow_pickle=False))
                        checks.append(dict(state_id=sid,old_zero_rgb_exact=True,old_zero_depth_exact=True));continue
                    folder=out/'references'/sid;folder.mkdir(parents=True,exist_ok=True)
                    path=folder/f'yaw_{angle:+d}.png';dp=path.with_suffix('.depth.npy')
                    if path.exists():assert np.array_equal(rgb,np.asarray(Image.open(path)))
                    else:
                        tmp=path.with_suffix('.tmp');Image.fromarray(rgb).save(tmp,format='PNG');tmp.replace(path)
                    if dp.exists():assert np.array_equal(depth,np.load(dp,allow_pickle=False))
                    else:
                        with dp.with_suffix('.tmp').open('wb') as f:np.save(f,depth)
                        dp.with_suffix('.tmp').replace(dp)
                    row=dict(path=str(path),sha256=digest(path),depth_path=str(dp),depth_sha256=digest(dp),
                        yaw_deg_left_positive=angle,planar_pose=pose,camera_position_world=np.asarray(actual.position).tolist(),camera_rotation_c2w=R.tolist())
                    # Independent relative-camera check, including wrap across ±pi.
                    rel=np.asarray(v0['camera_rotation_c2w']).T@R
                    assert abs(math.atan2(rel[0,2],rel[0,0])-math.radians(angle))<1e-6
                    frozen(path.with_suffix('.json'),row);views[str(angle)]=row
                references[sid]=dict(scene_id=entry['scene_id'],start_pose=pose0,reference_views=views)
                atomic(out/'render_progress.json',dict(positions=len(references),total=12,new_rgb=len(references)*4))
                print('ANGLE_REFERENCE',sid,flush=True)
        finally:sim.close()
    assert len(references)==12 and len(checks)==12
    cases=expanded_cases(read(old/'confirmation_manifest.json')['cases'],references)
    frozen(out/'references_manifest.json',dict(camera=camera,references=references,selections=original['selections'],
        previous_manifest_sha256=digest(old/'references_manifest.json'),protocol_sha256=digest(out/'protocol.json')))
    frozen(out/'confirmation_manifest.json',dict(cases=cases,angles=ANGLES,
        input_hashes={c['start_image']:digest(c['start_image']) for c in cases},
        references_sha256=digest(out/'references_manifest.json'),protocol_sha256=digest(out/'protocol.json')))
    atomic(out/'render_complete.json',dict(status='passed',positions=12,new_rgb=48,new_depth=48,
        reused_rgb=84,zero_reproductions=checks,world_model_updates=0,
        confirmation_manifest_sha256=digest(out/'confirmation_manifest.json')))


if __name__=='__main__':main()
