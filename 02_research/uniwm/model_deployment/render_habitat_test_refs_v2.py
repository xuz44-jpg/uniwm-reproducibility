"""Six missing held-out right views, with pose and exact rerender controls."""
import argparse
import math
from pathlib import Path
import numpy as np
from PIL import Image
from suite_common import atomic,digest,frozen,read
from habitat_study_common import case_name,wrap


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--study',type=Path,required=True);a=ap.parse_args()
    r=a.study;out=r/'v2/test_references';out.mkdir(parents=True,exist_ok=True)
    data=read(r/'collection_v1/dataset.json');camera=data['camera'];old_plan=read(r/'plan.json')
    assert len(data['trajectories'])==20 and data['trajectory_rgb_count']==320
    for name,sha in old_plan['source_hashes'].items():assert digest(r/'assets'/name)==sha
    frozen(out/'config.json',dict(script_sha256=digest(__file__),dataset_sha256=digest(r/'collection_v1/dataset.json'),
        camera=camera,heldout_trajectories=['room09_traj00','room09_traj01'],new_degrees=[-3,-5,-7],
        reused_degrees_to_frame={'0':0,'3':2,'5':4,'7':6},new_rgb_budget=6,
        original_dataset_untouched=True))
    import habitat_sim
    import quaternion
    from habitat_sim.utils.common import quat_from_angle_axis
    cfg=habitat_sim.SimulatorConfiguration();cfg.scene_dataset_config_file=str(r/'assets/mp3d.scene_dataset_config.json')
    cfg.scene_id=str(r/'assets/17DRP5sb8fy/17DRP5sb8fy.glb');cfg.enable_physics=False;cfg.gpu_device_id=0
    specs=[]
    for name,kind in [('color',habitat_sim.SensorType.COLOR),('depth',habitat_sim.SensorType.DEPTH)]:
        spec=habitat_sim.CameraSensorSpec();spec.uuid=name;spec.sensor_type=kind
        spec.sensor_subtype=habitat_sim.SensorSubType.PINHOLE;spec.resolution=[camera['height'],camera['width']]
        spec.hfov=camera['hfov_deg'];spec.position=[0.,0.,0.];spec.near=.01;spec.far=100.;specs.append(spec)
    ac=habitat_sim.agent.AgentConfiguration();ac.sensor_specifications=specs
    sim=habitat_sim.Simulator(habitat_sim.Configuration(cfg,[ac]));references={};controls=[]
    try:
        agent=sim.get_agent(0)
        P=np.asarray(sim._sensors['color']._sensor_object.render_camera.projection_matrix,dtype=float)
        assert np.allclose(P,camera['projection_matrix'],atol=1e-8,rtol=0)
        def render(pose,position):
            state=habitat_sim.AgentState();state.position=position
            state.rotation=quat_from_angle_axis(pose[2]-math.pi/2,np.array([0.,1.,0.]))
            agent.set_state(state,reset_sensors=True);obs=sim.get_sensor_observations()
            actual=agent.get_state().sensor_states['color'];R=quaternion.as_rotation_matrix(actual.rotation)
            forward=R@np.array([0.,0.,-1.]);yaw=math.atan2(-forward[2],forward[0])
            assert np.linalg.norm(np.asarray(actual.position)-position)<1e-5
            assert abs(float(wrap(yaw-pose[2])))<1e-5
            return np.asarray(obs['color'])[...,:3].copy(),np.asarray(obs['depth']).copy(),R,yaw,np.asarray(actual.position)
        for tr in data['trajectories']:
            if tr['split']!='test':continue
            tid=tr['trajectory_id'];assert tid in ['room09_traj00','room09_traj01'] and tr['room_id']==9
            base=tr['frames'][0];position=base['camera_position_world'];pose0=base['planar_pose'];refs={}
            for deg,frame in [(0,0),(3,2),(5,4),(7,6)]:
                row=tr['frames'][frame];p=r/'collection_v1/trajectories'/tid/f'{frame:03d}.png'
                assert digest(p)==row['image_sha256'] and row['camera_position_world']==position
                assert abs(math.degrees(float(wrap(row['planar_pose'][2]-pose0[2])))-deg)<1e-8
                refs[str(deg)]=dict(path=str(p),sha256=digest(p),source='existing_trajectory',source_frame=frame,
                    camera_position_world=position,camera_rotation_c2w=row['camera_rotation_c2w'],yaw_deg_left_positive=deg)
            # Control readbacks do not create new image files or change source data.
            for frame in [0,2]:
                rgb,_,_,_,_=render(tr['frames'][frame]['planar_pose'],position)
                original=np.array(Image.open(r/'collection_v1/trajectories'/tid/f'{frame:03d}.png').convert('RGB'))
                assert np.array_equal(rgb,original),(tid,frame,int(np.abs(rgb.astype(int)-original.astype(int)).max()))
                controls.append(dict(trajectory_id=tid,frame=frame,exact_pixel_match=True))
            folder=out/tid;folder.mkdir(exist_ok=True)
            for deg in [-3,-5,-7]:
                pose=[*pose0[:2],float(wrap(pose0[2]+math.radians(deg)))]
                rgb,depth,R,yaw,actual=render(pose,position);name=case_name(deg);p=folder/(name+'.png')
                if p.exists():assert np.array_equal(np.array(Image.open(p)),rgb)
                else:
                    tmp=p.with_suffix('.tmp');Image.fromarray(rgb).save(tmp,format='PNG');tmp.replace(p)
                depth_path=folder/(name+'.depth.npy')
                if not depth_path.exists():
                    with depth_path.with_suffix('.tmp').open('wb') as f:np.save(f,depth)
                    depth_path.with_suffix('.tmp').replace(depth_path)
                row=dict(path=str(p),sha256=digest(p),depth_path=str(depth_path),depth_sha256=digest(depth_path),
                    source='new_habitat_render',yaw_deg_left_positive=deg,planar_pose=pose,
                    actual_yaw_delta_deg=math.degrees(float(wrap(yaw-pose0[2]))),
                    camera_position_world=actual.tolist(),camera_rotation_c2w=R.tolist())
                atomic(folder/(name+'.json'),row);refs[str(deg)]=row
            references[tid]=dict(room_id=9,start_pose=pose0,reference_views=refs)
        assert len(list(out.glob('*/*.png')))==6 and len(references)==2
        atomic(out/'manifest.json',dict(status='passed',camera=camera,references=references,
            controls=controls,new_rgb_count=6,reused_rgb_count=8,training_images_added=0,
            original_dataset_sha256=digest(r/'collection_v1/dataset.json')))
        print('HELDOUT_REFERENCES_PASSED',len(controls),6,flush=True)
    finally:sim.close()


if __name__=='__main__':main()
