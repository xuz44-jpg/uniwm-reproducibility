"""Guard-owned Habitat rendering only. No world-model imports or fitting.

Twenty fixed 16-frame camera trajectories plus seven fixed-position references.
Writes actual reached poses and body-frame deltas, never substitutes commands
for the navigation-mesh-constrained displacement. Atomic per-image resume.
"""
import argparse
import json
import math
import os
from pathlib import Path
import sys
import numpy as np
from PIL import Image
from suite_common import atomic, digest, frozen, read
from habitat_study_common import case_name, proposed_pose, relative_action, wrap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--assets', type=Path, required=True)
    ap.add_argument('--plan', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    plan = read(a.plan); out = a.out; out.mkdir(parents=True, exist_ok=True)
    assert plan['world_model_training_steps_per_model'] == 2000
    assert len(plan['trajectories']) == 20
    assert len(plan['commanded_body_actions']) == 15
    for name, sha in plan['source_hashes'].items():
        assert digest(a.assets / name) == sha
    frozen(out / 'render_config.json', dict(plan_sha256=digest(a.plan),
        script_sha256=digest(__file__), common_sha256=digest(Path(__file__).with_name('habitat_study_common.py'))))
    # Set externally by the priority guard; no persistent GPU worker outside it.
    import habitat_sim
    import quaternion
    from habitat_sim.utils.common import quat_from_angle_axis
    cfg = habitat_sim.SimulatorConfiguration()
    cfg.scene_dataset_config_file = str(a.assets / 'mp3d.scene_dataset_config.json')
    cfg.scene_id = str(a.assets / '17DRP5sb8fy/17DRP5sb8fy.glb')
    cfg.enable_physics = False
    cfg.gpu_device_id = 0
    specs = []
    for name, kind in [('color', habitat_sim.SensorType.COLOR), ('depth', habitat_sim.SensorType.DEPTH)]:
        spec = habitat_sim.CameraSensorSpec()
        spec.uuid = name; spec.sensor_type = kind
        spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
        spec.resolution = [plan['height'], plan['width']]
        spec.hfov = plan['hfov_deg']
        spec.position = [0., 0., 0.]
        spec.near = .01; spec.far = 100.
        specs.append(spec)
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = specs
    sim = habitat_sim.Simulator(habitat_sim.Configuration(cfg, [agent_cfg]))
    try:
        sim.seed(plan['seed'])
        if not sim.pathfinder.is_loaded:
            assert sim.pathfinder.load_nav_mesh(str(a.assets / '17DRP5sb8fy/17DRP5sb8fy.navmesh'))
        agent = sim.get_agent(0)
        render_sensor = sim._sensors['color']._sensor_object
        P = np.asarray(render_sensor.render_camera.projection_matrix, dtype=float)
        w, h = plan['width'], plan['height']
        K = np.array([[P[0, 0]*w/2, 0., (w-1)/2], [0., P[1, 1]*h/2, (h-1)/2], [0., 0., 1.]])
        assert abs(K[0, 0] - w/(2*math.tan(math.radians(plan['hfov_deg'])/2))) < 1e-3
        frozen(out / 'camera.json', dict(K=K.tolist(), distortion=[0.]*5,
            projection_matrix=P.tolist(), width=w, height=h, hfov_deg=plan['hfov_deg'],
            native_axes='Habitat world +Y up; camera +X right,+Y up,-Z forward',
            planar_axes='x=world_X, y=-world_Z; yaw=atan2(forward_y,forward_x), positive left',
            pixel_convention='integer pixel centers; principal point ((W-1)/2,(H-1)/2)',
            simulator_version=habitat_sim.__version__, python=sys.version))

        def set_pose(pose, height):
            state = habitat_sim.AgentState()
            state.position = [pose[0], height, -pose[1]]
            state.rotation = quat_from_angle_axis(pose[2] - math.pi/2, np.array([0., 1., 0.]))
            agent.set_state(state, reset_sensors=True)
            obs = sim.get_sensor_observations()
            actual = agent.get_state().sensor_states['color']
            assert np.linalg.norm(np.asarray(actual.position)-state.position) < 1e-5
            R = quaternion.as_rotation_matrix(actual.rotation)
            forward = R @ np.array([0., 0., -1.])
            actual_yaw = math.atan2(-forward[2], forward[0])
            assert abs(float(wrap(actual_yaw-pose[2]))) < 1e-5
            return np.asarray(obs['color'])[..., :3].copy(), np.asarray(obs['depth']).copy(), actual, R

        trajectories = []
        for spec in plan['trajectories']:
            tid = spec['trajectory_id']; folder = out / 'trajectories' / tid
            folder.mkdir(parents=True, exist_ok=True)
            # Scan z-up coordinates -> Habitat's y-up coordinates. Snap only
            # the ground position; then freeze its measured camera height.
            original = np.array(spec['source_camera_xyz'])
            expected_floor = np.array([original[0], original[2]-1.5, -original[1]])
            floor = np.asarray(sim.pathfinder.snap_point(expected_floor), dtype=float)
            assert np.isfinite(floor).all() and np.linalg.norm(floor-expected_floor) < .5, (tid, floor, expected_floor)
            height = float(floor[1]+1.5)
            bx = spec['room_xy_bounds']; center = [(bx[0]+bx[2])/2, (bx[1]+bx[3])/2]
            pose = [float(floor[0]), float(-floor[2]), math.atan2(center[1]+floor[2], center[0]-floor[0])]
            if abs(center[0]-pose[0])+abs(center[1]-pose[1]) < 1e-5: pose[2] = 0.
            selection = dict(trajectory_id=tid, floor_xyz=floor.tolist(), camera_height_world_y=height,
                start_pose=pose, source_panorama_id=spec['source_panorama_id'], selection='first two panorama IDs per room; no image-quality selection')
            frozen(folder / 'selection.json', selection)
            rows = []
            for t in range(16):
                record_path = folder / f'{t:03d}.json'
                proposed = None; constrained = None
                if t:
                    proposed = proposed_pose(pose, plan['commanded_body_actions'][t-1])
                    old_floor = np.array([pose[0], floor[1], -pose[1]])
                    desired_floor = np.array([proposed[0], floor[1], -proposed[1]])
                    in_room = bx[0] <= proposed[0] <= bx[2] and bx[1] <= proposed[1] <= bx[3]
                    reached = np.asarray(sim.pathfinder.try_step(old_floor, desired_floor)) if in_room else old_floor
                    assert np.isfinite(reached).all()
                    # Camera stays on one horizontal plane for interpretable yaw.
                    reached[1] = floor[1]
                    constrained = bool(np.linalg.norm(reached-desired_floor) > 1e-4)
                    pose = [float(reached[0]), float(-reached[2]), proposed[2]]
                expected = dict(trajectory_id=tid, frame=t, split=spec['split'], room_id=spec['room_id'],
                    planar_pose=pose, camera_position_world=[pose[0],height,-pose[1]])
                if record_path.exists():
                    row = read(record_path)
                    for k, v in expected.items(): assert row[k] == v, (tid, t, k)
                    assert digest(folder/f'{t:03d}.png') == row['image_sha256']
                    assert digest(folder/f'{t:03d}.depth.npy') == row['depth_sha256']
                else:
                    rgb, depth, actual, R = set_pose(pose, height)
                    p = folder/f'{t:03d}.png'; tmp = p.with_suffix('.tmp')
                    Image.fromarray(rgb).save(tmp, format='PNG'); tmp.replace(p)
                    pdepth = folder/f'{t:03d}.depth.npy'
                    with pdepth.with_suffix('.tmp').open('wb') as f: np.save(f, depth)
                    pdepth.with_suffix('.tmp').replace(pdepth)
                    row = dict(expected, image_sha256=digest(p), depth_sha256=digest(pdepth),
                        camera_rotation_c2w=R.tolist(), commanded_action=None if not t else plan['commanded_body_actions'][t-1],
                        actual_body_action=None if not t else relative_action(rows[-1]['planar_pose'],pose),
                        translation_constrained=constrained, proposed_pose=proposed)
                    atomic(record_path, row)
                rows.append(row)
                atomic(out/'progress.json', dict(stage='collect', trajectory=tid, frame=t,
                    trajectory_images_complete=len(list((out/'trajectories').glob('*/*.png'))), trajectory_images_total=320))
            trajectory = dict(spec, frames=rows, camera_height_world_y=height)
            atomic(folder/'trajectory.json', trajectory); trajectories.append(trajectory)
            print('TRAJECTORY_COMPLETE', tid, spec['split'], flush=True)

        # A single, preregistered training-room start, independent of scores.
        first = trajectories[0]; start = first['frames'][0]['planar_pose']; height = first['camera_height_world_y']
        refs = []; reference_dir = out/'references'; reference_dir.mkdir(exist_ok=True)
        zero_rgb = None
        for degrees in plan['reference_degrees_left_positive']:
            pose = [*start[:2], float(wrap(start[2]+math.radians(degrees)))]
            rgb, depth, actual, R = set_pose(pose, height)
            name = case_name(degrees); p = reference_dir/(name+'.png')
            if p.exists():
                old = np.array(Image.open(p).convert('RGB')); assert np.array_equal(old,rgb)
            else: Image.fromarray(rgb).save(p)
            np.save(reference_dir/(name+'.depth.npy'), depth)
            row = dict(case=name, target_yaw_deg_left_positive=degrees, planar_pose=pose,
                camera_position_world=np.asarray(actual.position).tolist(), camera_rotation_c2w=R.tolist(),
                image_sha256=digest(p), depth_sha256=digest(reference_dir/(name+'.depth.npy')))
            atomic(reference_dir/(name+'.json'), row); refs.append(row)
            if degrees == 0: zero_rgb = rgb
        rgb, _, _, _ = set_pose(start,height)
        assert zero_rgb is not None and np.array_equal(zero_rgb, rgb)
        assert len(list((out/'trajectories').glob('*/*.png'))) == 320
        atomic(out/'dataset.json',dict(plan_sha256=digest(a.plan), camera=read(out/'camera.json'),
            trajectories=trajectories,references=refs,zero_repeat_exact=True,
            scene_count=1, trajectory_count=20, trajectory_rgb_count=320, reference_rgb_count=7,
            world_model_predictions=0,world_model_optimizer_steps=0,
            limitations=['one scanned house','room-start separation and bounding boxes do not guarantee visual disjointness',
                         'rendered static geometry, not real robot dynamics','frame offsets are simulator steps, not measured hardware seconds']))
        atomic(out/'progress.json',dict(stage='complete',trajectory_rgb_count=320,reference_rgb_count=7,zero_repeat_exact=True))
    finally:
        sim.close()


if __name__ == '__main__': main()
