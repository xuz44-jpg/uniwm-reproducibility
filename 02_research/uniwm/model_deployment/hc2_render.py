"""Draw new positions by the frozen rule, render references and history views, write the case manifest.

Runs in the Habitat environment. Never reads model outputs. Mirrors render_confirmation_v1.py and
prepare_context_controls_v1.py (render phase) with a new seed and exclusion of the 12 old positions.
"""
import argparse
import math
from pathlib import Path
import numpy as np
from PIL import Image
from suite_common import read, digest, atomic, frozen
from habitat_study_common import wrap
from uniwm_v3_common import action_text, action_tokens
from hc2_common import (root, REF_ANGLES, HISTORY_ANGLES, QUERY_ANGLES, HISTORIES, SEEDS, WEIGHTS,
                        SELECTION_SEED, POSITIONS_PER_SCENE, MIN_SEPARATION_M, MAX_ATTEMPTS, OLD_RUN, case_key)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--study', type=Path, required=True); a = ap.parse_args()
    out = root(a.study); old = a.study / 'v2' / OLD_RUN
    freeze = read(out / 'execution_freeze.json'); assert digest(out / 'protocol.json') == freeze['protocol_sha256']
    for path, sha in freeze['source_hashes'].items(): assert digest(path) == sha, path
    p = read(old / 'protocol.json'); camera = read(a.study / 'collection_v1/camera.json')
    assert camera == read(old / 'references_manifest.json')['camera']
    import habitat_sim
    import quaternion
    from habitat_sim.utils.common import quat_from_angle_axis

    def simulator(scene):
        cfg = habitat_sim.SimulatorConfiguration(); cfg.scene_id = str(scene); cfg.enable_physics = False; cfg.gpu_device_id = 0
        specs = []
        for name, kind in [('color', habitat_sim.SensorType.COLOR), ('depth', habitat_sim.SensorType.DEPTH)]:
            spec = habitat_sim.CameraSensorSpec(); spec.uuid = name; spec.sensor_type = kind
            spec.resolution = [camera['height'], camera['width']]; spec.hfov = camera['hfov_deg']
            spec.position = [0., 0., 0.]; spec.near = .01; spec.far = 100.; specs.append(spec)
        agent = habitat_sim.agent.AgentConfiguration(); agent.sensor_specifications = specs
        return habitat_sim.Simulator(habitat_sim.Configuration(cfg, [agent]))

    def render(sim, folder, angle, pose0, position):
        pose = [*pose0[:2], float(wrap(pose0[2] + math.radians(angle)))]
        agent = sim.get_agent(0); state = habitat_sim.AgentState(); state.position = position
        state.rotation = quat_from_angle_axis(pose[2] - math.pi / 2, np.array([0., 1., 0.]))
        agent.set_state(state, reset_sensors=True); obs = sim.get_sensor_observations()
        actual = agent.get_state().sensor_states['color']; R = quaternion.as_rotation_matrix(actual.rotation)
        forward = R @ np.array([0., 0., -1.]); yaw = math.atan2(-forward[2], forward[0])
        assert np.linalg.norm(np.asarray(actual.position) - position) < 1e-5
        assert abs(float(wrap(yaw - pose[2]))) < 1e-5
        rgb = np.asarray(obs['color'])[..., :3].copy(); depth = np.asarray(obs['depth']).copy()
        folder.mkdir(parents=True, exist_ok=True); path = folder / f'yaw_{angle:+d}.png'; dp = path.with_suffix('.depth.npy')
        if path.exists(): assert np.array_equal(np.asarray(Image.open(path)), rgb)
        else:
            tmp = path.with_suffix('.tmp'); Image.fromarray(rgb).save(tmp, format='PNG'); tmp.replace(path)
        if dp.exists(): assert np.array_equal(np.load(dp, allow_pickle=False), depth)
        else:
            with dp.with_suffix('.tmp').open('wb') as f: np.save(f, depth)
            dp.with_suffix('.tmp').replace(dp)
        row = dict(path=str(path), sha256=digest(path), depth_path=str(dp), depth_sha256=digest(dp),
                   yaw_deg_left_positive=angle, planar_pose=pose, camera_position_world=np.asarray(actual.position).tolist(),
                   camera_rotation_c2w=R.tolist())
        frozen(path.with_suffix('.json'), row); return row

    old_points = {s: [x['floor_world_xyz'] for x in read(old / 'selection' / f'{s}.json')['points']]
                  for s in ['apartment_1', 'skokloster_castle']}
    all_old = [pt for pts in old_points.values() for pt in pts]
    references = {}; selection = {}; shortfall = 0
    scenes = p['confirmation']['scenes']; assert [s['id'] for s in scenes] == ['apartment_1', 'skokloster_castle']
    for index, scene in enumerate(scenes):
        for filename, sha in scene['files'].items(): assert digest(old / 'assets' / filename) == sha
        target = POSITIONS_PER_SCENE + (shortfall if scene['id'] == 'skokloster_castle' else 0)
        sim = simulator(old / 'assets' / scene['glb'])
        try:
            assert sim.pathfinder.is_loaded or sim.pathfinder.load_nav_mesh(str(old / 'assets' / scene['navmesh']))
            sim.seed(SELECTION_SEED + index)
            P = np.asarray(sim._sensors['color']._sensor_object.render_camera.projection_matrix, dtype=float)
            assert np.allclose(P, camera['projection_matrix'], atol=1e-8, rtol=0)
            sp = out / 'selection' / f'{scene["id"]}.json'
            if sp.exists(): selected = read(sp)
            else:
                chosen = []; attempts = 0; rng = np.random.default_rng(SELECTION_SEED + index)
                while len(chosen) < target and attempts < MAX_ATTEMPTS:
                    attempts += 1
                    point = np.asarray(sim.pathfinder.get_random_navigable_point(), dtype=float)
                    if not np.isfinite(point).all(): continue
                    if any(np.linalg.norm(point - np.asarray(x)) < MIN_SEPARATION_M for x in all_old): continue
                    if any(np.linalg.norm(point - np.asarray(x['floor_world_xyz'])) < MIN_SEPARATION_M for x in chosen): continue
                    yaw = float(rng.uniform(-math.pi, math.pi)); position = point.copy(); position[1] += 1.5
                    chosen.append(dict(state_id=f'{scene["id"]}_n{len(chosen):02d}', floor_world_xyz=point.tolist(),
                                       position_world=position.tolist(), pose=[float(point[0]), float(-point[2]), yaw]))
                selected = dict(scene=scene['id'], seed=SELECTION_SEED + index, target=target, attempts=attempts,
                                points=chosen, excluded_old_positions=len(all_old), image_or_model_selection=False)
                frozen(sp, selected)
            if scene['id'] == 'apartment_1': shortfall = POSITIONS_PER_SCENE - len(selected['points'])
            selection[scene['id']] = selected
            for entry in selected['points']:
                sid = entry['state_id']; folder = out / 'references' / sid
                views = {str(d): render(sim, folder, d, entry['pose'], entry['position_world']) for d in REF_ANGLES}
                hist = {str(d): render(sim, folder, d, entry['pose'], entry['position_world']) for d in HISTORY_ANGLES}
                again = render(sim, folder, 0, entry['pose'], entry['position_world'])
                assert again['sha256'] == views['0']['sha256'], 'Current view must replay exactly'
                references[sid] = dict(scene_id=scene['id'], start_pose=entry['pose'], reference_views=views, history_views=hist)
                atomic(out / 'render_progress.json', dict(positions=len(references)))
                print('HC2_POSITION', sid, flush=True)
        finally:
            sim.close()
    assert len(references) == 2 * POSITIONS_PER_SCENE, (len(references), 'not enough separated positions')
    frozen(out / 'references_manifest.json', dict(camera=camera, references=references, selections=selection,
           protocol_sha256=digest(out / 'protocol.json'), model_predictions_seen=False))
    cases = []
    for weight in WEIGHTS:
        for history, angles in HISTORIES.items():
            for sid, ref in references.items():
                allv = dict(ref['reference_views'], **ref['history_views'])
                current = ref['reference_views']['0']['path']
                hpaths = [allv[str(d)]['path'] for d in angles]; assert hpaths[-1] == current
                for repeat, seed in enumerate(SEEDS):
                    for angle in QUERY_ANGLES:
                        action = [0., 0., math.radians(angle)]
                        cases.append(dict(key=case_key(weight, history, sid, repeat, angle), state_id=sid,
                            scene_id=ref['scene_id'], model='nwm', weight_arm=weight, context_arm=history,
                            history='teacher', history_paths=hpaths, start_image=current, goal_image=current,
                            target_image=ref['reference_views'][str(angle)]['path'], target_yaw_deg=angle,
                            action=action, action_text=action_text(action), action_tokens=action_tokens(action),
                            start_pose=ref['start_pose'], query_pose=ref['start_pose'], current_index=0,
                            memory_entries=[], memory_mode='none', repeat=repeat, seed=seed, horizon_frames=1, reuse=None))
    targets = {c['target_image'] for c in cases if c['target_yaw_deg']}
    for c in cases: assert not set(c['history_paths']) & targets, 'A queried future appears as an input'
    assert len(cases) == 648
    frozen(out / 'cases_nwm.json', dict(cases=cases,
           input_hashes={q: digest(q) for c in cases for q in c['history_paths'] + [c['start_image']]},
           target_hashes={c['target_image']: digest(c['target_image']) for c in cases},
           references_sha256=digest(out / 'references_manifest.json')))
    atomic(out / 'render_complete.json', dict(status='passed', positions=len(references),
           rgb_per_position=len(REF_ANGLES) + len(HISTORY_ANGLES), cases=len(cases),
           selection={k: dict(found=len(v['points']), attempts=v['attempts']) for k, v in selection.items()},
           cases_sha256=digest(out / 'cases_nwm.json'), world_model_updates=0))


if __name__ == '__main__':
    main()
