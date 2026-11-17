"""Freeze existing native cases and render only past/goal views at fixed poses."""
import argparse
import copy
import math
from pathlib import Path
import numpy as np
from PIL import Image
from suite_common import read, digest, atomic, frozen
from context_controls_common import root, ANGLES, HISTORIES, UNI_ARMS, checked_reuse, audit_reuse_inputs, habitat_case, validate_group


def prepare(a):
    from confirmation_common import selected_weights
    from train_habitat_model import CHECKPOINTS
    out = root(a.study); out.mkdir(parents=True, exist_ok=True)
    protocol=read(out/'protocol.json');assert protocol['world_model_updates'] == 0
    adapted = selected_weights(a.study)
    weights = {}
    for model in ['nwm', 'raenwm', 'uniwm']:
        path = Path(CHECKPOINTS[model])
        identity = path/'adapter_model.safetensors' if model == 'uniwm' else path
        weights[model] = dict(pre_habitat=dict(path=str(path), sha256=digest(identity),
            identity='our 1X-200 reproduction' if model == 'uniwm' else 'released EMA'), adapted=adapted[model])
        for arm,role in [('pre_habitat','before'),('adapted','after')]:
            expected=protocol['weights'][model][role]
            assert weights[model][arm]['path']==expected['path']
            assert weights[model][arm]['sha256']==expected.get('sha256',expected.get('adapter_sha256'))
        old = read(a.root/'plans/native_dev20_remaining_v1'/(model+'.json'))
        oldrun=Path('runs/navigation-world-model-runs')/f'suite_dev20_v1_{model}'
        cfgpath=oldrun/'config_components.json';cfg=read(cfgpath)
        assert cfg['code']['suite_backends.py']==digest(Path(__file__).with_name('suite_backends.py'))
        assert cfg['weights'][str(identity)]==weights[model]['pre_habitat']['sha256']
        for fp,sha in cfg.get('native_files',{}).items():assert digest(fp)==sha,fp
        rows = [copy.deepcopy(c) for c in old['cases'] if c['experiment']=='components'
                and c['condition']=='original' and c['repeat']==0 and c['memory_mode']=='none']
        assert len(rows) == 20 and len({c['state_id'] for c in rows}) == 20
        cases = []
        for arm in weights[model]:
            for source in rows:
                c = copy.deepcopy(source)
                c.update(original_key=source['key'], cohort='native', model=model, weight_arm=arm,
                         context_arm='observed_history', target_image=source['reference_future'], reuse=None)
                c['key'] = f'native/{arm}/{source["key"]}'
                if arm == 'pre_habitat':
                    oldrun = Path('runs/navigation-world-model-runs')/f'suite_dev20_v1_{model}'
                    p = oldrun/(source['key']+'.png')
                    if not p.exists():
                        assert source['reuse']; p = Path(source['reuse']['path'])
                    c['reuse'] = checked_reuse(p, source['reuse']['sha256'] if source['reuse'] else None)
                    c['reuse']['input_audit']=audit_reuse_inputs(c['reuse'],c)
                    c['reuse'].update(weight_and_backend_config=str(cfgpath),weight_and_backend_config_sha256=digest(cfgpath))
                assert max(0, c['current_index']) < c['future_index']
                assert c['reference_future'] not in c['history_paths']
                # Native goal is a fixed navigation input, never selected by the queried action.
                if model == 'uniwm':
                    assert c['start_image'] and c['goal_image'] != c['reference_future']
                cases.append(c)
        inputs = {p:digest(p) for c in cases for p in c['history_paths']+[c['start_image'], c['goal_image']] if p}
        targets = {c['target_image']:digest(c['target_image']) for c in cases}
        frozen(out/f'native_{model}.json', dict(cases=cases, input_hashes=inputs, target_hashes=targets,
            source_manifest_sha256=digest(a.root/'plans/native_dev20_remaining_v1'/(model+'.json'))))
    frozen(out/'weights.json', weights)
    original = a.study/'v2/paper_confirmation_v1'
    frozen(out/'source_metadata.json', dict(confirmation_manifest_sha256=digest(original/'confirmation_manifest.json'),
        reference_manifest_sha256=digest(original/'references_manifest.json'),
        calibration_complete_sha256=digest(original/'flow/references/complete.json'),
        original_prediction_records_unchanged=True))
    atomic(out/'prepare_complete.json', dict(status='passed', native_conditions=120, native_new_predictions=60,
        native_reused_predictions=60, weights_sha256=digest(out/'weights.json'), world_model_updates=0))


def render(a):
    import habitat_sim
    import quaternion
    from habitat_sim.utils.common import quat_from_angle_axis
    from habitat_study_common import wrap
    out = root(a.study); old = a.study/'v2/paper_confirmation_v1'
    refs = read(old/'references_manifest.json'); camera = refs['camera']; p = read(old/'protocol.json')
    all_views = {}; report = [];history_actions={}
    for scene in p['confirmation']['scenes']:
        cfg = habitat_sim.SimulatorConfiguration(); cfg.scene_id=str(old/'assets'/scene['glb'])
        cfg.enable_physics=False; cfg.gpu_device_id=0
        specs=[]
        for name, kind in [('color',habitat_sim.SensorType.COLOR), ('depth',habitat_sim.SensorType.DEPTH)]:
            spec=habitat_sim.CameraSensorSpec();spec.uuid=name;spec.sensor_type=kind
            spec.resolution=[camera['height'],camera['width']];spec.hfov=camera['hfov_deg']
            spec.position=[0.,0.,0.];spec.near=.01;spec.far=100.;specs.append(spec)
        agent=habitat_sim.agent.AgentConfiguration();agent.sensor_specifications=specs
        sim=habitat_sim.Simulator(habitat_sim.Configuration(cfg,[agent]))
        try:
            for sid, ref in refs['references'].items():
                if ref['scene_id'] != scene['id']:continue
                zero=ref['reference_views']['0']; pose0=ref['start_pose']; views={}
                for angle in [0,-9,-6,-2,9,6,2,30,-30]:
                    pose=[*pose0[:2],float(wrap(pose0[2]+math.radians(angle)))]
                    state=habitat_sim.AgentState();state.position=zero['camera_position_world']
                    state.rotation=quat_from_angle_axis(pose[2]-math.pi/2,np.array([0.,1.,0.]))
                    sim.get_agent(0).set_state(state,reset_sensors=True);obs=sim.get_sensor_observations()
                    actual=sim.get_agent(0).get_state().sensor_states['color']; R=quaternion.as_rotation_matrix(actual.rotation)
                    forward=R@np.array([0.,0.,-1.]);yaw=math.atan2(-forward[2],forward[0])
                    assert np.linalg.norm(np.asarray(actual.position)-zero['camera_position_world'])<1e-5
                    assert abs(float(wrap(yaw-pose[2])))<1e-5
                    rgb=np.asarray(obs['color'])[...,:3].copy();depth=np.asarray(obs['depth']).copy()
                    folder=out/'context_views'/sid;folder.mkdir(parents=True,exist_ok=True)
                    path=folder/f'yaw_{angle:+d}.png'; dp=path.with_suffix('.depth.npy')
                    if path.exists():assert np.array_equal(np.asarray(Image.open(path)),rgb)
                    else:
                        tmp=path.with_suffix('.tmp');Image.fromarray(rgb).save(tmp,format='PNG');tmp.replace(path)
                    if dp.exists():assert np.array_equal(np.load(dp,allow_pickle=False),depth)
                    else:
                        with dp.with_suffix('.tmp').open('wb') as f:np.save(f,depth)
                        dp.with_suffix('.tmp').replace(dp)
                    row=dict(path=str(path),sha256=digest(path),depth_path=str(dp),depth_sha256=digest(dp),
                        planar_pose=pose,camera_position_world=np.asarray(actual.position).tolist(),camera_rotation_c2w=R.tolist())
                    frozen(path.with_suffix('.json'),row); views[str(angle)]=row
                    if angle==0:
                        assert digest(path)==zero['sha256'], 'Current view must reproduce exactly'
                        views['0']=dict(zero,planar_pose=pose0)
                all_views[sid]=views;report.append(dict(state_id=sid,current_exact=True))
                history_actions[sid]={}
                for name,angles in HISTORIES.items():
                    actions=[]
                    for first,second in zip(angles,angles[1:]):
                        R=np.asarray(views[str(first)]['camera_rotation_c2w']);S=np.asarray(views[str(second)]['camera_rotation_c2w'])
                        rel=R.T@S;dyaw=math.atan2(rel[0,2],rel[0,0])
                        assert abs(dyaw-math.radians(second-first))<1e-5
                        actions.append([0.,0.,dyaw])
                    history_actions[sid][name]=actions
                atomic(out/'render_progress.json',dict(positions=len(all_views),total=12));print('CONTEXT_RENDER',sid,flush=True)
        finally:sim.close()
    assert len(all_views)==12
    frozen(out/'context_views.json',dict(camera=camera,views=all_views,current_checks=report,
        histories=HISTORIES,actual_relative_actions=history_actions,
        causal_interpretation='new controlled simulated past; not recovered original history'))
    original=read(old/'confirmation_manifest.json')
    sources=[c for c in original['cases'] if c['repeat']==0 and c['target_yaw_deg'] in ANGLES]
    assert len(sources)==36
    for model in ['nwm','raenwm','uniwm']:
        cases=[];arms=UNI_ARMS if model=='uniwm' else HISTORIES
        for weight in ['pre_habitat','adapted']:
            for arm in arms:
                for source in sources:
                    c=habitat_case(source,weight,arm,all_views[source['state_id']],model)
                    static=(arm=='s0_g0' if model=='uniwm' else arm=='static')
                    if weight=='adapted' and static:
                        c['reuse']=checked_reuse(old/'confirmation'/model/(source['key']+'.png'))
                        c['reuse']['input_audit']=audit_reuse_inputs(c['reuse'],c)
                        cfgpath=old/'confirmation'/model/'config.json';cfg=read(cfgpath)
                        assert cfg['weights']['sha256']==read(out/'weights.json')[model]['adapted']['sha256']
                        c['reuse'].update(weight_and_backend_config=str(cfgpath),weight_and_backend_config_sha256=digest(cfgpath))
                    cases.append(c)
        for weight in ['pre_habitat','adapted']:
            for arm in arms:
                for sid in all_views:
                    validate_group([c for c in cases if c['weight_arm']==weight and c['context_arm']==arm and c['state_id']==sid])
        inputs={p:digest(p) for c in cases for p in c['history_paths']+[c['start_image'],c['goal_image']]}
        targets={c['target_image']:digest(c['target_image']) for c in cases}
        expected=432 if model=='uniwm' else 216
        assert len(cases)==expected
        frozen(out/f'habitat_{model}.json',dict(cases=cases,input_hashes=inputs,target_hashes=targets,
            original_manifest_sha256=digest(old/'confirmation_manifest.json'),context_views_sha256=digest(out/'context_views.json')))
    atomic(out/'render_complete.json',dict(status='passed',positions=12,new_past_rgb=72,new_goal_rgb=24,
        verification_zero_rgb=12,depth_arrays=108,current_exact_replays=12,
        habitat_conditions=864,habitat_new_predictions=756,habitat_reused_predictions=108,
        world_model_updates=0))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--study',type=Path,required=True)
    ap.add_argument('--root',type=Path,required=True);ap.add_argument('--phase',choices=['prepare','render'],required=True)
    a=ap.parse_args();prepare(a) if a.phase=='prepare' else render(a)
