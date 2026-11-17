"""Frozen contracts for an inference-only confirmation and UniWM interventions."""
from pathlib import Path
import math
from suite_common import read, digest
from uniwm_v3_common import action_text, action_tokens

RUN = 'paper_confirmation_v1'
ANGLES = [0, 3, -3, 5, -5, 7, -7]
ARMS = {
    'native': dict(goal='zero', top_k=50, top_p=.9),
    'goal_left30': dict(goal='left30', top_k=50, top_p=.9),
    'goal_right30': dict(goal='right30', top_k=50, top_p=.9),
    'no_top_k': dict(goal='zero', top_k=0, top_p=.9),
    'no_top_p': dict(goal='zero', top_k=50, top_p=1.),
    'no_truncation': dict(goal='zero', top_k=0, top_p=1.),
}


def root(study):
    return Path(study) / 'v2' / RUN


def case(state_id, views, pose, repeat, scene_id, seed_base=202610270):
    result = []
    for angle in ANGLES:
        action = [0., 0., math.radians(angle)]
        source = views['0']['path']
        result.append(dict(key=f'{state_id}/r{repeat}/yaw_{angle:+d}',
            state_id=state_id, scene_id=scene_id, experiment='seven',
            history='teacher', history_paths=[source]*4, start_image=source,
            goal_image=source, target_image=views[str(angle)]['path'],
            target_yaw_deg=angle, action=action, action_text=action_text(action),
            action_tokens=action_tokens(action), start_pose=pose, query_pose=pose,
            current_index=0, memory_entries=[], memory_mode='none',
            repeat=repeat, seed=seed_base+repeat, horizon_frames=1))
    return result


def selected_weights(study):
    study = Path(study)
    result = {}
    for model in ['nwm', 'raenwm']:
        folder = study / 'v2/recon_symmetric_v3/training' / model
        done = read(folder/'complete.json')
        path = folder/'best_trainable.pt'
        assert digest(path) == done['checkpoint_sha256']
        result[model] = dict(path=str(path), sha256=done['checkpoint_sha256'],
            step=done['step'], optimizer_steps=done['optimizer_steps_completed'])
    folder = study/'v2/uniwm_symmetric_aligned_v3/training'
    done = read(folder/'complete.json'); path = folder/'final_adapter'
    assert digest(path/'adapter_model.safetensors') == done['adapter_sha256']
    result['uniwm'] = dict(path=str(path), sha256=done['adapter_sha256'],
        step=done['step'], optimizer_steps=done['optimizer_steps_completed'])
    return result
