"""Contracts for a fixed-weight action-scale extension of the completed study."""
import copy
import math
from pathlib import Path
from confirmation_common import selected_weights, ARMS
from uniwm_v3_common import action_text, action_tokens

RUN = 'angle_scale_v1'
OLD_ANGLES = [0, 3, -3, 5, -5, 7, -7]
NEW_ANGLES = [10, -10, 15, -15]
ANGLES = OLD_ANGLES + NEW_ANGLES


def root(study):
    return Path(study)/'v2'/RUN


def expanded_cases(old_cases, references):
    """Retain every old case exactly, including its seed and reference paths."""
    groups = {}
    for c in old_cases:
        groups.setdefault((c['state_id'], c['repeat']), {})[c['target_yaw_deg']] = c
    result = []
    for (sid, repeat), group in groups.items():
        assert set(group) == set(OLD_ANGLES)
        assert len({c['seed'] for c in group.values()}) == 1
        for angle in ANGLES:
            if angle in group:
                c = copy.deepcopy(group[angle])
            else:
                c = copy.deepcopy(group[0])
                c.update(key=f'{sid}/r{repeat}/yaw_{angle:+d}',
                    target_yaw_deg=angle, action=[0., 0., math.radians(angle)],
                    target_image=references[sid]['reference_views'][str(angle)]['path'])
                c.update(action_text=action_text(c['action']), action_tokens=action_tokens(c['action']))
            result.append(c)
    assert len(result) == 396
    assert sum(c['target_yaw_deg'] in NEW_ANGLES for c in result) == 144
    return result
