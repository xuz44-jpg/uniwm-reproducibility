"""Independent, inference-only controls; never mutate the original study."""
from pathlib import Path
import copy
from suite_common import read, digest

RUN = 'input_context_controls_v1'
ANGLES = [0, 5, -5]
HISTORIES = {'static': [0, 0, 0, 0], 'left_approach': [-9, -6, -2, 0],
             'right_approach': [9, 6, 2, 0]}
UNI_ARMS = {f's{s}_g{g}': (s, g) for s in [0, 1] for g in ['0', 'left30', 'right30']}


def root(study):
    return Path(study) / 'v2' / RUN


def checked_reuse(png, expected=None):
    png = Path(png)
    assert png.is_file(), png
    sha = digest(png)
    if expected is not None:
        assert sha == expected, png
    result = dict(path=str(png), sha256=sha)
    rec = png.with_suffix('.json')
    if rec.exists():
        old = read(rec)
        assert old.get('status', 'ok') == 'ok'
        if 'image_sha256' in old:
            assert old['image_sha256'] == sha
        result.update(record_path=str(rec), record_sha256=digest(rec))
    return result


def audit_reuse_inputs(reuse, case):
    assert 'record_path' in reuse, 'Reuse needs an original generation record'
    old=read(reuse['record_path']);assert digest(reuse['record_path'])==reuse['record_sha256']
    prior=old['case']
    fields=['state_id','history_paths','start_image','goal_image','start_pose','query_pose',
            'action','action_text','seed','horizon_frames','memory_mode','memory_entries']
    for k in fields:assert prior[k]==case[k],('Reused input differs',k)
    for path,sha in old.get('input_hashes',{}).items():assert digest(path)==sha,path
    if case['model']=='uniwm' and old.get('sampling'):
        for k,v in dict(temperature=.7,top_k=50,top_p=.9).items():assert old['sampling'][k]==v
    return dict(status='passed',compared_case_fields=fields,original_input_hashes_checked=True,
                old_sampling=old.get('sampling'),record_sha256=reuse['record_sha256'])


def habitat_case(original, weight_arm, context_arm, views, model):
    c = copy.deepcopy(original)
    c.update(original_key=original['key'], cohort='habitat', model=model,
             weight_arm=weight_arm, context_arm=context_arm, reuse=None)
    if model == 'uniwm':
        s, g = UNI_ARMS[context_arm]
        start = views['-9'] if s else views['0']
        c['start_image'] = start['path']
        c['start_pose'] = start['planar_pose']
        c['goal_image'] = views[{'0': '0', 'left30': '30', 'right30': '-30'}[g]]['path']
    else:
        c['history_paths'] = [views[str(d)]['path'] for d in HISTORIES[context_arm]]
    c['key'] = f'habitat/{weight_arm}/{context_arm}/{original["key"]}'
    return c


def validate_group(cases):
    """A query cannot pick its own history/goal, or expose nonzero targets."""
    first = cases[0]
    assert sorted(c['target_yaw_deg'] for c in cases) == [-5, 0, 5]
    for field in ['history_paths', 'start_image', 'start_pose', 'goal_image', 'query_pose', 'seed']:
        assert all(c[field] == first[field] for c in cases), field
    forbidden = {c['target_image'] for c in cases if c['target_yaw_deg']}
    used = set(first['history_paths'] + [first['start_image'], first['goal_image']])
    assert not used & forbidden, 'Nonzero future reference appears as an input'
    for c in cases:
        assert c['history_paths'][-1] == first['history_paths'][-1]


def aggregate_direction(counts):
    return dict(**counts, pooled_near_zero_unknown=counts['near_zero'] + counts['unknown'],
                acceptance_not_passed=counts['reverse'] + counts['near_zero'] + counts['unknown'],
                unknown_is_observed_zero_or_wrong_turn=False)
