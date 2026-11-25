"""Pre-registered history-intervention confirmation at new positions (NWM only, inference only)."""
from pathlib import Path

RUN = 'history_confirmation_v2'
REF_ANGLES = [0, 3, -3, 5, -5, 7, -7]      # the unchanged reference gate needs 3/5/7 separation
HISTORY_ANGLES = [-9, -6, -2, 9, 6, 2]
QUERY_ANGLES = [0, 5, -5]
HISTORIES = {'static': [0, 0, 0, 0], 'left_approach': [-9, -6, -2, 0], 'right_approach': [9, 6, 2, 0]}
SEEDS = [202610270, 202610271, 202610272]
WEIGHTS = ['pre_habitat', 'adapted']
SELECTION_SEED = 20261118
POSITIONS_PER_SCENE = 6
MIN_SEPARATION_M = 1.25
MAX_ATTEMPTS = 20000
OLD_RUN = 'paper_confirmation_v1'


def root(study):
    return Path(study) / 'v2' / RUN


def case_key(weight, history, sid, repeat, angle):
    return f'{weight}/{history}/{sid}/r{repeat}/yaw_{angle:+d}'
