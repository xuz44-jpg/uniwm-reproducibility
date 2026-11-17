"""Contracts for the isolated, symmetric UniWM retraining experiment."""
from pathlib import Path
import math

RUN = 'uniwm_symmetric_aligned_v3'
BASELINE = 'external/uniwm-action-pilot/checkpoints/1x200'
SEED = 20261023


def experiment(study):
    return Path(study) / 'v2' / RUN


def action_tokens(action):
    result = []
    for value, prefix in zip(action, ['dx', 'dy', 'dyaw']):
        sign = 'pos' if value >= 0 else 'neg'
        result.append(f'<{prefix}_{sign}_bin_{math.floor(abs(value)/.01):02d}>')
    return result


def action_text(action):
    x, y, yaw = action_tokens(action)
    return f'Move by dx: {x}, dy: {y}, dyaw: {yaw}'


def stopping_update(state, loss, step, min_step=300, patience=3, relative_delta=.005):
    """Checkpoint best actual loss; plateau uses a separate significant anchor."""
    assert math.isfinite(loss)
    state = dict(state)
    if loss < state.get('best_loss', float('inf')):
        state.update(best_loss=loss, best_step=step)
    anchor = state.get('anchor_loss', float('inf'))
    if loss < anchor * (1-relative_delta):
        state.update(anchor_loss=loss, bad_checks=0)
    elif step >= min_step:
        state['bad_checks'] = state.get('bad_checks', 0) + 1
    state['stop'] = step >= min_step and state.get('bad_checks', 0) >= patience
    return state
