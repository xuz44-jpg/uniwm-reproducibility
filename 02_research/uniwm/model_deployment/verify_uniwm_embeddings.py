"""Verify the effective embedding tables against tensors saved in a PEFT adapter.

The absence of action_embed.safetensors says nothing about full embedding tensors
inside adapter_model.safetensors. This verifier does not change any model weights.
"""
import hashlib
import json
from pathlib import Path


def compare_matrix(actual, stored_slice, block_rows=1024):
    """Compare every row, casting saved weights to the model's effective dtype."""
    import torch
    assert list(actual.shape) == stored_slice.get_shape()
    expected_hash, actual_hash = hashlib.sha256(), hashlib.sha256()
    mismatches = 0
    maximum = 0.0
    for begin in range(0, actual.shape[0], block_rows):
        end = min(actual.shape[0], begin + block_rows)
        observed = actual[begin:end].detach().cpu().contiguous()
        expected = stored_slice[begin:end].to(dtype=observed.dtype).contiguous()
        mismatches += int(torch.count_nonzero(observed != expected))
        maximum = max(maximum, float((observed.float() - expected.float()).abs().max()))
        actual_hash.update(observed.float().numpy().tobytes())
        expected_hash.update(expected.float().numpy().tobytes())
    return dict(shape=list(actual.shape), effective_dtype=str(actual.dtype),
                matched=mismatches == 0, mismatched_elements=mismatches,
                max_absolute_difference=maximum,
                actual_float32_sha256=actual_hash.hexdigest(),
                checkpoint_cast_float32_sha256=expected_hash.hexdigest())


def verify_loaded_embeddings(model, tokenizer, checkpoint):
    import torch
    from safetensors import safe_open
    from uniwm.extra_tokens import unwrap
    checkpoint = Path(checkpoint)
    saved_tokenizer = json.loads((checkpoint / 'tokenizer.json').read_text())
    saved_actions = {t['content']: t['id'] for t in saved_tokenizer['added_tokens'] if '_bin_' in t['content']}
    actual_actions = {t: i for t, i in tokenizer.get_vocab().items() if '_bin_' in t}
    assert saved_actions == actual_actions and len(saved_actions) == 1554
    for text, token_id in saved_actions.items():
        assert tokenizer.encode(text, add_special_tokens=False) == [token_id]
    ids = sorted(saved_actions.values())
    assert ids == list(range(65536, 67090))
    hf = unwrap(model)
    embedding = hf.model.embed_tokens.weight
    head_module = hf.lm_head
    if hasattr(head_module, 'modules_to_save'):
        active_adapter = head_module.active_adapter
        assert isinstance(active_adapter, str)
        head = head_module.modules_to_save[active_adapter].weight
    else:
        head = head_module.weight
    source = checkpoint / 'adapter_model.safetensors'
    # This run audits the legacy full-table recipe, without auxiliary overrides.
    assert not (checkpoint / 'action_embed.safetensors').exists()
    assert not (checkpoint / 'extra_tokens.safetensors').exists()
    with safe_open(source, framework='pt', device='cpu') as saved:
        embedding_key = 'base_model.model.model.embed_tokens.weight'
        head_key = 'base_model.model.lm_head.weight'
        assert embedding_key in saved.keys() and head_key in saved.keys()
        input_check = compare_matrix(embedding, saved.get_slice(embedding_key))
        output_check = compare_matrix(head, saved.get_slice(head_key))
        old_rows = saved.get_slice(embedding_key)[ids[0]:ids[-1]+1].float()
        new_rows = embedding[ids[0]:ids[-1]+1].detach().float().cpu()
        assert torch.equal(old_rows, new_rows)
        action_sha = hashlib.sha256(new_rows.numpy().tobytes()).hexdigest()
        duplicate_head_key = 'base_model.model.lm_head.modules_to_save.weight'
        duplicate_check = (compare_matrix(head, saved.get_slice(duplicate_head_key))
                           if duplicate_head_key in saved.keys() else None)
    report = dict(status='passed' if input_check['matched'] and output_check['matched'] else 'failed',
        checkpoint=str(checkpoint), input_tensor_key=embedding_key,
        input_embedding_matches_checkpoint=input_check['matched'], input_table=input_check,
        active_output_tensor_key=head_key, active_output_head_matches_checkpoint=output_check['matched'],
        output_table=output_check, duplicate_saved_output_head=duplicate_check,
        all_action_tokens_map_correctly=True, action_token_count=len(ids),
        action_token_id_range=[ids[0], ids[-1]], action_input_rows_sha256=action_sha,
        action_embedding_source='full input table saved inside adapter_model.safetensors',
        auxiliary_action_file_required=False, verification_changes_weights=False)
    assert report['status'] == 'passed', json.dumps(report)
    assert duplicate_check is None or duplicate_check['matched'], json.dumps(report)
    return report
