"""Restore FP32 adapter masters after a native BF16 model load.

Only an exact checkpoint match or its exact BF16 round trip is accepted before
restoring. This never repairs an unexplained mismatch or changes frozen tables.
"""


def restore_masters(model, checkpoint):
    from pathlib import Path
    import torch
    from safetensors import safe_open
    from verify_uniwm_embeddings import compare_matrix

    class RoundedSlice:
        def __init__(self, source):
            self.source = source

        def get_shape(self):
            return self.source.get_shape()

        def __getitem__(self, item):
            return self.source[item].to(torch.bfloat16).float()

    records = {}
    with safe_open(str(Path(checkpoint) / 'adapter_model.safetensors'), framework='pt', device='cpu') as saved:
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            assert parameter.dtype == torch.float32, (name, parameter.dtype)
            key = name.replace('.modules_to_save.default.', '.')
            for part in ['lora_A', 'lora_B']:
                key = key.replace(f'.{part}.default.', f'.{part}.')
            assert key in saved.keys(), (name, key)
            source = saved.get_slice(key)
            before = compare_matrix(parameter, source)
            rounded = None if before['matched'] else compare_matrix(parameter, RoundedSlice(source))
            assert before['matched'] or rounded['matched'], (name, before, rounded)
            if not before['matched']:
                with torch.no_grad():
                    for start in range(0, parameter.shape[0], 1024):
                        end = min(start + 1024, parameter.shape[0])
                        parameter[start:end].copy_(source[start:end].to(parameter.device, dtype=torch.float32))
            after = compare_matrix(parameter, source)
            assert after['matched'], (name, after)
            records[name] = dict(checkpoint_key=key, before=before,
                                 exact_bf16_round_trip=rounded, after=after)
    assert records
    return dict(status='passed', parameters_checked=len(records),
                restored_parameters=sum(not v['before']['matched'] for v in records.values()),
                optimizer_updates=0, frozen_embedding_modified=False, parameters=records)
