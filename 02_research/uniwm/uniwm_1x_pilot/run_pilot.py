"""Single-GPU inference only; preserve the existing UniWM loader/prompt/VQ protocol."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    tmp.replace(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', type=Path, required=True)
    ap.add_argument('--data', type=Path, required=True)
    ap.add_argument('--checkpoint', type=Path, required=True)
    ap.add_argument('--base-model', required=True)
    ap.add_argument('--manifest', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--max-images', type=int, default=0, help='0 runs all remaining planned cells')
    a = ap.parse_args()
    sys.path.insert(0, str(a.repo.resolve()))
    import numpy as np
    import torch
    from PIL import Image
    from transformers import set_seed
    from scripts.action_utils import action_to_text
    from scripts.load_model import load_model
    from scripts.prompt_builder import build_viz_prompt

    manifest = json.loads(a.manifest.read_text())
    assert len(manifest['states']) == 20 and manifest['expected_images'] == 240
    assert len({tuple(s['source']) for s in manifest['states']}) == 20
    assert not {tuple(s['source']) for s in manifest['states']} & {tuple(s) for s in manifest['split']['training_sources']}
    # Validate all source files before taking GPU memory. Raw future is never a prompt input.
    for state in manifest['states']:
        for key, relative in state['files'].items():
            assert sha(a.data / relative) == state['sha256'][key], (state['state_id'], key)
        assert len(set(state['action_text'].values())) == 4
    config = dict(manifest_sha256=sha(a.manifest), checkpoint=str(a.checkpoint.resolve()),
        checkpoint_files={p.name: sha(p) for p in sorted(a.checkpoint.iterdir()) if p.is_file()},
        code_files={str(p.relative_to(a.repo)): sha(p) for folder in ('scripts', 'uniwm')
                    for p in sorted((a.repo / folder).rglob('*.py'))},
        runner_sha256=sha(Path(__file__)), base_model=a.base_model,
        base_files={p.name: sha(p) for p in sorted(Path(a.base_model).glob('*.json'))} if Path(a.base_model).is_dir() else {},
        precision='fp16', fp32_mlp_layers='31', vq_dtype='float32', memory='none',
        sampling=dict(num_beams=1, do_sample=True, max_new_tokens=804, temperature=0.7, top_p=0.9),
        current_image='RGB resized 256 then VQ reconstructed 448; start/goal RGB resized 256',
        model_init_seed=42,
        packages={p: importlib.metadata.version(p) for p in ('torch', 'transformers', 'peft', 'accelerate', 'tokenizers')})
    if a.output.exists():
        if not a.resume:
            raise FileExistsError('Output exists; use a new directory or explicit --resume')
        assert json.loads((a.output / 'config.json').read_text()) == config, 'Resume config/code/weights mismatch'
    else:
        a.output.mkdir(parents=True)
        write_json(a.output / 'config.json', config)
        write_json(a.output / 'manifest.json', manifest)
    # Prevent two processes from writing the same run; flock releases on process exit.
    import fcntl
    lock = (a.output / 'run.lock').open('w')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    lock.write(str(os.getpid())); lock.flush()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    set_seed(42)
    args = SimpleNamespace(model='anole', model_ckpt=str(a.checkpoint), image_seq_length=784,
        precision='fp16', attn_impl='sdpa', base_model=a.base_model, use_memory_bank_inference=False,
        do_task_level_eval=False, do_train=False, do_single_step_eval=False, do_rollout_eval=False,
        do_action_probe=True, lora_r=16, lora_alpha=32, lora_dropout=.05,
        lora_targets='q_proj,k_proj,v_proj,o_proj', lm_head_mode='full',
        fp32_mlp_layers='auto', init_adapter=None, lora_fp32=True)
    print(json.dumps(dict(event='loading', gpu=torch.cuda.get_device_name(), free_bytes=torch.cuda.mem_get_info()[0])), flush=True)
    mp = load_model(args, None)
    model, processor = mp['model'], mp['processor']
    model.eval()
    token_ids = sorted(i for t, i in processor.tokenizer.get_vocab().items() if '_bin_' in t)
    rows = model.get_input_embeddings().weight[token_ids].detach().float().cpu().numpy()
    write_json(a.output / 'embedding_provenance.json', dict(action_input_rows_sha256=hashlib.sha256(rows.tobytes()).hexdigest(),
        saved_action_embedding_file=(a.checkpoint / 'action_embed.safetensors').exists(),
        limitation='Legacy baseline recreates frozen action input rows at load time; init seed and dependency versions recorded. Cross-environment equality to training rows is unverified.'))
    del rows
    device = model.device
    def read_image(path):
        with Image.open(path) as im:
            return im.convert('RGB').resize((256, 256))
    def decode(tokens):
        pixels = model.decode_image_tokens(tokens)
        array = processor.postprocess_pixel_values(pixels)[0].cpu().numpy().transpose(1, 2, 0)
        assert np.isfinite(array).all(), 'Nonfinite decoded image'
        return Image.fromarray(array.astype(np.uint8))
    def sanitize(image):
        inputs = processor(text='<image>', images=image, return_tensors='pt').to(device)
        tokens = model.model.model.get_image_tokens(inputs['pixel_values'].to(model.dtype))
        return decode(tokens)
    attempts = 0
    start = time.perf_counter()
    with torch.inference_mode():
        for state in manifest['states']:
            folder = a.output / state['state_id']; folder.mkdir(exist_ok=True)
            paths = {key: a.data / path for key, path in state['files'].items()}
            start_image, goal_image = read_image(paths['start']), read_image(paths['goal'])
            raw_current, raw_next = read_image(paths['current']), read_image(paths['next'])
            raw_current.save(folder / 'current_raw.png'); raw_next.save(folder / 'next_raw.png')
            current = sanitize(raw_current); current.save(folder / 'current_vq.png')
            sanitize(raw_next).save(folder / 'next_vq.png')
            sp = state['start_pose']
            pose_text = f'Starting Point Coordinate: x={sp[0]:.3f}, y={sp[1]:.3f}, yaw={sp[2]:.3f}\n'
            # First four outputs cover all actions for one seed, providing the smoke test.
            for repeat, seed in enumerate(state['seeds']):
                for condition in manifest['conditions']:
                    stem = folder / f'{condition}_s{repeat}'
                    record_path = stem.with_suffix('.json'); image_path = stem.with_suffix('.png')
                    if record_path.exists():
                        record = json.loads(record_path.read_text())
                        if record['status'] == 'ok':
                            assert sha(image_path) == record['image_sha256']
                            continue
                        # A previous failed attempt remains part of the denominator; don't resample it silently.
                        continue
                    action = state['actions'][condition]
                    text = action_to_text(action)
                    assert text == state['action_text'][condition]
                    prompt = build_viz_prompt(text, pose_text, memory_block='')
                    inputs = processor(text=[prompt], images=[start_image, goal_image, current],
                                       return_tensors='pt', padding=False, truncation=False).to(device)
                    ids = inputs['input_ids'][0].tolist()
                    # Token lookup must exclude the punctuation following dx/dy tokens.
                    import re
                    expected = re.findall(r'<[^>]+>', text)
                    action_tokens = [processor.tokenizer.convert_tokens_to_ids(t) for t in expected]
                    assert len(set(action_tokens)) == 3 and all(t in ids for t in action_tokens)
                    write_json(folder / f'{condition}_input.json', dict(prompt=prompt, input_ids=ids, action_token_ids=action_tokens))
                    set_seed(seed)
                    torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    record = dict(state_id=state['state_id'], source=state['source'], condition=condition,
                                  repeat=repeat, seed=seed, action=action, action_text=text, status='pending')
                    try:
                        with torch.autocast(device_type='cuda', dtype=torch.float16):
                            out = model.generate(**inputs, **config['sampling'])[0]
                        row = out[0] if out.ndim == 2 else out
                        begin = (row == model.config.boi_token_id).nonzero().flatten()
                        if begin.numel() != 1:
                            raise ValueError(f'Expected one generated image segment, found {begin.numel()}')
                        b = int(begin[0]); e = b + 785
                        if len(row) <= e or int(row[e]) != model.config.eoi_token_id:
                            raise ValueError('Incomplete image token segment')
                        predicted = decode(row[b+1:e].unsqueeze(0))
                        predicted.save(image_path)
                        record.update(status='ok', image_sha256=sha(image_path), output_size=list(predicted.size))
                        del out, row, predicted
                    except ValueError as err:
                        record.update(status='parse_failed', error=str(err))
                    torch.cuda.synchronize()
                    record.update(seconds=time.perf_counter() - t0,
                                  peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                                  peak_reserved_bytes=torch.cuda.max_memory_reserved())
                    write_json(record_path, record)
                    attempts += 1
                    print(json.dumps(record), flush=True)
                    del inputs
                    torch.cuda.empty_cache()
                    if a.max_images > 0 and attempts >= a.max_images:
                        print(json.dumps(dict(event='attempt_limit', attempts_this_run=attempts)), flush=True)
                        return
    records = [json.loads(p.read_text()) for p in a.output.glob('traj_*/*_s[0-9].json')]
    summary = dict(attempts=len(records), ok=sum(r['status'] == 'ok' for r in records), expected=240,
                   elapsed_this_process=time.perf_counter() - start)
    write_json(a.output / 'summary.json', summary)
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
