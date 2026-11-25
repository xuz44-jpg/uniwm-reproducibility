"""Ask a frozen Qwen VLM whether the camera turned left or right between two renders.

Resumable: answers are appended to results.jsonl and finished ids are skipped.
"""
import argparse, json, subprocess, time
from pathlib import Path
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText

HERE = Path(__file__).resolve().parent


def free_mib():
    out = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free',
                                   '--format=csv,noheader,nounits'], text=True, timeout=5)
    return int(out.strip().splitlines()[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--out', default=str(HERE / 'results.jsonl'))
    ap.add_argument('--min-free-mib', type=int, default=16000)
    ap.add_argument('--gpu-max-gib', type=int, default=12)
    ap.add_argument('--options-reversed', action='store_true',
                    help='post hoc control: list RIGHT before LEFT in the prompt')
    ap.add_argument('--prompt-file', help='post hoc variant: replace the prompt with this file')
    ap.add_argument('--labels', default='LEFT,RIGHT', help='answer words parsed and scored by first-token logit')
    a = ap.parse_args()

    free = free_mib()
    if free < a.min_free_mib:
        raise SystemExit(f'abort: only {free} MiB GPU memory free (< {a.min_free_mib})')

    spec = json.loads((HERE / 'queries.json').read_text())
    prompt, queries = spec['prompt'], spec['queries']
    if a.options_reversed:
        old = ('to the LEFT or to the RIGHT', 'one word: LEFT or RIGHT.')
        assert all(o in prompt for o in old)
        prompt = prompt.replace(old[0], 'to the RIGHT or to the LEFT').replace(old[1], 'one word: RIGHT or LEFT.')
    if a.prompt_file:
        prompt = Path(a.prompt_file).read_text().strip()
    labels = a.labels.split(',')
    print('prompt:', prompt, flush=True)
    out = Path(a.out)
    done = set()
    if out.exists():
        done = {json.loads(l)['id'] for l in out.read_text().splitlines() if l.strip()}

    proc = AutoProcessor.from_pretrained(a.model)
    model = AutoModelForImageTextToText.from_pretrained(
        a.model, dtype=torch.bfloat16, device_map='auto', attn_implementation='sdpa',
        max_memory={0: f'{a.gpu_max_gib}GiB', 'cpu': '80GiB'}).eval()
    print('device map:', sorted({str(v) for v in model.hf_device_map.values()}), flush=True)
    tok = proc.tokenizer
    label_ids = {w: tok.encode(w, add_special_tokens=False)[0] for w in labels}
    assert len(set(label_ids.values())) == len(labels), label_ids

    with out.open('a') as fo:
        for q in queries:
            if q['id'] in done:
                continue
            t0 = time.time()
            msgs = [{'role': 'user', 'content': [
                {'type': 'text', 'text': 'Image 1:'},
                {'type': 'image', 'image': str(HERE / q['image1'])},
                {'type': 'text', 'text': 'Image 2:'},
                {'type': 'image', 'image': str(HERE / q['image2'])},
                {'type': 'text', 'text': prompt}]}]
            inputs = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                              return_dict=True, return_tensors='pt').to('cuda:0')
            with torch.inference_mode():
                g = model.generate(**inputs, max_new_tokens=8, do_sample=False,
                                   output_scores=True, return_dict_in_generate=True)
            text = proc.batch_decode(g.sequences[:, inputs['input_ids'].shape[1]:],
                                     skip_special_tokens=True)[0].strip()
            s0 = g.scores[0][0].float()
            word = text.upper().strip(' .!"\'')
            parsed = next((w for w in labels if word.startswith(w)), None)
            logits = {w: float(s0[i]) for w, i in label_ids.items()}
            rec = dict(id=q['id'], text=text, parsed=parsed, logits=logits,
                       logit_left=logits['LEFT'], logit_right=logits['RIGHT'],
                       n_input_tokens=int(inputs['input_ids'].shape[1]),
                       seconds=round(time.time() - t0, 3))
            fo.write(json.dumps(rec) + '\n'); fo.flush()
    print('done', sum(1 for _ in out.open()))


if __name__ == '__main__':
    main()
