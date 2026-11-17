"""Exploratory LPIPS paired controls + RAFT calibration. No offline counterfactual GT claim."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--device', default='cuda')
    a = ap.parse_args()
    import lpips
    from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
    manifest = json.loads((a.run / 'manifest.json').read_text())
    output = a.run / 'analysis'
    output.mkdir(exist_ok=False)
    device = torch.device(a.device)
    lp = lpips.LPIPS(net='alex', verbose=False).to(device).eval()
    weights = Raft_Large_Weights.C_T_SKHT_V2
    flow_model = raft_large(weights=weights, progress=True).to(device).eval()
    def image(path):
        with Image.open(path) as im:
            ar = np.array(im.convert('RGB').resize((256, 256)), copy=True)
        return torch.from_numpy(ar).permute(2, 0, 1).unsqueeze(0).to(device).float() / 127.5 - 1
    def flow(x, y):
        return flow_model(x, y, num_flow_updates=12)[-1][0, :, 40:-40, 40:-40]
    def scalar_flow(f):
        return dict(median_x=float(f[0].median()), median_y=float(f[1].median()),
                    median_magnitude=float(f.square().sum(0).sqrt().median()))
    rows = []
    with torch.inference_mode():
        for state in manifest['states']:
            folder = a.run / state['state_id']
            raw, vq = image(folder / 'current_raw.png'), image(folder / 'current_vq.png')
            nr, nv = image(folder / 'next_raw.png'), image(folder / 'next_vq.png')
            calibration = []
            for domain, x in [('raw', raw), ('vq', vq)]:
                for dx in [0, 1, 2, 4, 8, 16, 32, -8]:
                    shifted = (F.pad(x, (dx, 0, 0, 0), mode='replicate')[:, :, :, :256] if dx > 0
                               else F.pad(x, (0, -dx, 0, 0), mode='replicate')[:, :, :, -dx:])
                    f = flow(x, shifted)
                    target = torch.zeros_like(f); target[0] = dx
                    calibration.append(dict(domain=domain, shift_px=dx, **scalar_flow(f),
                        median_epe=float((f-target).square().sum(0).sqrt().median())))
            refs = {'raw': flow(raw, nr), 'mixed': flow(vq, nr), 'vq': flow(vq, nv)}
            paths = [(c, r, folder / f'{c}_s{r}.png') for c in manifest['conditions'] for r in range(3)]
            valid = []
            for c, r, path in paths:
                record_path = folder / f'{c}_s{r}.json'
                if not record_path.exists():
                    continue
                rec = json.loads(record_path.read_text())
                if rec['status'] == 'ok':
                    assert hashlib.sha256(path.read_bytes()).hexdigest() == rec['image_sha256']
                    valid.append((c, r, image(path)))
            distances = {}
            for i, j in itertools.combinations(range(len(valid)), 2):
                distances[(i, j)] = float(lp(valid[i][2], valid[j][2]).item())
            def dist(i, j):
                return distances[tuple(sorted((i, j)))]
            contrasts = {}
            for other in manifest['conditions'][1:]:
                orig = [i for i, (c, _, _) in enumerate(valid) if c == 'original']
                alt = [i for i, (c, _, _) in enumerate(valid) if c == other]
                # A state enters a contrast only with all six images; missing states remain in coverage.
                if len(orig) != 3 or len(alt) != 3:
                    continue
                within = np.mean([dist(i, j) for inds in [orig, alt] for i, j in itertools.combinations(inds, 2)])
                between = np.mean([dist(i, j) for i in orig for j in alt if valid[i][1] != valid[j][1]])
                paired = np.mean([dist(i, j) for i in orig for j in alt if valid[i][1] == valid[j][1]])
                contrasts[other] = dict(within=float(within), between_independent=float(between),
                    same_seed_response=float(paired), effect=float(between-within))
            generated_motion = []
            for c, r, im in valid:
                f = flow(vq, im)
                item = dict(condition=c, repeat=r, **scalar_flow(f))
                if c == 'original':
                    item['estimated_flow_epe_mixed_ref'] = float((f-refs['mixed']).square().sum(0).sqrt().mean())
                    item['estimated_flow_epe_vq_ref'] = float((f-refs['vq']).square().sum(0).sqrt().mean())
                generated_motion.append(item)
            row = dict(state_id=state['state_id'], source=state['source'], generated_ok=len(valid),
                       calibration=calibration, reference_flow={k: scalar_flow(v) for k, v in refs.items()},
                       contrasts=contrasts, generated_motion=generated_motion)
            (output / f'{state["state_id"]}.json').write_text(json.dumps(row, indent=2))
            rows.append(row)
            print(json.dumps(dict(state_id=state['state_id'], generated_ok=len(valid), contrasts=contrasts)), flush=True)
    rng = np.random.default_rng(20261021)
    contrasts = {}
    for c in manifest['conditions'][1:]:
        values = np.array([r['contrasts'][c]['effect'] for r in rows if c in r['contrasts']])
        if not len(values):
            continue
        # One state per source group: resampling these states is source-group bootstrap.
        bootstrap = rng.choice(values, (10000, len(values)), replace=True).mean(1)
        contrasts[c] = dict(sources=len(values), mean_effect=float(values.mean()),
                            exploratory_ci95=np.quantile(bootstrap, [.025, .975]).tolist())
    summary = dict(purpose='Development diagnostics, not a confirmatory hypothesis test',
        motion_estimator='torchvision RAFT Large C_T_SKHT_V2; 12 updates; 256x256; central 176x176 region',
        reference='Estimated flow, not physical ground truth; no alternate-action real frames available',
        direction_accuracy='Not reported: yaw-to-image geometry and timing not validated',
        measurable_subset='Not defined yet: use calibration to freeze later protocol, never generated success',
        expected_images=240, actual_images=sum(r['generated_ok'] for r in rows),
        sources=len(rows), lpips='AlexNet at 256x256', primary_contrast='original vs yaw_flip', contrasts=contrasts)
    (output / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
