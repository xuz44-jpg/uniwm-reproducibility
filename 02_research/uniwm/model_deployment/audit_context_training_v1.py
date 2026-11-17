"""Read-only, CPU-only audit of existing Habitat adaptation metadata.

No model is imported, no checkpoint tensor is loaded, and no training or image
generation runs.  Only the explicitly supplied output directory is written.
PNG equality means equality of file SHA-256, not estimated camera geometry.
"""
import argparse
import ast
from collections import Counter
import hashlib
import json
import math
from pathlib import Path


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def fraction(count, known, total):
    return dict(count=count, known=known, unknown=total-known, total=total,
                fraction_of_known=count/known if known else None,
                fraction_of_all=count/total if known == total and total else None)


class Inputs:
    def __init__(self, study, source):
        self.study = Path(study)
        self.v2 = Path(source) if source else self.study/'v2'
        if (self.v2/'v2').is_dir():
            self.v2 = self.v2/'v2'
        self.local_study = self.v2.parent
        self.files = {}
        self.missing = set()
        self.errors = []
        self.recorded_image_hashes = {}
        self.image_hash_cache = {}
        self.image_evidence = {}

    def resolve(self, path):
        path = Path(path)
        try:
            relative = path.relative_to(self.study)
        except ValueError:
            return path
        return self.local_study/relative

    def read(self, path, required=True):
        path = Path(path)
        if not path.is_file():
            if required:
                self.missing.add(str(path))
            return None
        self.files[str(path)] = sha256(path)
        try:
            return json.loads(path.read_text())
        except (ValueError, OSError) as error:
            self.errors.append(dict(path=str(path), error=str(error)))
            return None

    def source(self, path):
        path = Path(path)
        if not path.is_file():
            self.missing.add(str(path))
            return None
        self.files[str(path)] = sha256(path)
        try:
            return ast.parse(path.read_text(), filename=str(path))
        except (SyntaxError, OSError) as error:
            self.errors.append(dict(path=str(path), error=str(error)))
            return None

    def add_hashes(self, mapping, source):
        if not isinstance(mapping, dict):
            return
        for path, digest in mapping.items():
            if not isinstance(digest, str) or len(digest) != 64:
                self.errors.append(dict(path=str(source), error='invalid_image_hash', image=path))
                continue
            old = self.recorded_image_hashes.get(path)
            if old and old != digest:
                self.errors.append(dict(path=str(source), error='conflicting_image_hash', image=path))
            self.recorded_image_hashes[path] = digest

    def image_hash(self, recorded_path):
        if not isinstance(recorded_path, str):
            return None
        if recorded_path in self.image_hash_cache:
            return self.image_hash_cache[recorded_path]
        path = self.resolve(recorded_path)
        recorded = self.recorded_image_hashes.get(recorded_path)
        if path.is_file():
            actual = sha256(path)
            self.files[str(path)] = actual
            if recorded and actual != recorded:
                self.errors.append(dict(path=str(path), error='image_hash_mismatch',
                                        recorded_sha256=recorded, actual_sha256=actual))
            result = actual
            evidence = 'actual_file_sha256'
        elif recorded:
            result = recorded
            evidence = 'recorded_sha256_only_file_unavailable'
            self.missing.add(str(path))
        else:
            result = None
            evidence = 'missing_file_and_recorded_sha256'
            self.missing.add(str(path))
        self.image_hash_cache[recorded_path] = result
        self.image_evidence[recorded_path] = dict(resolved_path=str(path),
            sha256=result, evidence=evidence)
        return result


def source_scope(inputs, path):
    """Parse declarations as evidence; never execute the audited Python source."""
    tree = inputs.source(path)
    if tree is None:
        return dict(status='missing_or_invalid', source=str(path))
    filtered = []
    frozen_calls = []
    counts = []
    literals = {}
    known_options = {'lora_r', 'lora_alpha', 'lora_dropout', 'lora_targets',
                     'lm_head_mode', 'base_model', 'init_adapter'}
    for node in ast.walk(tree):
        if isinstance(node, (ast.DictComp, ast.ListComp, ast.GeneratorExp)):
            named = any(isinstance(x, ast.Call) and isinstance(x.func, ast.Attribute)
                        and x.func.attr == 'named_parameters' for x in ast.walk(node))
            requires = any(isinstance(x, ast.Attribute) and x.attr == 'requires_grad'
                           for gen in node.generators for cond in gen.ifs for x in ast.walk(cond))
            if named and requires:
                filtered.append(dict(line=node.lineno, expression=ast.unparse(node)))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == 'requires_grad_' and node.args:
                try:
                    frozen = ast.literal_eval(node.args[0]) is False
                except (ValueError, TypeError):
                    frozen = False
                if frozen:
                    frozen_calls.append(dict(line=node.lineno, expression=ast.unparse(node)))
        if isinstance(node, ast.Assert) and isinstance(node.test, ast.Compare):
            left = node.test.left
            if (isinstance(left, ast.Call) and isinstance(left.func, ast.Name)
                    and left.func.id == 'len' and len(left.args) == 1
                    and isinstance(left.args[0], ast.Name) and left.args[0].id == 'params'):
                for value in node.test.comparators:
                    if isinstance(value, ast.Constant) and isinstance(value.value, int):
                        counts.append(dict(line=node.lineno, trainable_tensors=value.value))
        if isinstance(node, ast.keyword) and node.arg in known_options:
            try:
                value = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                value = dict(expression=ast.unparse(node.value))
            literals[node.arg] = value
    return dict(status='parsed_not_executed', source=str(path),
        sha256=inputs.files[str(Path(path))],
        requires_grad_filtered_named_parameters=filtered,
        frozen_module_calls=frozen_calls, trainable_tensor_assertions=counts,
        configuration_literals=literals,
        limitation='Static declarations do not prove every tensor received a nonzero update.')


def row_statistics(rows, inputs, split, model):
    if rows is None:
        return dict(status='missing', split=split)
    if not isinstance(rows, list):
        return dict(status='invalid_rows', split=split)
    counts = Counter()
    histories_paths, histories_hashes = Counter(), Counter()
    relations = {k: [0, 0] for k in ['target_equals_goal', 'target_equals_start',
        'target_equals_current', 'current_equals_goal', 'start_equals_goal',
        'target_in_uniwm_inputs', 'target_in_recon_history', 'target_in_any_history']}
    yaw_values = Counter()
    horizon = Counter()
    trajectory_ids, source_ids, rooms, scene_ids = set(), set(), set(), set()
    sample_ids = []
    invalid_rows = []
    examples = {k: [] for k in relations}
    action_groups = {}
    for index, row in enumerate(rows):
        sample = row.get('sample_id', str(index))
        sample_ids.append(sample)
        trajectory_ids.add(str(row.get('trajectory_id', 'missing')))
        source_ids.add(str(row.get('source_trajectory_id', row.get('trajectory_id', 'missing'))))
        if 'room_id' in row:
            rooms.add(str(row['room_id']))
        if row.get('scene_id') is not None:
            scene_ids.add(str(row['scene_id']))
        if row.get('split') != split:
            counts['split_field_mismatch'] += 1
        history = row.get('history_paths', [])
        histories_paths[str(len(set(history)))] += 1
        counts['history_length_'+str(len(history))] += 1
        hs = [inputs.image_hash(path) for path in history]
        if all(x is not None for x in hs):
            histories_hashes[str(len(set(hs)))] += 1
        else:
            histories_hashes['unknown'] += 1
        hashes = {key: inputs.image_hash(row.get(key)) for key in
                  ['start_image', 'goal_image', 'current_image', 'target_image']}
        target = hashes['target_image']
        values = {}
        for name, a, b in [('target_equals_goal', 'target_image', 'goal_image'),
                            ('target_equals_start', 'target_image', 'start_image'),
                            ('target_equals_current', 'target_image', 'current_image'),
                            ('current_equals_goal', 'current_image', 'goal_image'),
                            ('start_equals_goal', 'start_image', 'goal_image')]:
            values[name] = (hashes[a] == hashes[b]) if hashes[a] and hashes[b] else None
        uni = [hashes[k] for k in ['start_image', 'goal_image', 'current_image']]
        values['target_in_uniwm_inputs'] = (target in uni) if target and all(uni) else None
        values['target_in_recon_history'] = (target in hs) if target and all(hs) and hs else None
        values['target_in_any_history'] = values['target_in_recon_history']
        for name, value in values.items():
            if value is not None:
                relations[name][1] += 1
                relations[name][0] += int(value)
                if value and len(examples[name]) < 5:
                    examples[name].append(sample)
        if history and row.get('current_image') == history[-1]:
            counts['last_history_path_is_current'] += 1
        h = row.get('horizon_frames')
        horizon[str(h)] += 1
        if (isinstance(row.get('target_index'), (int, float))
                and isinstance(row.get('current_index'), (int, float))):
            if row['target_index'] <= row['current_index']:
                counts['target_index_not_future'] += 1
            if h is not None and row['target_index'] - row['current_index'] != h:
                counts['index_horizon_mismatch'] += 1
        action = row.get('action')
        if (not isinstance(action, list) or len(action) != 3
                or not all(isinstance(x, (int, float)) and math.isfinite(x) for x in action)):
            invalid_rows.append(dict(sample_id=sample, reason='missing_or_invalid_action'))
            continue
        dx, dy, yaw = action
        translation = math.hypot(dx, dy)
        sign = 'left' if yaw > 1e-6 else 'right' if yaw < -1e-6 else 'zero_yaw'
        counts['all_actions_'+sign] += 1
        if translation <= 1e-5:
            counts['pure_turn_'+sign] += 1
            if sign != 'zero_yaw':
                yaw_values[f'{math.degrees(yaw):+.6f}'] += 1
        else:
            counts['translation_nonzero'] += 1
        if translation <= 1e-5 and abs(yaw) <= 1e-6:
            counts['zero_action'] += 1
        group = ('pure_'+sign if translation <= 1e-5 else 'translation_'+sign)
        g = action_groups.setdefault(group, dict(rows=0, target_equals_goal=0,
            target_equals_goal_known=0, target_in_actual_inputs=0,
            target_in_actual_inputs_known=0))
        g['rows'] += 1
        for name, key in [('target_equals_goal', 'target_equals_goal'),
                ('target_in_uniwm_inputs' if model == 'uniwm' else 'target_in_recon_history',
                 'target_in_actual_inputs')]:
            if values[name] is not None:
                g[key+'_known'] += 1
                g[key] += int(values[name])
        if row.get('translation_constrained') is True:
            counts['translation_constrained'] += 1
    return dict(status='passed' if not invalid_rows else 'invalid_rows', model=model, split=split,
        rows=len(rows), expected_rows=336 if split == 'train' else 42,
        expected_count_matches=len(rows) == (336 if split == 'train' else 42),
        trajectory_count=len(trajectory_ids), trajectory_ids=sorted(trajectory_ids),
        source_start_group_count=len(source_ids), room_ids=sorted(rooms),
        explicit_row_scene_ids=sorted(scene_ids), counters=dict(sorted(counts.items())),
        pure_turn_definition='translation_norm <= 1e-5 native meters and abs(yaw_rad) > 1e-6',
        pure_turn_angle_deg_histogram=dict(sorted(yaw_values.items())),
        history_unique_paths_histogram=dict(sorted(histories_paths.items())),
        history_unique_png_sha256_histogram=dict(sorted(histories_hashes.items())),
        horizon_frames_histogram=dict(horizon),
        png_hash_relations={k: fraction(n, known, len(rows)) for k, (n, known) in relations.items()},
        png_hash_match_examples=examples, invalid_rows=invalid_rows,
        conditions_by_action_group=action_groups,
        sample_ids_unique=len(set(sample_ids)) == len(sample_ids),
        actual_model_inputs=['start_image', 'goal_image', 'current_image'] if model == 'uniwm'
                            else ['history_paths[0:4]'],
        goal_is_input=model == 'uniwm',
        hash_equality_interpretation='Exact PNG file identity; repeats and valid returns are not automatically future leakage.')


def build_audit(study, source=None):
    inputs = Inputs(study, source)
    scripts = Path(__file__).resolve().parent
    inputs.files[str(Path(__file__).resolve())] = sha256(__file__)
    plan = inputs.read(inputs.local_study/'plan.json')
    collection_path = inputs.local_study/'collection_v1/dataset.json'
    collection = inputs.read(collection_path)
    scene_files = [p for p in (plan or {}).get('source_hashes', {}) if p.endswith('.glb')]
    lineage_matches = None
    if collection and plan:
        lineage_matches = collection.get('plan_sha256') == inputs.files[str(inputs.local_study/'plan.json')]
    dataset = dict(scene_count_recorded=(collection or {}).get('scene_count'),
        scene_ids_from_plan_glb=sorted(Path(p).stem for p in scene_files),
        scene_count_from_plan_glb=len(scene_files) if scene_files else None,
        collection_plan_hash_matches=lineage_matches,
        evidence='Collection metadata plus source asset identities; missing fields remain unknown.',
        base_pretraining_scene_coverage='unknown')
    runs = {m: inputs.v2/('uniwm_symmetric_aligned_v3' if m == 'uniwm' else 'recon_symmetric_v3')
            for m in ['uniwm', 'nwm', 'raenwm']}
    for run in sorted(set(runs.values())):
        path = run/'input_image_hashes.json'
        inputs.add_hashes(inputs.read(path), path)
    models = {}
    raw_rows = {}
    for model, run in runs.items():
        protocol = inputs.read(run/'protocol.json')
        data_complete = inputs.read(run/'data_complete.json')
        training = run/'training' if model == 'uniwm' else run/'training'/model
        config = inputs.read(training/'training_config.json')
        completed = inputs.read(training/'complete.json')
        config, completed = config or {}, completed or {}
        splits = {}
        raw_rows[model] = {}
        for split in ['train', 'validation']:
            rows_path = run/(split+'.json')
            rows = inputs.read(rows_path)
            raw_rows[model][split] = rows
            splits[split] = row_statistics(rows, inputs, split, model)
            declared = config.get('data_rows_sha256', {}).get(split)
            if declared is None and data_complete:
                declared = data_complete.get(split+'_sha256') or data_complete.get('rows_sha256', {}).get(split)
            splits[split]['rows_sha256_expected'] = declared
            splits[split]['rows_sha256_actual'] = inputs.files.get(str(rows_path))
            splits[split]['rows_sha256_matches_recorded'] = (
                declared == inputs.files.get(str(rows_path)) if declared else None)
            if declared and declared != inputs.files.get(str(rows_path)):
                inputs.errors.append(dict(path=str(rows_path), error='row_manifest_hash_mismatch',
                    recorded_sha256=declared, actual_sha256=inputs.files.get(str(rows_path))))
        source_script = scripts/('train_uniwm_symmetric_v3.py' if model == 'uniwm'
                                  else 'train_recon_symmetric_v3.py')
        scope = source_scope(inputs, source_script)
        scope['matches_recorded_training_script_sha256'] = (
            scope.get('sha256') == config.get('script_sha256') if config.get('script_sha256') else None)
        if model == 'uniwm':
            evidence_path = run/'loading_action_audit_v1/cpu_complete.json'
            evidence = inputs.read(evidence_path)
            runtime_count = (evidence or {}).get('exported_trainables_checked')
            runtime_hash = (evidence or {}).get('new_adapter_sha256')
            scope.update(update_scope='LoRA attention projections and full image output head; frozen base/codec per training configuration',
                exported_trainable_tensor_count_recorded=runtime_count,
                export_sha256_matches_completion=runtime_hash == completed.get('adapter_sha256') if runtime_hash else None,
                runtime_evidence_path=str(evidence_path))
            checkpoint = training/'final_adapter/adapter_model.safetensors'
            selected_sha = completed.get('adapter_sha256')
            parity = inputs.read(run/'preprocessing/parity.json')
        else:
            evidence_path = run/'evaluation'/model/'loaded_parameters.json'
            evidence = inputs.read(evidence_path)
            runtime_count = (evidence or {}).get('trainable_tensors_verified')
            scope.update(update_scope='All native generator parameters with requires_grad=True; codec frozen',
                loaded_trainable_tensor_count_recorded=runtime_count,
                loaded_all_tensors_match_recorded=(evidence or {}).get('all_tensors_match'),
                runtime_evidence_path=str(evidence_path),
                action_and_relative_time_modules='Part of native generator; static named_parameters optimizer scope, not per-tensor update proof')
            checkpoint = training/'best_trainable.pt'
            selected_sha = completed.get('checkpoint_sha256')
            parity = None
        identity = dict(initialization=config.get('baseline') or completed.get('baseline') or completed.get('initialization'),
            initialization_sha256=config.get('baseline_sha256') or config.get('baseline_hashes'),
            selected_checkpoint_path=str(checkpoint), selected_checkpoint_sha256_recorded=selected_sha,
            selected_step=completed.get('step'), optimizer_steps_completed=completed.get('optimizer_steps_completed'),
            checkpoint_file_present=checkpoint.is_file(), checkpoint_bytes=checkpoint.stat().st_size if checkpoint.is_file() else None,
            checkpoint_content_loaded=False, checkpoint_content_rehashed=False,
            identity_limitation='Weight identity is taken from existing metadata; giant tensor files are not loaded or rehashed.')
        train_ids = {r.get('sample_id') for r in raw_rows[model]['train'] or []}
        val_ids = {r.get('sample_id') for r in raw_rows[model]['validation'] or []}
        models[model] = dict(identity=identity, parameter_scope=scope, splits=splits,
            train_validation_sample_id_overlap=sorted(x for x in train_ids & val_ids if x is not None),
            native_prefix_parity_recorded=parity,
            test_used_for_selection_recorded=completed.get('test_used_for_selection'),
            training_config=config, protocol_version=(protocol or {}).get('version'),
            goal_condition_interpretation='start=goal is present throughout these adaptation rows; changing only inference goal can alter training support'
                if model == 'uniwm' else 'Row goal fields are not used by this model; generator receives four history frames, endpoint action and frame-offset condition')
    same_corpus = {}
    keys = ['sample_id', 'history_paths', 'start_image', 'goal_image', 'current_image',
            'target_image', 'action', 'current_index', 'target_index', 'horizon_frames']
    for split in ['train', 'validation']:
        a, b = raw_rows['uniwm'][split], raw_rows['nwm'][split]
        same_corpus[split] = (len(a) == len(b) and
            all(all(x.get(k) == y.get(k) for k in keys) for x, y in zip(a, b))) if a is not None and b is not None else None
    limitations = [
        'One adaptation scene and correlated repeated views do not establish sufficient training or model-wide capability.',
        'PNG target/input matches can be legitimate zero motion or return-to-start; they diagnose copying opportunities, not automatically label leakage.',
        'Changing only inference history or goal is a condition intervention and may change training support.',
        'Native rel_t is endpoint frame offset divided by 128, not seconds; simulator steps are not calibrated hardware time.',
        'Generator parameter scopes are source and prior metadata evidence; no new forward pass, gradient or checkpoint tensor inspection occurs.',
        'UniWM initialization is the existing 1X-200 adaptation, not a verified official navigation checkpoint.',
        'Missing metadata, images or checkpoint files remain explicitly unavailable; prior recorded SHA values are distinguished from freshly checked files.',
    ]
    missing_images = sum(e['evidence'] != 'actual_file_sha256' for e in inputs.image_evidence.values())
    if lineage_matches is False:
        inputs.errors.append(dict(path=str(collection_path), error='collection_plan_hash_mismatch'))
    # Source evidence and metadata failures must never silently become a full pass.
    checks = [x['expected_count_matches'] and x['sample_ids_unique']
              for m in models.values() for x in m['splits'].values() if 'expected_count_matches' in x]
    status = 'failed' if inputs.errors or not all(checks) else 'partial' if inputs.missing else 'passed'
    report = dict(schema='context_training_audit_v1', status=status,
        study=str(study), source_v2=str(inputs.v2),
        execution=dict(device='cpu', imported_world_models=False, tensor_checkpoints_loaded=False,
                       optimizer_updates=0, new_world_model_predictions=0, source_data_modified=False),
        dataset=dataset, models=models, same_underlying_corpus=same_corpus,
        image_hash_evidence=dict(unique_paths=len(inputs.image_evidence), fresh_file_hashes=len(inputs.image_evidence)-missing_images,
                                 recorded_only_or_missing=missing_images, equality='SHA-256 of PNG file bytes'),
        missing=sorted(inputs.missing), errors=inputs.errors, limitations=limitations)
    return report, inputs


def markdown_report(report):
    lines = ['# Existing adaptation training/context audit', '',
             f"Status: **{report['status']}**. CPU metadata/image-hash inspection; no training or model inference.", '',
             f"Recorded adaptation scene count: {report['dataset']['scene_count_recorded']}; "
             f"scene identities from plan: {report['dataset']['scene_ids_from_plan_glb']}.", '',
             '| Model | Selected step | Actual updates | Train / val rows |',
             '|---|---:|---:|---:|']
    for name, model in report['models'].items():
        identity = model['identity']
        lines.append(f"| {name} | {identity['selected_step']} | {identity['optimizer_steps_completed']} | "
                     f"{model['splits']['train'].get('rows', 'missing')} / {model['splits']['validation'].get('rows', 'missing')} |")
    for name, model in report['models'].items():
        lines.extend(['', f'## {name}', '', f"Initialization: `{model['identity']['initialization']}`.", '',
                      model['parameter_scope'].get('update_scope', 'Parameter scope unavailable.'), '',
                      model['goal_condition_interpretation'], ''])
        for split, stats in model['splits'].items():
            if stats.get('status') == 'missing':
                lines.append(f'- {split}: missing.'); continue
            relations = stats['png_hash_relations']
            target_goal = relations['target_equals_goal']
            native = relations['target_in_uniwm_inputs' if name == 'uniwm' else 'target_in_recon_history']
            lines.extend([
                f"- {split}: {stats['rows']} rows; pure left/right "
                f"{stats['counters'].get('pure_turn_left', 0)}/{stats['counters'].get('pure_turn_right', 0)}.",
                f"- {split} target=goal PNG hash: {target_goal['count']}/{target_goal['known']} known, "
                f"{target_goal['unknown']} unknown, total {target_goal['total']}. "
                + ('Goal is an actual input.' if name == 'uniwm' else 'Goal is not a model input.'),
                f"- {split} target appears among actual image inputs: {native['count']}/{native['known']} known; "
                f"{native['unknown']} unknown, total {native['total']}.",
                f"- {split} history unique paths: `{stats['history_unique_paths_histogram']}`; "
                f"unique PNG hashes: `{stats['history_unique_png_sha256_histogram']}`.",
            ])
    lines.extend(['', '## Interpretation limits', ''])
    lines.extend('- '+item for item in report['limitations'])
    lines.extend(['', f"Missing paths: {len(report['missing'])}; explicit errors: {len(report['errors'])}.", '',
                  'Full fractions, source/config hashes, actual image hash evidence and missing paths are in the JSON files.', ''])
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, default=Path('external/habitat-yaw-reference'))
    parser.add_argument('--source', type=Path,
                        help='Optional local v2 directory (or study root) containing the retrieved metadata/images.')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    report, inputs = build_audit(args.study, args.source)
    args.out.mkdir(parents=True, exist_ok=True)
    write_json(args.out/'summary.json', report)
    write_json(args.out/'input_file_hashes.json', dict(files=inputs.files,
        images=inputs.image_evidence, checkpoint_tensors_read=False))
    (args.out/'REPORT.md').write_text(markdown_report(report))
    write_json(args.out/'complete.json', dict(status=report['status'], optimizer_updates=0,
        new_world_model_predictions=0, errors=len(report['errors']), missing=len(report['missing']),
        outputs_sha256={name: sha256(args.out/name) for name in
                        ['summary.json', 'input_file_hashes.json', 'REPORT.md']}))
    print(json.dumps(dict(status=report['status'], out=str(args.out),
                         models=list(report['models']), errors=len(report['errors']), missing=len(report['missing']))))
    if report['status'] == 'failed':
        raise SystemExit(2)


if __name__ == '__main__':
    main()
