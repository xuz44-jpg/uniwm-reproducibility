#!/usr/bin/env python3
"""Reproduce a support-set audit from frozen, already computed JSON rows.

Standard library only. Does not import executable experiment code, load a
model, estimate flow, contact a server, or modify inputs. Output metrics are
descriptive reaggregations of stored scalars, not a new pixel-level evaluation.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any


LABELS = ("correct", "reverse", "near_zero", "unknown")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path) -> Any:
    return json.loads(path.read_text())


def finite(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def quantile(xs: list[float], p: float) -> float | None:
    if not xs:
        return None
    s = sorted(xs)
    q = (len(s) - 1) * p
    lo = math.floor(q)
    return float(s[lo] + (s[math.ceil(q)] - s[lo]) * (q - lo))


def distribution(values: list[Any]) -> dict:
    xs = [float(x) for x in values if finite(x)]
    return dict(n=len(xs), mean=statistics.mean(xs) if xs else None,
                median=statistics.median(xs) if xs else None,
                minimum=min(xs) if xs else None, maximum=max(xs) if xs else None,
                p05=quantile(xs, .05), p25=quantile(xs, .25),
                p75=quantile(xs, .75), p95=quantile(xs, .95))


def csv_write(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, sort_keys=True) if isinstance(v, (list, dict)) else v
                             for k, v in row.items()})


def flatten(value: dict, prefix: str = "") -> dict:
    result = {}
    for k, v in value.items():
        key = prefix + k
        if isinstance(v, dict):
            result.update(flatten(v, key + "."))
        else:
            result[key] = v
    return result


def normalize(rows: list[dict], dataset: str, model: str, input_path: str) -> list[dict]:
    out = []
    context = dataset == "context_nwm"
    for r in rows:
        angle = r["angle"]
        direction = r.get("direction" if context else "action_direction", {}) if angle else {}
        effect = r.get("effect" if context else "action_effect", {}) if angle else {}
        valid = direction.get("response", {})
        out.append(dict(dataset=dataset, model=model, key=r["key"],
                        input_path=input_path, state_id=r["state_id"], scene_id=r["scene_id"],
                        weight_arm=r["weight_arm"] if context else "adapted",
                        context_arm=r["context_arm"] if context else "original_static",
                        repeat=r.get("repeat", 0), angle=angle, abs_angle=abs(angle),
                        status=r["status"], reference_gate=r.get("reference_gate", r.get("reference_gate_passed")),
                        label=direction.get("label"), reason=direction.get("reason"),
                        coverage=direction.get("coverage"), valid_pixels=direction.get("valid_pixels"),
                        reference_pixels=direction.get("reference_pixels"),
                        raw_epe_px=effect.get("mean_epe_px"), raw_gain=effect.get("gain_projection"),
                        raw_pixels=effect.get("pixels"),
                        raw_motion_median_px=effect.get("predicted", {}).get("median_magnitude"),
                        valid_epe_px=valid.get("mean_epe_px"), valid_gain=valid.get("gain_projection"),
                        valid_motion_median_px=direction.get("motion", {}).get("median_magnitude"),
                        valid_direction_consensus=valid.get("direction_consensus"),
                        zero_drift_raw_px=r.get("zero_drift", {}).get("mean_magnitude"),
                        zero_single_branch_coverage=r.get("generated_fb_coverage") if not angle else None,
                        direction=direction))
    assert len({x["key"] for x in out}) == len(out), "Duplicate query keys"
    return out


def gain_distribution(rows: list[dict], field: str) -> dict:
    xs = [x[field] for x in rows if finite(x[field])]
    result = distribution(xs)
    result.update(negative=sum(x < 0 for x in xs),
                  absolute_le_0_1=sum(abs(x) <= .1 for x in xs),
                  between_0_5_and_1_5=sum(.5 <= x <= 1.5 for x in xs))
    return result


def summarize(rows: list[dict]) -> dict:
    nz = [x for x in rows if x["angle"]]
    raw = [x for x in nz if x["reference_gate"] and finite(x["raw_epe_px"])]
    valid = [x for x in nz if finite(x["valid_epe_px"])]
    unknown = [x for x in nz if x["label"] == "unknown"]
    raw_unknown = [x for x in raw if x["label"] == "unknown"]
    raw_known = [x for x in raw if x["label"] != "unknown"]
    zero = [x for x in rows if not x["angle"]]
    all_sum = sum(x["raw_epe_px"] for x in raw)
    unknown_sum = sum(x["raw_epe_px"] for x in raw_unknown)
    cov = [x["coverage"] for x in nz if finite(x["coverage"])]
    return dict(
        row_appearances=len(rows), nonzero_queries=len(nz), zero_queries=len(zero),
        scenes=len({x["scene_id"] for x in rows}), positions=len({x["state_id"] for x in rows}),
        query_repeats=sorted({x["repeat"] for x in rows}),
        reference_gate_passed_nonzero=sum(bool(x["reference_gate"]) for x in nz),
        direction={k: sum(x["label"] == k for x in nz) for k in LABELS},
        unknown_reasons=dict(sorted(Counter(x["reason"] for x in unknown).items())),
        coverage={**distribution(cov), "below_0_2": sum(x < .2 for x in cov),
                  "below_0_5": sum(x < .5 for x in cov), "below_0_8": sum(x < .8 for x in cov)},
        raw_fixed_M_epe_px=distribution([x["raw_epe_px"] for x in raw]),
        raw_fixed_M_gain=gain_distribution(raw, "raw_gain"),
        raw_fixed_M_median_motion_px=distribution([x["raw_motion_median_px"] for x in raw]),
        stored_valid_region_epe_px=distribution([x["valid_epe_px"] for x in valid]),
        stored_valid_region_gain=gain_distribution(valid, "valid_gain"),
        stored_valid_region_motion_median_px=distribution([x["valid_motion_median_px"] for x in valid]),
        valid_region_query_note="Query set and per-query generated-valid pixels differ across arms; not a common-support comparison.",
        unknown_contribution=dict(
            raw_unknown=distribution([x["raw_epe_px"] for x in raw_unknown]),
            raw_nonunknown=distribution([x["raw_epe_px"] for x in raw_known]),
            sum_epe_unknown=unknown_sum, sum_epe_all=all_sum,
            fraction_of_raw_epe_sum=unknown_sum / all_sum if all_sum else None,
            additive_component_of_all_query_mean_px=unknown_sum / len(raw) if raw else None,
            definition="Sum of unknown per-query raw EPE divided by all raw-query count; descriptive arithmetic, not causation."),
        zero_drift_raw_fixed_M_px=distribution([x["zero_drift_raw_px"] for x in zero]),
        zero_single_branch_coverage=distribution([x["zero_single_branch_coverage"] for x in zero]),
        zero_valid_region_drift_status="pending: full-M drift cannot be corrected using coverage scalars alone")


def grouped(rows: list[dict], keys: tuple[str, ...]) -> list[dict]:
    groups = defaultdict(list)
    for r in rows:
        groups[tuple(r[k] for k in keys)].append(r)
    return [dict(group=dict(zip(keys, k)), summary=summarize(v)) for k, v in sorted(groups.items())]


def scalar_direction(row: dict, coverage: float, near_zero: float, consensus: float) -> tuple[str, str]:
    """Relabel stored scalar summaries; no pixels are selected or re-estimated."""
    d = row["direction"]
    if not row["reference_gate"]:
        return "unknown", "reference_calibration_failed"
    if not all(finite(d.get(k)) for k in ("reference_pixels", "valid_pixels", "coverage")):
        return "unknown", d.get("reason", "missing_stored_measurement")
    if d["reference_pixels"] < 400 or d["valid_pixels"] < 200 or d["coverage"] < coverage:
        return "unknown", "insufficient_generated_flow_coverage"
    motion = d.get("motion", {}).get("median_magnitude")
    response = d.get("response", {})
    if not finite(motion):
        raise ValueError("No stored motion for a requested coverage-passing query; cannot relax original coverage gate")
    if motion <= near_zero:
        return "near_zero", "below_fixed_pixel_resolution"
    if response.get("gain_projection") is None:
        return "unknown", "zero_reference_motion"
    c = response.get("direction_consensus")
    assert finite(c)
    if c >= consensus:
        return "correct", "motion_along_reference_flow"
    if 1 - c >= consensus:
        return "reverse", "motion_opposite_reference_flow"
    return "unknown", "mixed_motion_directions"


def sensitivity(rows: list[dict]) -> list[dict]:
    nz = [x for x in rows if x["angle"]]
    # Only changes that can be reconstructed exactly from stored scalar inputs.
    variants = [("frozen", .2, 1., .7)]
    variants += [("coverage_only", c, 1., .7) for c in (.35, .5, .8)]
    variants += [("near_zero_only", .2, z, .7) for z in (.5, 2.)]
    variants += [("consensus_only", .2, 1., c) for c in (.6, .8)]
    out = []
    for variant, c, z, s in variants:
        labels = [scalar_direction(x, c, z, s) for x in nz]
        if variant == "frozen":
            assert all(label == x["label"] and reason == x["reason"]
                       for (label, reason), x in zip(labels, nz)), "Frozen scalar classifier did not reproduce rows"
        counts = {k: sum(label == k for label, _ in labels) for k in LABELS}
        out.append(dict(variant=variant, min_generated_coverage=c, near_zero_px=z,
                        consensus_threshold=s, nonzero_queries=len(nz), **counts,
                        unknown_reasons=dict(Counter(reason for label, reason in labels if label == "unknown"))))
    return out


def pair_metrics(pairs: list[dict]) -> dict:
    nonzero = [x for x in pairs if x["angle"]]
    zeros = [x for x in pairs if not x["angle"]]
    positions = defaultdict(list)
    for p in nonzero:
        if finite(p["raw_epe_delta_px"]):
            positions[p["state_id"]].append(p["raw_epe_delta_px"])
    per_position = [statistics.mean(v) for v in positions.values()]
    return dict(nonzero_query_pairs=len(nonzero), zero_query_pairs=len(zeros),
                positions=len({x["state_id"] for x in pairs}), scenes=len({x["scene_id"] for x in pairs}),
                raw_fixed_M_epe_delta_px=distribution([x["raw_epe_delta_px"] for x in nonzero]),
                raw_fixed_M_gain_delta=distribution([x["raw_gain_delta"] for x in nonzero]),
                per_position_mean_raw_epe_delta_px=distribution(per_position),
                positions_with_negative_mean_raw_epe_delta=sum(x < 0 for x in per_position),
                stored_valid_region_epe_delta_px=distribution([x["valid_epe_delta_px"] for x in nonzero]),
                stored_valid_region_gain_delta=distribution([x["valid_gain_delta"] for x in nonzero]),
                valid_region_comparison_status="exploratory paired query intersection only; per-query pixel supports not common",
                raw_zero_drift_delta_px=distribution([x["zero_drift_raw_delta_px"] for x in zeros]),
                transitions=dict(sorted(Counter(str(x["reference_label"]) + "->" + str(x["test_label"])
                                                for x in nonzero).items())))


def paired_context(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    arms = defaultdict(dict)
    for r in rows:
        arms[(r["weight_arm"], r["context_arm"])][(r["state_id"], r["angle"])] = r
    comparisons = []
    for weight in ("pre_habitat", "adapted"):
        comparisons += [((weight, "static"), (weight, "left_approach")),
                        ((weight, "static"), (weight, "right_approach")),
                        ((weight, "left_approach"), (weight, "right_approach"))]
    comparisons += [(("pre_habitat", h), ("adapted", h))
                    for h in ("static", "left_approach", "right_approach")]
    query_rows, summaries = [], []
    for reference_arm, test_arm in comparisons:
        a, b = arms[reference_arm], arms[test_arm]
        assert a.keys() == b.keys() and len(a) == 36, "Not the same 12 positions x 3 query angles"
        cid = "/".join(test_arm) + " minus " + "/".join(reference_arm)
        pairs = []
        for key in sorted(a):
            ref, test = a[key], b[key]
            def delta(field: str) -> float | None:
                return test[field] - ref[field] if finite(test[field]) and finite(ref[field]) else None
            if ref["angle"]:
                assert ref["raw_pixels"] == test["raw_pixels"] == ref["reference_pixels"] == test["reference_pixels"]
            pairs.append(dict(comparison=cid, state_id=ref["state_id"], scene_id=ref["scene_id"], angle=ref["angle"],
                              repeat=0, reference_key=ref["key"], test_key=test["key"],
                              reference_label=ref["label"], test_label=test["label"],
                              reference_reason=ref["reason"], test_reason=test["reason"],
                              reference_coverage=ref["coverage"], test_coverage=test["coverage"],
                              fixed_reference_pixels=ref["reference_pixels"],
                              reference_valid_pixels=ref["valid_pixels"], test_valid_pixels=test["valid_pixels"],
                              raw_epe_reference_px=ref["raw_epe_px"], raw_epe_test_px=test["raw_epe_px"],
                              raw_epe_delta_px=delta("raw_epe_px"), raw_gain_delta=delta("raw_gain"),
                              valid_epe_reference_px=ref["valid_epe_px"], valid_epe_test_px=test["valid_epe_px"],
                              valid_epe_delta_px=delta("valid_epe_px"), valid_gain_delta=delta("valid_gain"),
                              zero_drift_raw_reference_px=ref["zero_drift_raw_px"],
                              zero_drift_raw_test_px=test["zero_drift_raw_px"],
                              zero_drift_raw_delta_px=delta("zero_drift_raw_px")))
        query_rows.extend(pairs)
        for level, field in (("all", None), ("scene", "scene_id"), ("position", "state_id"), ("signed_angle", "angle")):
            groups = defaultdict(list)
            for p in pairs:
                groups[p[field] if field else "all"].append(p)
            for group, selected in sorted(groups.items(), key=lambda kv: str(kv[0])):
                summaries.append(dict(comparison=cid, level=level, group=group, metrics=pair_metrics(selected)))
    return query_rows, summaries


def load_criteria(path: Path) -> dict:
    # Literal AST read only; do not execute/import frozen source.
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "CRITERIA" for t in node.targets):
            assert isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and node.value.func.id == "dict"
            return {x.arg: ast.literal_eval(x.value) for x in node.value.keywords}
    raise ValueError("Frozen CRITERIA not found")


def self_test() -> None:
    assert quantile([0, 10], .25) == 2.5
    assert distribution([])["mean"] is None and distribution([1, None, 3])["n"] == 2
    row = dict(reference_gate=True, direction=dict(reference_pixels=1000, valid_pixels=500, coverage=.5,
        motion=dict(median_magnitude=1.5), response=dict(gain_projection=.1, direction_consensus=.75)))
    assert scalar_direction(row, .2, 1, .7)[0] == "correct"
    assert scalar_direction(row, .8, 1, .7)[0] == "unknown"
    assert scalar_direction(row, .2, 2, .7)[0] == "near_zero"
    assert scalar_direction(row, .2, 1, .8)[1] == "mixed_motion_directions"
    row["direction"]["response"]["direction_consensus"] = .1
    assert scalar_direction(row, .2, 1, .7)[0] == "reverse"
    no_motion = dict(reference_gate=True, direction=dict(reference_pixels=1000, valid_pixels=500, coverage=.5))
    try:
        scalar_direction(no_motion, .2, 1, .7)
    except ValueError:
        pass
    else:
        raise AssertionError("Missing stored motion must not be guessed")
    print("self-test passed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[4])
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    root, out = args.root.resolve(), args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    paths = dict(
        context_nwm=root / "model_deployment/input_context_controls_v1/retrieval/bundle/habitat/nwm/analysis/rows.json",
        baseline_nwm=root / "model_deployment/confirmation_v1/confirmation/nwm/analysis/rows.json",
        baseline_raenwm=root / "model_deployment/confirmation_v1/confirmation/raenwm/analysis/rows.json",
        baseline_uniwm=root / "model_deployment/confirmation_v1/confirmation/uniwm/analysis/rows.json")
    provenance_paths = [root / "model_deployment/input_context_controls_v1/retrieval/bundle/protocol.json",
        root / "model_deployment/input_context_controls_v1/retrieval/bundle/execution_freeze.json",
        root / "model_deployment/confirmation_v1/protocol.json",
        root / "model_deployment/confirmation_v1/execution_freeze.json",
        root / "model_deployment/confirmation_v1/flow/references/summary.json"]
    source_paths = [root / "model_deployment" / x for x in
        ("score_context_controls_v1.py", "score_confirmation_v1.py", "probe_habitat_flow_v1.py", "habitat_flow_common.py")]
    context = normalize(read(paths["context_nwm"]), "context_nwm", "nwm", str(paths["context_nwm"].relative_to(root)))
    baseline = []
    for model in ("nwm", "raenwm", "uniwm"):
        p = paths["baseline_" + model]
        rows = normalize(read(p), "confirmation", model, str(p.relative_to(root)))
        assert len(rows) == 252 and sum(bool(x["angle"]) for x in rows) == 216
        baseline.extend(rows)
    assert len(context) == 216 and sum(bool(x["angle"]) for x in context) == 144
    all_rows = context + baseline
    criteria = load_criteria(root / "model_deployment/habitat_flow_common.py")
    for key, expected in (("min_texture_pixels", 400), ("min_valid_pixels", 200),
                          ("min_generated_coverage", .2), ("near_zero_px", 1.), ("direction_consensus", .7)):
        assert criteria[key] == expected, "Scalar sensitivity constants no longer match frozen criteria"
    freeze = read(provenance_paths[1])
    freeze_checks = {}
    for p in source_paths:
        frozen_sha = freeze["source_hashes"].get("external/habitat-yaw-reference/" + p.name)
        freeze_checks[p.name] = dict(current_sha256=sha(p), context_frozen_sha256=frozen_sha,
                                     matches_context_freeze=sha(p) == frozen_sha if frozen_sha else None)
        if frozen_sha:
            assert sha(p) == frozen_sha, "Local source changed since context execution freeze"
    masks = defaultdict(set)
    for r in all_rows:
        if r["angle"] and r["reference_gate"]:
            masks[r["state_id"]].add(r["raw_pixels"])
    assert len(masks) == 12 and all(len(v) == 1 for v in masks.values()), "Reference mask sizes not position-fixed"
    grouped_context = []
    for level, keys in (("arm", ("weight_arm", "context_arm")),
                        ("scene", ("weight_arm", "context_arm", "scene_id")),
                        ("position", ("weight_arm", "context_arm", "state_id")),
                        ("signed_angle", ("weight_arm", "context_arm", "angle"))):
        for group in grouped(context, keys):
            grouped_context.append(dict(level=level, **group))
    grouped_baseline = []
    for level, keys in (("model", ("model",)), ("absolute_angle", ("model", "abs_angle")),
                        ("signed_angle", ("model", "angle")), ("scene_angle", ("model", "scene_id", "abs_angle")),
                        ("position_angle", ("model", "state_id", "abs_angle"))):
        for group in grouped(baseline, keys):
            grouped_baseline.append(dict(level=level, **group))
    query_pairs, paired = paired_context(context)
    threshold_rows = []
    threshold_groups = defaultdict(list)
    for r in all_rows:
        if r["dataset"] == "context_nwm":
            group = (r["dataset"], r["model"], r["weight_arm"], r["context_arm"], "all_nonzero")
            threshold_groups[group].append(r)
        else:
            for angle in ("all_nonzero", str(r["abs_angle"])):
                threshold_groups[(r["dataset"], r["model"], r["weight_arm"], r["context_arm"], angle)].append(r)
    for group, rows in sorted(threshold_groups.items()):
        if not any(x["angle"] for x in rows):
            continue
        for result in sensitivity(rows):
            threshold_rows.append(dict(zip(("dataset", "model", "weight_arm", "context_arm", "angle_group"), group), **result))
    original_lookup = {(x["state_id"], x["angle"], x["repeat"]): x
                       for x in baseline if x["model"] == "nwm"}
    reuse_checks = []
    for r in context:
        if r["weight_arm"] == "adapted" and r["context_arm"] == "static":
            old = original_lookup[r["state_id"], r["angle"], 0]
            fields = ("raw_epe_px", "raw_gain", "zero_drift_raw_px", "valid_epe_px", "valid_gain", "coverage")
            matches = all(r[k] == old[k] for k in fields)
            assert matches and r["label"] == old["label"]
            reuse_checks.append(dict(context_key=r["key"], baseline_key=old["key"], stored_metrics_exact_match=True))
    assert len(reuse_checks) == 36
    inputs = {str(p.relative_to(root)): dict(sha256=sha(p), bytes=p.stat().st_size)
              for p in list(paths.values()) + provenance_paths + source_paths}
    report = dict(
        status="completed_stored_scalar_reaggregation_only", audit_date="2026-10-30",
        script_sha256=sha(Path(__file__)), inputs=inputs, source_freeze_checks=freeze_checks,
        units=dict(row_appearances=972, context_nwm_conditions=216, original_confirmation_conditions=756,
                   duplicate_context_adapted_static_conditions=36, world_model_predictions_added=0,
                   independent_scenes=2, positions=12,
                   note="Stored row appearances are not new predictions, independent scenes, or an effective sample-size claim."),
        definitions=dict(
            fixed_reference_M="Per-position common intersection of raw-reference texture/visibility/FB-valid masks over 0 and +/-3/5/7 degrees; output-independent.",
            raw_fixed_M="EPE and gain of F(I0,Ga)-F(I0,G0) vs F(I0,Ia)-F(I0,I0), computed on all M. Generated validity is not required.",
            direction_valid_support="M & valid(I0,Ga) & valid(I0,G0); unknown on low coverage or mixed directions.",
            stored_valid_region="direction.response already stored EPE/gain on the direction-valid support for coverage-passing queries, including mixed-direction unknowns.",
            raw_zero_drift="Magnitude of F(I0,G0)-F(I0,I0) on all M; generated-valid pixels are not selected.",
            means="Unweighted means of per-query scalar values; distribution.n is the precise contributing denominator.",
            paired_sign="Test minus reference; negative raw EPE or drift delta is numerically lower.",
            thresholds="One-factor stored-scalar reclassification, not a fresh pixel-level sensitivity evaluation."),
        frozen_criteria=criteria,
        per_position_fixed_reference_pixels={k: next(iter(v)) for k, v in sorted(masks.items())},
        context_nwm=grouped_context, original_confirmation=grouped_baseline,
        context_paired_contrasts=paired, threshold_scalar_sensitivity=threshold_rows,
        exact_reused_stored_metric_checks=reuse_checks,
        pending_measurements=[
            "Cross-arm common generated-reliable pixel intersections and metrics on them need cached per-pixel flow/valid arrays; absent from these rows and the retrieved context bundle.",
            "Valid-region zero-action drift needs per-pixel zero-branch flow/valid arrays; context rows do not even store zero-branch coverage.",
            "Changing forward/backward pixel-validity thresholds, texture masks, or relaxing generated coverage below 0.2 cannot be re-evaluated from stored summaries.",
            "Real-reference or synthetic-shift calibration does not establish generated-image flow accuracy or physical yaw accuracy."],
        limitations=[
            "Raw EPE/gain/drift are unvalidated full-M flow diagnostics where generated correspondence fails; no physical-error or reliable-control claim follows from their means.",
            "Direction unknown is not measured near-zero, reverse, or generation failure. Mixed-direction unknowns can have high valid coverage.",
            "Stored valid-region means use varying query and pixel sets. Even paired query intersections do not equal common-pixel support; not a repaired primary comparison.",
            "Scalar threshold sensitivity preserves the original per-pixel validity masks and does not validate RAFT in generated domains.",
            "Context uses +/-5 only and one paired seed; original +/-3/5/7 uses three seeds. They are not interchangeable angle-response experiments.",
            "The 12 reused positions lie in only two scenes; seed repeats are not independent scenes and no population confidence interval is produced.",
            "Old adapted/static noise hash includes dtype/shape while new context hashes raw bytes. Actual cross-format noise equality is not independently demonstrated.",
            "Both fixed supports and varying valid supports may reflect rendering artifacts; image-space gain is not angle calibration or causal attribution."])
    (out / "support_audit.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    csv_write(out / "stored_rows.csv", [{k: v for k, v in x.items() if k != "direction"} for x in all_rows])
    csv_write(out / "context_grouped.csv", [dict(level=x["level"], **x["group"], **flatten(x["summary"])) for x in grouped_context])
    csv_write(out / "baseline_angle_response.csv", [dict(level=x["level"], **x["group"], **flatten(x["summary"])) for x in grouped_baseline])
    csv_write(out / "context_query_pairs.csv", query_pairs)
    csv_write(out / "context_paired_aggregates.csv", [dict(comparison=x["comparison"], level=x["level"], group=x["group"], **flatten(x["metrics"])) for x in paired])
    csv_write(out / "threshold_scalar_sensitivity.csv", threshold_rows)
    csv_write(out / "unknown_queries.csv", [{k: v for k, v in x.items() if k != "direction"}
                                            for x in all_rows if x["angle"] and x["label"] == "unknown"])
    table_rows = []
    for x in grouped_context:
        if x["level"] != "arm":
            continue
        s = x["summary"]
        table_rows.append(dict(**x["group"], nonzero_n=s["nonzero_queries"],
            **s["direction"], coverage_mean=s["coverage"]["mean"], coverage_median=s["coverage"]["median"],
            raw_epe_full_M_mean=s["raw_fixed_M_epe_px"]["mean"],
            raw_epe_unknown_n=s["unknown_contribution"]["raw_unknown"]["n"],
            raw_epe_unknown_mean=s["unknown_contribution"]["raw_unknown"]["mean"],
            unknown_fraction_of_raw_epe_sum=s["unknown_contribution"]["fraction_of_raw_epe_sum"],
            valid_region_epe_n=s["stored_valid_region_epe_px"]["n"],
            valid_region_epe_mean=s["stored_valid_region_epe_px"]["mean"],
            raw_gain_mean=s["raw_fixed_M_gain"]["mean"], raw_gain_median=s["raw_fixed_M_gain"]["median"],
            valid_gain_mean=s["stored_valid_region_gain"]["mean"],
            valid_gain_median=s["stored_valid_region_gain"]["median"],
            raw_zero_drift_mean=s["zero_drift_raw_fixed_M_px"]["mean"],
            support_note="Raw full-M unvalidated generated flow; valid-region columns have varying sample/pixel supports."))
    csv_write(out / "context_support_table.csv", table_rows)
    angle_rows = []
    for x in grouped_baseline:
        if x["level"] != "absolute_angle" or not x["group"]["abs_angle"]:
            continue
        s = x["summary"]
        angle_rows.append(dict(**x["group"], nonzero_n=s["nonzero_queries"], positions=s["positions"], scenes=s["scenes"],
            **s["direction"], coverage_mean=s["coverage"]["mean"],
            raw_gain_mean=s["raw_fixed_M_gain"]["mean"], raw_gain_median=s["raw_fixed_M_gain"]["median"],
            valid_response_n=s["stored_valid_region_gain"]["n"], valid_gain_mean=s["stored_valid_region_gain"]["mean"],
            valid_gain_median=s["stored_valid_region_gain"]["median"],
            valid_response_median_motion_median_px=s["stored_valid_region_motion_median_px"]["median"],
            support_note="Three paired seeds, not independent scenes; valid sample/pixel support differs by model and angle."))
    csv_write(out / "baseline_small_angles_table.csv", angle_rows)
    manifest = {p.name: dict(sha256=sha(p), bytes=p.stat().st_size)
                for p in sorted(out.iterdir()) if p.is_file() and p.name != "artifact_manifest.json"}
    (out / "artifact_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(dict(status=report["status"], output=str(out), row_appearances=len(all_rows),
                          context_nonzero=144, original_nonzero=648, duplicate_reuse_checks=36,
                          threshold_frozen_reclassification="exact for all 792 nonzero row appearances")))


if __name__ == "__main__":
    main()
