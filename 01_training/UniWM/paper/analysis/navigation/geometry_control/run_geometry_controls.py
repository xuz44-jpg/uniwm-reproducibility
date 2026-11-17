#!/usr/bin/env python3
"""CPU-only analytic controls; never render, infer flow, or call a world model.

Use the existing camera K and frozen reference-only common masks.  The inputs
contain old real-reference flow audits, but only ``common_mask`` is read from
each NPZ.  All displacement fields below are exact pinhole projections.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


ANGLES = (-7, -5, -3, 3, 5, 7)
DRIFTS = (0, 5, 10, 15)
RESPONSES = {
    "correct": 1.0,
    "zero": 0.0,
    "reverse": -1.0,
    "under_0p2": 0.2,
    "under_0p5": 0.5,
    "over_2": 2.0,
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def scaled_k(camera, size):
    # Exactly the pixel-center convention of habitat_study_common.scaled_K.
    sx, sy = size / camera["width"], size / camera["height"]
    k = np.asarray(camera["K"], dtype=np.float64).copy()
    k[0, :] *= sx
    k[1, :] *= sy
    k[0, 2] += 0.5 * sx - 0.5
    k[1, 2] += 0.5 * sy - 0.5
    assert np.allclose(k[2], [0, 0, 1])
    assert k[0, 1] == k[1, 0] == 0, "This control assumes the existing zero-skew K"
    assert np.allclose(camera["distortion"], 0), "Existing camera is undistorted"
    return k


def yaw_rotation(angle_deg):
    theta = np.deg2rad(angle_deg)
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], np.float64)


def x_rotation(angle_deg):
    theta = np.deg2rad(angle_deg)
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], np.float64)


def project_yaw(points, angle_deg, k):
    """h_theta for arbitrary pixel coordinates, before any image resampling."""
    x = (points[..., 0] - k[0, 2]) / k[0, 0]
    y = (points[..., 1] - k[1, 2]) / k[1, 1]
    theta = np.deg2rad(angle_deg)
    c, s = np.cos(theta), np.sin(theta)
    z = c - x * s
    with np.errstate(divide="raise", invalid="raise", over="raise"):
        u = k[0, 2] + k[0, 0] * (c * x + s) / z
        v = k[1, 2] + k[1, 1] * y / z
    projected = np.stack([u, v], axis=-1)
    assert np.isfinite(projected).all()
    return projected, z > 0


def visibility(points, positive_depth, size):
    # Same open border as the frozen geometry_flow and RAFT FB-validity path.
    return (positive_depth & (points[..., 0] >= 1) &
            (points[..., 0] < size - 1) & (points[..., 1] >= 1) &
            (points[..., 1] < size - 1))


def vector_stats(values):
    if not len(values):
        return {name: None for name in (
            "mean_magnitude_px", "median_magnitude_px", "median_x_px", "median_y_px")}
    magnitude = np.linalg.norm(values, axis=-1)
    return dict(mean_magnitude_px=float(magnitude.mean()),
                median_magnitude_px=float(np.median(magnitude)),
                median_x_px=float(np.median(values[:, 0])),
                median_y_px=float(np.median(values[:, 1])))


def measure(predicted, reference, mask, criteria):
    """Replicate frozen EPE/gain/sign rules, without flow-estimator uncertainty."""
    x, y = predicted[mask], reference[mask]
    n = len(x)
    if not n:
        return dict(pixels=0, mean_epe_px=None, median_epe_px=None,
                    p90_epe_px=None, projected_gain=None,
                    positive_projection_fraction=None, label="unknown",
                    label_reason="empty_analytic_common_visibility")
    epe = np.linalg.norm(x - y, axis=-1)
    dot = np.sum(x * y, axis=-1)
    denom = float(np.square(y).sum())
    stats = vector_stats(x)
    if n < max(criteria["min_texture_pixels"], criteria["min_valid_pixels"]):
        label, reason = "unknown", "insufficient_analytic_pixels"
    elif stats["median_magnitude_px"] <= criteria["near_zero_px"]:
        label, reason = "near_zero", "below_frozen_one_pixel_median_rule"
    elif denom <= 1e-10:
        label, reason = "unknown", "zero_reference_motion"
    elif float((dot > 0).mean()) >= criteria["direction_consensus"]:
        label, reason = "correct", "positive_projection_rule_not_angle_accuracy"
    elif 1 - float((dot > 0).mean()) >= criteria["direction_consensus"]:
        label, reason = "reverse", "negative_projection_rule"
    else:
        label, reason = "unknown", "mixed_projection_signs"
    return dict(pixels=n, mean_epe_px=float(epe.mean()),
                median_epe_px=float(np.median(epe)),
                p90_epe_px=float(np.percentile(epe, 90)),
                projected_gain=float(dot.sum() / denom) if denom > 1e-10 else None,
                positive_projection_fraction=float((dot > 0).mean()),
                predicted_mean_magnitude_px=stats["mean_magnitude_px"],
                predicted_median_magnitude_px=stats["median_magnitude_px"],
                reference_mean_magnitude_px=float(np.linalg.norm(y, axis=-1).mean()),
                label=label, label_reason=reason)


def describe(values):
    groups = defaultdict(list)
    for row in values:
        groups[(row["support"], row["response"], row["drift_deg"])].append(row)
    result = []
    for (support, response, drift), rows in sorted(groups.items()):
        # Every original position has the same six angles in these summaries.
        # No seed/scene independence or inferential interval is claimed.
        metrics = ["mean_epe_px", "projected_gain", "mask_retained_fraction",
                   "known_relative_yaw_error_deg", "relative_map_identity_max_px",
                   "command_map_mean_residual_px"]
        item = dict(support=support, response=response, drift_deg=drift,
                    conditions=len(rows), positions=len({r["state_id"] for r in rows}),
                    labels={label: sum(r["label"] == label for r in rows)
                            for label in ["correct", "reverse", "near_zero", "unknown"]})
        for metric in metrics:
            v = [r[metric] for r in rows if r[metric] is not None]
            item[metric] = dict(conditions=len(v), mean=float(np.mean(v)),
                                min=float(min(v)), max=float(max(v))) if v else None
        result.append(item)
    return result


def load_inputs(repository, snapshot):
    """Read the original repository once, or the portable K/mask-only snapshot."""
    masks, sources = {}, {}
    if snapshot is not None:
        saved = read_json(snapshot / "input_snapshot.json")
        arrays = snapshot / "input_masks.npz"
        assert sha256(arrays) == saved["mask_archive_sha256"]
        with np.load(arrays, allow_pickle=False) as data:
            masks = {key: data[key].astype(bool) for key in data.files}
        return saved["camera"], saved["criteria"], saved["reference_audit"], masks, saved["original_sources_sha256"]
    camera_path = repository / "model_deployment/confirmation_v1/references_manifest.json"
    mask_root = repository / "model_deployment/confirmation_v1/flow/references"
    config_path = mask_root / "config.json"
    summary_path = mask_root / "summary.json"
    camera = read_json(camera_path)["camera"]
    criteria = read_json(config_path)["criterion"]
    summary = read_json(summary_path)
    for path in [camera_path, config_path, summary_path,
                 repository / "model_deployment/habitat_flow_common.py",
                 repository / "model_deployment/habitat_study_common.py",
                 repository / "model_deployment/score_context_controls_v1.py"]:
        sources[str(path.relative_to(repository))] = sha256(path)
    for state_id in sorted(summary["states"]):
        path = mask_root / (state_id + ".npz")
        with np.load(path, allow_pickle=False) as data:
            masks[state_id] = data["common_mask"].astype(bool)
        sources[str(path.relative_to(repository))] = sha256(path)
    audit = dict(all_reference_gates_passed=summary["all_reference_gates_passed"],
                 states={sid: {name: row[name] for name in ["common_pixels", "reference_gate_passed"]}
                         for sid, row in summary["states"].items()})
    return camera, criteria, audit, masks, sources


def run(repository, output, snapshot=None):
    camera, criteria, summary, masks, sources = load_inputs(repository, snapshot)
    assert summary["all_reference_gates_passed"]
    size = criteria["resolution"]
    k = scaled_k(camera, size)
    yy, xx = np.mgrid[:size, :size]
    points = np.stack([xx, yy], axis=-1).astype(np.float64)
    projected = {}

    def h(angle):
        key = round(float(angle), 12)
        if key not in projected:
            q, positive = project_yaw(points, angle, k)
            projected[key] = (q, visibility(q, positive, size))
        return projected[key]

    for state_id, audit in sorted(summary["states"].items()):
        mask = masks[state_id]
        assert mask.shape == (size, size) and mask.sum() == audit["common_pixels"]
        assert audit["reference_gate_passed"]
    assert len(masks) == 12

    # A fully matched visibility support, specific to each position/angle but
    # identical across every drift and response arm.  No generated result is
    # consulted; this is analytic camera-FOV intersection only.
    shared = {}
    for state_id, mask in masks.items():
        for angle in ANGLES:
            selected = mask & h(0)[1] & h(angle)[1]
            for drift in DRIFTS:
                selected = selected & h(drift)[1]
                for gain in RESPONSES.values():
                    selected = selected & h(drift + gain * angle)[1]
            shared[(state_id, angle)] = selected
            assert selected.sum() >= criteria["min_texture_pixels"]

    rows, tests = [], []
    max_group_identity = 0.0
    for state_id, original in masks.items():
        for angle in ANGLES:
            qa, va = h(angle)
            reference = qa - points
            for drift in DRIFTS:
                qb, vb = h(drift)
                for name, gain in RESPONSES.items():
                    qba, vba = h(drift + gain * angle)
                    delta = qba - qb
                    qcomposed, _ = project_yaw(qb, gain * angle, k)
                    qcommand, _ = project_yaw(qb, angle, k)
                    identity = np.linalg.norm(qba - qcomposed, axis=-1)
                    command_residual = np.linalg.norm(qba - qcommand, axis=-1)
                    max_group_identity = max(max_group_identity, float(identity.max()))
                    within = original & h(0)[1] & va & vb & vba
                    for support, selected in [
                        ("within_condition_common_visible", within),
                        ("matched_across_all_drifts_and_responses", shared[(state_id, angle)]),
                    ]:
                        row = dict(state_id=state_id,
                                   scene_id="apartment_1" if state_id.startswith("apartment_1_") else "skokloster_castle",
                                   query_yaw_deg=angle, drift_deg=drift,
                                   response=name, response_multiplier=gain,
                                   constructed_zero_yaw_deg=drift,
                                   constructed_action_yaw_deg=drift + gain * angle,
                                   known_relative_yaw_deg=gain * angle,
                                   known_relative_yaw_error_deg=abs((gain - 1) * angle),
                                   support=support, original_reference_pixels=int(original.sum()),
                                   common_visible_pixels=int(selected.sum()),
                                   mask_retained_fraction=float(selected.sum() / original.sum()),
                                   relative_map_identity_max_px=float(identity[selected].max()),
                                   command_map_mean_residual_px=float(command_residual[selected].mean()))
                        row.update(measure(delta, reference, selected, criteria))
                        rows.append(row)
                        if name == "correct":
                            assert row["known_relative_yaw_error_deg"] == 0
                            assert row["command_map_mean_residual_px"] < 1e-10
                        if name == "correct" and drift == 0:
                            assert row["mean_epe_px"] < 1e-10
                            assert abs(row["projected_gain"] - 1) < 1e-12
                        if name == "zero":
                            assert row["label"] == "near_zero"
                            assert abs(row["projected_gain"]) < 1e-12
                        if name == "reverse":
                            assert row["label"] == "reverse"
    assert len(rows) == 12 * 6 * 4 * 6 * 2
    assert max_group_identity < 1e-10
    tests.append(dict(name="all_pure_yaw_group_compositions",
                      passed=True, max_endpoint_residual_px=max_group_identity))
    tests.append(dict(name="identity_correct_zero_and_reverse_control_assertions", passed=True))
    r_y, r_x = yaw_rotation(10), x_rotation(7)
    noncommutation = float(np.linalg.norm(r_y @ r_x - r_x @ r_y))
    assert noncommutation > 1e-3
    tests.append(dict(name="general_rotation_is_not_assumed_commutative",
                      passed=True, xy_commutator_frobenius=noncommutation))
    positive = [r for r in rows if r["response"] in ["under_0p2", "under_0p5", "over_2"]]
    tests.append(dict(name="positive_projection_does_not_require_unit_amplitude",
                      passed=any(r["label"] == "correct" for r in positive),
                      labels={label: sum(r["label"] == label for r in positive)
                              for label in ["correct", "reverse", "near_zero", "unknown"]}))
    assert all(t["passed"] for t in tests)

    # Exact reproduction of the previously communicated 24-condition example.
    example = [r for r in rows if r["support"] == "within_condition_common_visible"
               and r["response"] == "correct" and r["drift_deg"] == 10
               and abs(r["query_yaw_deg"]) == 5]
    assert len(example) == 24
    sanity = dict(conditions=24, positions=12, yaw_angles_deg=[-5, 5], drift_deg=10,
                  relative_yaw_perfect_by_construction=True,
                  equal_weight_condition_mean_epe_px=float(np.mean([r["mean_epe_px"] for r in example])),
                  mean_original_mask_retained_fraction=float(np.mean([r["mask_retained_fraction"] for r in example])),
                  condition_mean_epe_range_px=[min(r["mean_epe_px"] for r in example),
                                               max(r["mean_epe_px"] for r in example)])
    assert abs(sanity["equal_weight_condition_mean_epe_px"] - 2.236554824017427) < 1e-10
    matched_example = [r for r in rows if r["support"] == "matched_across_all_drifts_and_responses"
                       and r["response"] == "correct" and r["drift_deg"] == 10
                       and abs(r["query_yaw_deg"]) == 5]
    assert len(matched_example) == 24
    matched_sanity = dict(conditions=24, positions=12, yaw_angles_deg=[-5, 5], drift_deg=10,
                         relative_yaw_perfect_by_construction=True,
                         support="matched_across_all_drifts_and_responses",
                         retention_denominator="original frozen reference common_mask pixels at each position, not the full image or texture mask",
                         equal_weight_condition_mean_epe_px=float(np.mean([r["mean_epe_px"] for r in matched_example])),
                         mean_original_mask_retained_fraction=float(np.mean([r["mask_retained_fraction"] for r in matched_example])),
                         condition_mean_epe_range_px=[min(r["mean_epe_px"] for r in matched_example),
                                                      max(r["mean_epe_px"] for r in matched_example)])

    source_self = Path(__file__).resolve()
    script_key = "output/aaai_workshop/geometry_control/run_geometry_controls.py"
    if snapshot is not None:
        assert sources[script_key] == sha256(source_self), "Portable inputs require the recorded calibration script"
    sources[script_key] = sha256(source_self)
    output.mkdir(parents=True, exist_ok=True)
    # This small portable snapshot contains K, criteria and boolean masks only.
    # Original file identities remain recorded; no RGB, flow or model data is copied.
    input_archive = output / "input_masks.npz"
    np.savez_compressed(input_archive, **{key: masks[key] for key in sorted(masks)})
    save_json(output / "input_snapshot.json", dict(camera=camera, criteria=criteria,
              reference_audit=summary, original_sources_sha256=sources,
              mask_archive_sha256=sha256(input_archive),
              scope="K, frozen criteria and existing boolean reference masks only; no RGB, depth, flow fields, or model predictions"))
    csv_path = output / "analytic_controls.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = dict(status="passed", evidence_type="CPU analytic pinhole projection only",
                  world_model_predictions=0, flow_estimator_calls=0, rendered_images=0,
                  training_updates=0, new_positions=0,
                  independent_scenes_added=0,
                  original_positions=12, analytic_conditions=len(rows) // 2,
                  support_views_per_condition=2, csv_rows=len(rows),
                  camera_k_scaled=k.tolist(), image_size=[size, size],
                  controls=dict(query_yaw_deg=list(ANGLES), common_drift_deg=list(DRIFTS),
                                response_multipliers=RESPONSES),
                  construction="generated-zero surrogate=h_b(p), action surrogate=h_(b+g*a)(p); reference effect=h_a(p)-p",
                  support_policies={
                      "within_condition_common_visible": "existing reference mask intersect positive depth and [1,255) projection for h_0,h_a,h_b,h_(b+g*a)",
                      "matched_across_all_drifts_and_responses": "for each position/query angle, intersect the within-condition masks across all b and g before computing any metric"},
                  direction_rule="Frozen positive/negative dot-product consensus and median one-pixel threshold, evaluated on perfect analytic visibility; not a new RAFT acceptance audit",
                  known_relative_yaw_error="Known from constructed rotations, not inferred from images or claimed for model outputs",
                  aggregation="Equal-weight per-condition means; each summary has 12 positions x 6 angles; descriptive only, no scene/seed independence or CI",
                  limitations=[
                      "No RAFT or SIFT was run; this isolates geometric non-invariance, not estimator error or cross-domain reliability.",
                      "Known drift is a perfect coaxial yaw; model drift can be translation, deformation, unknown rotation, or correspondence error.",
                      "Nonzero ideal paired EPE cannot be subtracted from measured model EPE or used to explain all observed NWM differences.",
                      "Projected gain and the positive-projection label remain pixel-response diagnostics, not physical angle accuracy.",
                      "The common-axis group identity must not be generalized to scalar addition of arbitrary 3D rotations.",
                      "The old reference masks include their original real-reference audit decisions; no image data, generated-valid mask, or model success was used to select these control pixels."],
                  tests=tests, communicated_example=sanity,
                  recommended_matched_example=matched_sanity,
                  by_support_response_drift=describe(rows),
                  sources_sha256=sources,
                  portable_inputs_sha256={"input_snapshot.json": sha256(output / "input_snapshot.json"),
                                          "input_masks.npz": sha256(input_archive)},
                  csv_sha256=sha256(csv_path))
    save_json(output / "analytic_controls.json", result)
    save_json(output / "source_hashes.json", dict(sources_sha256=sources))
    print(json.dumps(dict(status="passed", output=str(output), conditions=len(rows) // 2,
                          csv_rows=len(rows), communicated_example=sanity,
                          tests=tests), indent=2, allow_nan=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path,
                        default=next((p for p in Path(__file__).resolve().parents
                                      if (p / "model_deployment/confirmation_v1").is_dir()), Path.cwd()))
    parser.add_argument("--inputs", type=Path,
                        help="Portable snapshot directory; avoids reading any original repository files")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    run(args.repository.resolve(), args.output.resolve(), args.inputs.resolve() if args.inputs else None)
