# Navigation evidence analysis

These are the retained simulator records and CPU analyses underlying Chapters 3–7 and Appendices E–F. No model predictions or training were added during manuscript revision. `data/` includes all 984 follow-up records and the original static/angle summaries. `support_audit/` keeps separate raw fixed-mask and existing valid-region metrics; common generated-reliable pixels and reliable zero drift remain pending. `geometry_control/` is an analytic projection check, not image-RAFT calibration. `image_evidence/` includes the complete NWM position gallery.

R is the legacy reverse label computed as 1 minus the fraction of positive dot products; zero dot products are included. Do not reinterpret its counts as measured reverse yaw.

Recompute stored metrics (standard-library Python):

```sh
python support_audit/audit_support_from_rows.py --root recompute_inputs --out output/support
python geometry_control/run_geometry_controls.py --inputs geometry_control --output output/geometry
```

The analytic script requires NumPy. File hashes establish identity, not scientific correctness. The original generated PNGs are not included. The older 1X measurements have their own provenance and are not rerun here.
