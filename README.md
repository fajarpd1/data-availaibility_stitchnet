# Data Availability

Experiment code and result artifacts for the StitchingNet-lineage
deployment-validity study. No manuscript, response-letter, or style-guide
content is included in this folder; only experiment code and result
artifacts.

Raw model checkpoints (`.keras`) and full per-run image-split manifests are
not included in this repository due to size; they are available from the
corresponding author on reasonable request.

## 01_first_experiment/ (original evidence package, 2026-08-09)

- `dataset_profile.json`, `final_model_selection.json`, `runtime_info.json`,
  `FINAL_PUBLICATION_EVIDENCE_MANIFEST_20260809.json` — run/dataset metadata.
- `models/final_model.tflite` — exported edge-deployment model.
- `results/` — per-run metrics, confusion matrices, classification reports
  (wide-screening, fine-tuning, LODO, cold-start).
- `reports/` — aggregated summary CSVs/JSON and audit files (LODO summary by
  domain/seed, cold-start summary by support level, prediction-coverage audit,
  protocol-validity audit).
- `figures/` — Grad-CAM failure-case gallery.
- `source_manifests/` — LODO source/coverage manifests.

## 02_revision_additional_experiments/ (round-3 addendum, 2026-09)

- `code/` — notebooks/scripts used to generate the addendum evidence:
  - `stitchnet-v5-revision-r3-step1-matched-budget-lodo.ipynb` (matched-budget
    LODO control)
  - `stitchnet-v5-revision-r3-step2-fixed-query-coldstart-jupyterlab-full.py`
    / `.ipynb` (fixed-query-set cold-start protocol, 198 runs)
- `results/` — per-run metrics, predictions (`y_true`/`y_pred`/`y_prob`),
  confusion matrices, classification reports for the fixed-query cold-start
  runs and the matched-budget/Grad-CAM LODO checkpoints. **This is the
  canonical source for every matched-budget LODO and LODO-condition
  calibration number reported in the manuscript**
  (`domain_generalisation_matched_budget_results.csv`,
  `calibration_lodo_matched_results.csv`, `calibration_lodo_matched_summary.csv`).
- `figures/` — Grad-CAM galleries (fixed-query cold-start, matched-budget
  LODO, domains I and C).
- `kaggle_matched_budget_run/` — a preliminary, Kaggle-executed run of the
  matched-budget LODO protocol, superseded by the canonical run above. Kept
  for transparency only; **not** the source of any number reported in the
  manuscript. Its two summary CSVs are suffixed `_PRELIMINARY_KAGGLE_SUPERSEDED`
  to avoid being confused with the identically-named canonical files in
  `results/` above (see this folder's own `README.md`).
- `dataset_profile.json`, `runtime_info.json` — run metadata for this stage.

## Source paths on the working machine

- First experiment: `q1-deplo..study/Stitchne..l run/00_FINAL..60809/02_final..acted/final_pu..60809/`
- Revision code: `q1-deployment-validity-study/revision_round3_20260917/`
- Revision results: `stitchnet_q1_ready_v5_compact/`
