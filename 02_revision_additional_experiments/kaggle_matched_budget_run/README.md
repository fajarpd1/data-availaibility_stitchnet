# Preliminary Kaggle matched-budget LODO run (superseded)

This folder holds an early, Kaggle-executed run of the matched-budget LODO
protocol, produced before the canonical run's checkpoints were available.
**It is not the source of any matched-budget or LODO-condition-calibration
number reported in the manuscript.**

The canonical run underlying the manuscript's Section 4.2, Supplementary
Table S10 (matched-budget macro-F1), and Supplementary Table S11
(LODO-condition calibration) is deposited one directory up, at
`../results/domain_generalisation_matched_budget_results.csv` and
`../results/calibration_lodo_matched_results.csv` /
`../results/calibration_lodo_matched_summary.csv`.

The two summary CSVs in this folder's own `stitchne..mpact/results/` are
suffixed `_PRELIMINARY_KAGGLE_SUPERSEDED` to prevent them being confused
with the identically-named canonical files one directory up. The per-run
prediction/checkpoint artifacts in this folder (classification reports,
confusion matrices, `y_true`/`y_pred`/`y_prob` arrays) are kept for
transparency and reproducibility of the exploratory run itself, not as
supporting evidence for any number in the manuscript.
