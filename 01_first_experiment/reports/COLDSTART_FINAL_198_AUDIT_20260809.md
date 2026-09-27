# Cold-Start Final 198-Row Audit - 2026-08-09

## Verdict

Cold-start adaptation is complete and can now be used as selected cold-start evidence.

The final source is:

```text
stitchnet_v5_cold_start_checkpoint_FINAL_SEED1_IJK_FROM_180_ROWS_after_seed1_domainK_20260809_111350
```

The intermediate domain-J checkpoint is valid provenance but not final:

```text
stitchnet_v5_cold_start_checkpoint_FINAL_SEED1_IJK_FROM_180_ROWS_after_seed1_domainJ_20260809_101622
```

## Protocol Audit

Expected protocol:

- 11 held-out domains: A-K
- 3 seeds: 0, 1, 2
- 6 support sizes: 0, 1, 5, 10, 25, 50
- expected rows: 11 x 3 x 6 = 198

Observed final table:

- rows: 198
- valid protocol keys: 198 / 198
- duplicate `(held_out_domain, support_per_class, seed)` keys: 0
- missing keys: 0
- extra keys: 0
- table hash: `7e67ef84afda`

Complete domains:

- seed 0: A-K
- seed 1: A-K
- seed 2: A-K

## Final Selected Evidence

Created:

```text
_selected_for_use_20260809/coldstart_final_198_complete_20260809
_selected_for_use_20260809/coldstart_final_198_complete_20260809.zip
_selected_for_use_20260809/cold_start_selected_merged_198_rows.csv
```

The selected final package contains:

```text
results/cold_start_adaptation_results.csv
results/cold_start_support_summary.csv
reports/coldstart_final_198_local_audit.json
reports/coldstart_final_198_coverage_by_domain_seed.csv
reports/coldstart_final_198_summary_by_support.csv
reports/coldstart_final_198_summary_by_support_seed.csv
README_COLDSTART_FINAL_198.md
```

## Newly Completed Rows

Seed 1, domain J:

| support | accuracy | balanced accuracy | macro F1 |
|---:|---:|---:|---:|
| 0 | 0.318883 | 0.330757 | 0.264512 |
| 1 | 0.556296 | 0.605766 | 0.542263 |
| 5 | 0.691424 | 0.757624 | 0.738279 |
| 10 | 0.766587 | 0.790705 | 0.784838 |
| 25 | 0.959484 | 0.957998 | 0.957307 |
| 50 | 0.966708 | 0.966255 | 0.962138 |

Seed 1, domain K:

| support | accuracy | balanced accuracy | macro F1 |
|---:|---:|---:|---:|
| 0 | 0.476911 | 0.495743 | 0.448544 |
| 1 | 0.745382 | 0.739990 | 0.729007 |
| 5 | 0.815987 | 0.838182 | 0.829841 |
| 10 | 0.959860 | 0.957416 | 0.959179 |
| 25 | 0.988787 | 0.988939 | 0.988555 |
| 50 | 0.990085 | 0.988544 | 0.988788 |

These rows pass the basic ML-engineering sanity check: increasing support produces strong, plausible gains, with no missing or duplicate protocol rows.

## Manuscript-Ready Cold-Start Summary

Aggregated across all 33 domain-seed combinations:

| support | n | accuracy mean | accuracy SD | balanced accuracy mean | macro F1 mean |
|---:|---:|---:|---:|---:|---:|
| 0 | 33 | 0.382188 | 0.134463 | 0.390235 | 0.348879 |
| 1 | 33 | 0.484746 | 0.159008 | 0.500099 | 0.471705 |
| 5 | 33 | 0.665368 | 0.152759 | 0.675368 | 0.664979 |
| 10 | 33 | 0.736635 | 0.130427 | 0.749537 | 0.741950 |
| 25 | 33 | 0.850036 | 0.111075 | 0.857192 | 0.855502 |
| 50 | 33 | 0.919932 | 0.066278 | 0.923222 | 0.920664 |

This supports the paper claim that adaptation performance improves substantially as domain-specific support increases.

## Important Caveat

The checkpoint's packaged `reports/protocol_validity_audit.csv` is stale: it still says cold-start expected 198 and found 186, even though the actual `results/cold_start_adaptation_results.csv` in the domain-K checkpoint has 198 / 198 keys.

Use the local selected audit instead:

```text
_selected_for_use_20260809/coldstart_final_198_complete_20260809/reports/coldstart_final_198_local_audit.json
```

## Next Step

Do not rerun cold-start. It is complete.

Next, regenerate the final compact/publication evidence package from the selected 198-row cold-start CSV and the selected complete LODO evidence, so that the package-level `protocol_validity_audit.csv` is no longer stale. After that, update the manuscript tables/figures using:

```text
_selected_for_use_20260809/coldstart_final_198_complete_20260809/reports/coldstart_final_198_summary_by_support.csv
_selected_for_use_20260809/coldstart_final_198_complete_20260809/reports/coldstart_final_198_summary_by_support_seed.csv
```
