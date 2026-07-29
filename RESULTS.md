# Reproduction results

This CPU-only reproduction of *Conditional Coverage Diagnostics for Conformal
Prediction* (OpenReview `vaApZm6MKM`, arXiv `2512.11779`) passed the local,
fail-closed five-claim publication gate.

## Evidence scope

The evidence consists of deterministic ERT foundation checks, a ten-seed
released high-dimensional synthetic replay, an eight-dataset Appendix-H
integrity manifest, and a direct pinned-`covmetrics` comparator replay.

The comparator uses the released 40/10/50 split, `alpha=0.1`, five ERT folds,
CPU execution, ten random seeds, and ten released test sizes for each of
eight datasets. It evaluates `ERT_L1_miscoverage` directly through
`covmetrics@a5205aada6a0f39e3812daf087753217ef66b159`.

| Method | Paired cells | Mean L1-ERT | SEM |
|---|---:|---:|---:|
| CheapBetterLGBMClassifier | 800 | 0.068375 | 0.001374 |
| PartitionWise | 800 | 0.030095 | 0.001245 |

LightGBM is higher in 759 of 800 paired cells (PartitionWise: 41; ties: 0).
This is a direct two-CPU-comparator result, not a substitute for the paper's
GPU-inclusive Table-2 percentage statistic.

At released synthetic test size 4,843, the standard-conformal L1-ERT estimate
has absolute error 0.004283 against the retained truth, compared with CovGap
error 0.082749; for the oracle construction, L1-ERT error is 0.000434 versus
CovGap error 0.009607. The observed L1-ERT separation is 0.092799, compared
with 0.004291 for CovGap.

## Re-run

The complete gate can be re-run after the underlying outputs exist:

```bash
uv run --with numpy python repro/src/audit_publication_gate.py \
  --foundations outputs/foundations/ert_foundations.json \
  --appendix-manifest outputs/full-cpu/appendix_h_integrity.json \
  --synthetic-summary outputs/synthetic/summary.json \
  --cpu-comparator-summary outputs/pinned-covmetrics/comparator_summary.json \
  --result outputs/publication_gate.json
```

The gate output reports `publication_eligible: true` and `claim_count: 5`.
