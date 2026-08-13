# Results

This repository audits [Conditional Coverage Diagnostics for Conformal
Prediction](https://arxiv.org/abs/2512.11779) by Sacha Braun, David Holzmüller,
Michael I. Jordan, and Francis Bach.

## Claim status

| Claim | Result | Evidence route | Interpretation |
| --- | --- | --- | --- |
| C1 | `VERIFIED` | Exact conditional-coverage population principle and released proper-loss checks | Retains the live 2/2 evidence. |
| C2 | `BLOCKED — >2h CPU` | Full seven-method Table-2 denominator recovered; complete faithful rerun not affordable in the authorized envelope | Neither verified nor falsified; no reduced subset is substituted. |
| C3 | `VERIFIED` | Ten-seed heteroscedastic synthetic L1-ERT/CovGap convergence audit | Retains the live 2/2 evidence and records the release-vs-caption scale discrepancy. |
| C4 | `VERIFIED` | Exact ERT additivity and conservative/aggressive decomposition controls | Retains the live 2/2 evidence. |
| C5 | `BLOCKED — >2h CPU` | Source-faithful Table-4 classification attempts produced predictors but no complete ERT cell | Neither verified nor falsified; partial predictor fits are excluded. |
| C6 | `VERIFIED` | Five-seed full-size Algorithm-1 cross-validation audit with independent checker | Candidate evidence; no new live judge score is claimed. |

The historical live result is **6/12** from the three banked claims. The local
additive gate does not change that score.

## CPU comparator result

The completed two-method CPU comparator is useful context but is deliberately
not presented as the paper's GPU-inclusive Table-2 percentage statistic:

| Method | Paired cells | Mean L1-ERT | SEM |
| --- | ---: | ---: | ---: |
| CheapBetterLGBMClassifier | 800 | 0.068375 | 0.001374 |
| PartitionWise | 800 | 0.030095 | 0.001245 |

LightGBM is higher in 759 of 800 paired cells. The literal paper statistic
requires all seven methods, four datasets, ten repeats, and all released
test-size cells; the full run was measured at roughly 55 CPU-box-hours.

## Claim 6 direct result

The true ERT in the oracle construction is zero. No-CV L1-ERT is `+0.097280`,
`+0.097352`, and `+0.040688` at 2,000, 10,000, and 50,000 test rows. Every
cross-fitted result for `k ∈ {2,3,5,10}` is below `0.01` in magnitude and at
least five times smaller across all 12 size/fold cells.

The independent standard-library audit recomputes 270 stored summary values,
checks exact fold partitions and coverage, and rejects mutations for a wrong
partition, missing no-CV inflation, and failed cross-fitting.

## Re-run

```bash
python3 repro/src/audit_claim6.py
```

The full gate is available when its source-scale artifacts are present:

```bash
uv run --with numpy python repro/src/audit_publication_gate.py \
  --foundations outputs/foundations/ert_foundations.json \
  --appendix-manifest outputs/full-cpu/appendix_h_integrity.json \
  --synthetic-summary outputs/synthetic/summary.json \
  --cpu-comparator-summary outputs/pinned-covmetrics/comparator_summary.json \
  --result outputs/publication_gate.json
```

See [`docs/SOURCE_AUDIT.md`](docs/SOURCE_AUDIT.md) for paper anchors,
quantifiers, pinned upstream commits, release defects, data provenance, and
the exact finite-versus-paper-scope boundary.
