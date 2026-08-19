# Reproduction report

## Executive result

This repository is a partial, source-faithful reproduction audit of
*Conditional Coverage Diagnostics for Conformal Prediction*. Claims 1, 3, and
4 retain the historical live evaluator evidence (`2/2` each). Claim 6 has a
new independent CPU audit that passes its fixed criteria. Claims 2 and 5 remain
honest compute blockers because the literal paper protocols require more than
the authorized CPU budget.

Overall status:

```text
PARTIAL_C1_C3_C4_LIVE_VERIFIED_C6_CPU_VERIFIED_C2_C5_BLOCKED_HISTORICAL_SCORE_6_OF_12_NO_CURRENT_SCORE
```

## Claim 6 result

The oracle construction has true ERT zero. Without cross-fitting, the recorded
L1-ERT means are `0.097280`, `0.097352`, and `0.040688` at 2,000, 10,000, and
50,000 rows. With `k ∈ {2, 3, 5, 10}` cross-fitting, every stored mean is below
`0.01` in magnitude and at least five times smaller. The independent audit
recomputes 270 values, checks complete disjoint folds, and rejects three broken
controls.

## Blocked claims

Claim 2's percentage is normalized over all seven methods, four datasets, ten
repeats, and ten test sizes. A two-method CPU comparator is retained as context,
but it is not relabeled as the paper's percentage. The measured full matrix is
approximately 55 CPU-box-hours.

Claim 5 requires four datasets, two strategies, ten repeats, and five-fold ERT.
The CPU attempts produced predictors but no complete ERT cells; predictor-only
results are excluded. CIFAR10 predictor fits alone took roughly 8,233–9,812
seconds, and CIFAR100 adds ResNet-18 training inside every fold.

## Score and publication boundary

The historical live score is `6/12`. No current score, forecast, publication
approval, or author endorsement is claimed. The local additive gate passing
means the evidence package is structurally eligible for review; it does not
bank a new evaluator point.
