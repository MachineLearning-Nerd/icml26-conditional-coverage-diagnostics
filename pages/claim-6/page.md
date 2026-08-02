# Claim 6 - Algorithm 1 cross-validation

---
<!-- trackio-cell
{"type":"markdown","id":"claim6_outcome_first","created_at":"2026-08-03T00:00:00+00:00","title":"Claim 6 outcome and evidence","pinned":true,"pinned_at":"2026-08-03T00:00:00+00:00"}
-->
## VERIFIED — five-seed, full-size CPU reproduction

> Algorithm 1 estimates the ERT metrics from finite samples using k-fold cross-validation to avoid overfitting the classifier used in the estimation.

The paper explicitly says that the classifier must not be evaluated on rows used to train it, then gives the k-fold procedure in [Algorithm 1](https://arxiv.org/html/2512.11779v1#alg1). This reproduction tests both parts: the exact partitions and the anti-overfitting effect.

The preregistered checks all pass:

| Required observation | Result |
| --- | --- |
| Recorded folds exactly match independent `KFold(shuffle=True, random_state=42)` | PASS |
| Test folds are disjoint and cover every row once | PASS |
| Constant-target cross-fitted ERT is exactly zero | PASS |
| No-CV L1-ERT is above `0.02` at every size | PASS |
| Every cross-fitted L1-ERT is below `0.01` in magnitude and at least 5× smaller | PASS, all 12 `(size, k)` cells |

## Direct numerical result

True ERT is exactly zero in this oracle construction. Positive no-CV values are therefore spurious in-sample overfitting.

| Test rows | No CV | k=2 | k=3 | k=5 | k=10 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2,000 | +0.097280 | -0.002900 | -0.004521 | -0.009200 | -0.007020 |
| 10,000 | +0.097352 | +0.004048 | +0.001640 | +0.001652 | +0.000284 |
| 50,000 | +0.040688 | -0.000546 | +0.000349 | +0.000862 | +0.000190 |

Cross-fitting removes the false violation for every tested fold count and scale. The no-CV arm is a meaningful negative control: it fits and scores the same rows and fails exactly as the paper's warning predicts.

## Rerunnable evidence

- [Raw five-seed result](https://huggingface.co/spaces/DineshAI/vaApZm6MKM/blob/main/raw/claim6_algorithm1__raw.json), SHA-256 `59aeb547948d67117da86f28d3708c572ba296b45594bc3c656f5307e1843852`
- [Independent audit output](https://huggingface.co/spaces/DineshAI/vaApZm6MKM/blob/main/outputs/claim6_independent_audit.json), SHA-256 `2e08d3f03901cd97db882598cbaa2c70dcbf5aa62e930833af8d29ca9127483b`
- [Independent audit source](https://github.com/MachineLearning-Nerd/icml26-repro-vaApZm6MKM-conditional-coverage-diagnostics/blob/main/repro/src/audit_claim6.py)
- [Full reproduction source at exact commit](https://github.com/MachineLearning-Nerd/icml26-repro-vaApZm6MKM-conditional-coverage-diagnostics/tree/c6f68ec340b9e01a261a02e753666721bf210645/repro)
- [Completed Hugging Face CPU Job](https://huggingface.co/jobs/DineshAI/6a6c481323ed89c748ec92cd)

```bash
python3 repro/src/audit_claim6.py
```

The standard-library audit independently recomputes 270 stored mean/standard-deviation/SEM/CI fields from the per-seed rows, checks every contract condition, and fails closed. It also proves the checker is live by rejecting three in-memory mutations: a wrong partition, missing no-CV inflation, and failed cross-fitting. Two clean audit runs were byte-identical.

## Provenance and scope

| Field | Value |
| --- | --- |
| Full-run Git commit | `c6f68ec340b9e01a261a02e753666721bf210645` |
| Official experiments code | `ElSacho/Conditional_Coverage_Estimation@39a99dcad92205a15d93f2c5fec40c76540abf1c` |
| Official metric code | `ElSacho/covmetrics@a5205aada6a0f39e3812daf087753217ef66b159` |
| Seeds | 0, 1, 2, 3, 4 |
| Sizes | 2,000; 10,000; 50,000 |
| Folds | 2, 3, 5, 10 |
| Classifier | `CheapBetterLGBMClassifier` |
| Hardware | Hugging Face `cpu-upgrade`, 64 vCPU; no GPU |
| Completed stage runtime | 161.9 s wall, 133.4 s CPU |

The partition audit uses an instrumented constant predictor so exact row identities can be checked. The anti-overfitting experiment uses the paper's default classifier. The no-CV comparator is a diagnostic control created for this audit, not a method proposed by the paper.
