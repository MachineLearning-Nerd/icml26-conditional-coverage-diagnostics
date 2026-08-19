# ICML 2026 — Conditional Coverage Diagnostics

Independent reproduction and claim audit for [Conditional Coverage Diagnostics
for Conformal Prediction](https://arxiv.org/abs/2512.11779).

| Field | Value |
| --- | --- |
| Paper | [arXiv:2512.11779](https://arxiv.org/abs/2512.11779) |
| Authors | Sacha Braun, David Holzmüller, Michael I. Jordan, Francis Bach |
| ICML submission | `vaApZm6MKM` |
| Repository | [MachineLearning-Nerd/icml26-conditional-coverage-diagnostics](https://github.com/MachineLearning-Nerd/icml26-conditional-coverage-diagnostics) |
| Historical live result | `6/12` from Claims 1, 3, and 4; no new live score is claimed |
| Current local status | Claim 6 `VERIFIED`; Claims 2 and 5 `BLOCKED — >2h CPU` |

## Audit record

- Overall status: `PARTIAL_C1_C3_C4_LIVE_VERIFIED_C6_CPU_VERIFIED_C2_C5_BLOCKED_HISTORICAL_SCORE_6_OF_12_NO_CURRENT_SCORE`
- Current score claim: none; `6/12` is historical evaluator context only.
- Publication gate: local additive gate passed; publication and author endorsement are not claimed.
- Branch contract: 23 descriptive branches including `main`; no public `orx/*` branch remains.
- History: all 174 reachable pre-dossier commits use `MachineLearning-Nerd <MachineLearning-Nerd@users.noreply.github.com>`.
- Canonical machine-checkable records: [claims.json](claims.json), [reproduction_verdicts.json](reproduction_verdicts.json), [EVIDENCE_MANIFEST.json](EVIDENCE_MANIFEST.json), and [verify_final.py](verify_final.py).

## What the paper does

The paper reframes conditional-coverage diagnosis as a classification problem.
Its excess risk of the target coverage (ERT) metrics compare a target coverage
predictor with a classifier under proper losses. The paper studies the
population constant-target principle, practical ERT power, convergence against
CovGap, asymmetric over/under-coverage decomposition, classification examples,
and the cross-validation procedure used to prevent estimator overfitting.

The repository is a source-faithful audit, not an official implementation. It
pins the authors' experiment release
[`ElSacho/Conditional_Coverage_Estimation@39a99dc`](https://github.com/ElSacho/Conditional_Coverage_Estimation/tree/39a99dcad92205a15d93f2c5fec40c76540abf1c)
and metric package
[`ElSacho/covmetrics@a5205aa`](https://github.com/ElSacho/covmetrics/tree/a5205aada6a0f39e3812daf087753217ef66b159).
Where the release is incomplete or too expensive to rerun, the limitation is
recorded as `BLOCKED`; a reduced experiment is not relabeled as the paper's
claim.

## Claim-to-evidence ledger

| Claim | Paper result and source anchor | How this repository produces evidence | Status |
| --- | --- | --- | --- |
| C1 | Under exact conditional coverage, no classifier improves the risk of the constant `1 − α` predictor for a proper score. Section 3.1. | `repro/pipeline/stage_principle.py` evaluates the released losses and an independent population grid; `repro/verify/claims.py` and the foundation artifact check the constant-target and negative-control constructions. | `VERIFIED` — live 2/2 evidence retained |
| C2 | Table 2 reports relative L1-ERT power for seven methods across four datasets, ten repeats, and test-size ladders. Section 4.1, Appendix H. | `repro/src/aggregate_cpu_comparator_claim.py` and the Claim-2 branches audit the recovered denominator and run the released CPU blocks. The full seven-method protocol needs foundation-model runs beyond the two-hour envelope, so no subset is used as the paper's number. | `BLOCKED — >2h CPU` |
| C3 | At the paper's synthetic scale, L1-ERT gives a more reliable diagnostic than CovGap for standard and oracle constructions. Figure 4, Section 4.2. | `repro/pipeline/stage_convergence.py` generates the released heteroscedastic scenarios, keeps the true conditional miscoverage, and aggregates ten seeds at the released sizes; the source audit records the figure/release `σ(x)` discrepancy. | `VERIFIED` — live 2/2 evidence retained |
| C4 | ERT decomposes conditional error into asymmetric over- and under-coverage components. Section 3.3. | `repro/pipeline/stage_decomposition.py` and the ERT foundation call the pinned metric and independently check additivity on conservative and aggressive constructions. | `VERIFIED` — live 2/2 evidence retained |
| C5 | Table 4 demonstrates divergent KL-plus/KL-minus ERT behavior across classification conformal strategies. Section 4.3.2. | `repro/pipeline/stage_table4.py` preserves the source classifier/strategy route, while the Claim-5 branches record MNIST, FashionMNIST, and CIFAR10 attempts. No complete ERT cell was produced within the allowed CPU budget, so predictor-only results are not counted. | `BLOCKED — >2h CPU` |
| C6 | Algorithm 1 uses held-out k-fold evaluation to avoid fitting/scoring the same rows. | `repro/src/audit_claim6.py` independently recomputes 270 summary values, verifies exact folds, compares no-CV against k-fold ERT for five seeds and `k ∈ {2,3,5,10}`, and rejects three mutations. | `VERIFIED` — candidate evidence; judge score unchanged |

## Evidence boundary

Claims 1, 3, and 4 retain the evidence already banked by the live evaluator.
Claim 6 is a new deterministic CPU audit and is not presented as a new live
score. Claims 2 and 5 are honest blockers, not failures of the paper and not
successes of a smaller proxy. The publication manifest records the additive
gate and the no-regression checks for the retained pages.

The historical live result is kept for transparency. Only a future external
judge can change it.

## Reproduce

The smallest independent current audit is Claim 6:

```bash
python3 repro/src/audit_claim6.py
```

The full additive gate requires the committed full-scale artifacts:

```bash
uv run --with numpy python repro/src/audit_publication_gate.py \
  --foundations outputs/foundations/ert_foundations.json \
  --appendix-manifest outputs/full-cpu/appendix_h_integrity.json \
  --synthetic-summary outputs/synthetic/summary.json \
  --cpu-comparator-summary outputs/pinned-covmetrics/comparator_summary.json \
  --result outputs/publication_gate.json
```

The release-repair protocol is described in [`docs/SOURCE_AUDIT.md`](docs/SOURCE_AUDIT.md),
[`docs/METHOD.md`](docs/METHOD.md), and [`docs/RELEASE_AUDIT.md`](docs/RELEASE_AUDIT.md).
The full claim pages and raw Space evidence are linked from [`RESULTS.md`](RESULTS.md).

## Branch organization

`main` is the canonical cumulative audit. The descriptive branch map in
[`branch-audit.md`](branch-audit.md) records the old experiment-tree lineage,
the purpose of every branch, and the current status of each route. Published
names use `audit/` for claim or protocol work and `integration/` for cumulative
assembly; the old `orx/` names are removed from the public interface.

## Citation

```bibtex
@article{braun2025conditional,
  title         = {Conditional Coverage Diagnostics for Conformal Prediction},
  author        = {Braun, Sacha and Holzm{\"u}ller, David and Jordan, Michael I. and Bach, Francis},
  journal       = {arXiv preprint arXiv:2512.11779},
  year          = {2025},
  doi           = {10.48550/arXiv.2512.11779}
}
```

## Thank you

Thank you to Sacha Braun, David Holzmüller, Michael I. Jordan, and Francis Bach
for making the paper, metric definitions, experiment release, and benchmark
structure available. That transparency made it possible to separate exact
source audits, reproducible CPU evidence, honest blockers, and historical
judge results instead of collapsing them into one score.

This repository is an independent reproduction maintained by
**MachineLearning-Nerd**. It is not the authors' official implementation and
does not imply author endorsement.

## Attribution

The paper, research contribution, theorem statements, released upstream code,
and `covmetrics` implementation belong to their respective authors and
maintainers. The audit scripts, evidence packaging, branch cleanup, and
documentation in this repository are the independent work of
**MachineLearning-Nerd**.
