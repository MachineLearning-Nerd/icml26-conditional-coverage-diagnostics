# Claim evidence ledger

This ledger separates the paper's six official claims, the evidence actually
present in this repository, and the boundary of each verdict. A blocked claim
is not a falsification; a candidate verification is not a new evaluator score.

| Claim | Paper anchor | How evidence is produced | Controls and boundary | Status |
| --- | --- | --- | --- | --- |
| C1 | Section 3.1, Table 1, and the sentence supporting Algorithm 1 | `repro/pipeline/stage_principle.py` evaluates the released proper losses; `repro/verify/claims.py` checks the exact conditional-coverage construction and negative control. | Constant-target population principle, closed-form proper-loss checks, and source-faithful classifiers. The universal theorem remains the paper's proof. | `VERIFIED_SCOPED_LIVE` — live 2/2 retained |
| C2 | Table 2, Section 4.1, Appendix H Table 5 | `repro/src/aggregate_cpu_comparator_claim.py` and the dataset-specific `audit/c2-*` branches recover the denominator and CPU protocol. | The literal statistic needs seven methods, four datasets, ten repeats, and ten sizes; the full run is beyond the authorized two-hour CPU envelope. Reduced subsets are not substituted. | `BLOCKED_COMPUTE` |
| C3 | Figure 4, Section 4.2 | `repro/pipeline/stage_convergence.py` generates the released heteroscedastic scenarios and aggregates ten seeds. | The source release uses a different `sigma(x)` scale than the figure caption; the discrepancy is recorded. The audit retains the live 2/2 evidence and does not turn a finite sweep into a universal convergence proof. | `VERIFIED_SCOPED_LIVE` — live 2/2 retained |
| C4 | Section 3.3 | `repro/pipeline/stage_decomposition.py` and the pinned `covmetrics` route check additivity and isolated over/under-coverage constructions. | Exact finite construction, three proper scores, and source-faithful prediction-set routes. | `VERIFIED_SCOPED_LIVE` — live 2/2 retained |
| C5 | Table 4, Section 4.3.2 | `repro/pipeline/stage_table4.py` preserves the source route; `audit/c5-*` branches record the MNIST, FashionMNIST, and CIFAR10 attempts. | Predictor-only outputs do not produce the required ERT cells; the complete four-dataset, two-strategy, ten-repeat, five-fold protocol exceeds the authorized CPU envelope. | `BLOCKED_COMPUTE` |
| C6 | Algorithm 1 | `repro/src/audit_claim6.py` independently recomputes 270 summaries from the five-seed evidence, checks folds, and writes `outputs/claim6_independent_audit.json`. | Three mutations are rejected: wrong partition, missing no-CV inflation, and failed cross-fitting. This is candidate evidence; only an external evaluator can bank new points. | `VERIFIED_SCOPED_CANDIDATE` |

## Historical score boundary

The repository retains the historical live evaluator result of `6/12`, earned
from Claims 1, 3, and 4. Claim 6's local audit does not change that number, and
Claims 2 and 5 remain blocked rather than being scored by proxy.

## Shared evidence contract

The pinned experiment and metric revisions are recorded in
[`docs/SOURCE_AUDIT.md`](docs/SOURCE_AUDIT.md). The release gate and no-regression
checks are preserved in [`outputs/publication_manifest.json`](outputs/publication_manifest.json).
The machine-checkable summary is [`claims.json`](claims.json), and
[`verify_final.py`](verify_final.py) checks the public branch, identity, evidence,
and score boundaries.
