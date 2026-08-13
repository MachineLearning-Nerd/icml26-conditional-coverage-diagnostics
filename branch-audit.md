# Branch audit

This file is the public branch contract for
[`icml26-conditional-coverage-diagnostics`](https://github.com/MachineLearning-Nerd/icml26-conditional-coverage-diagnostics).
The historical `orx/` prefix encoded internal experiment-tree stages. Each
published branch now has a descriptive `audit/` or `integration/` name.

| Historical branch | Published branch | Purpose and evidence status |
| --- | --- | --- |
| `main` | `main` | Canonical cumulative audit, claim ledger, release notes, and current evidence. |
| `orx/baseline-pinned-uv-environment-and-source-faithf` | `audit/baseline-pinned-environment` | Pinned Python/uv environment and source-faithful release baseline. |
| `orx/claim-1-constant-target-principle-at-population` | `audit/c1-constant-target-population` | Population constant-target principle and exact conditional-coverage route; banked. |
| `orx/claim-1-rerun-under-the-tabpfn-pinned-environmen` | `audit/c1-tabpfn-rerun` | Claim-1 rerun under the pinned TabPFN-oriented environment. |
| `orx/claim-2-diamonds-experiments-5-9` | `audit/c2-diamonds` | Diamonds Table-2 experiment block; part of the blocked full seven-method protocol. |
| `orx/claim-2-food-delivery-time-experiments-5-9` | `audit/c2-food-delivery` | Food Delivery Time Table-2 experiment block; same blocker. |
| `orx/claim-2-physiochemical-protein-experiments-5-9` | `audit/c2-physiochemical-protein` | Physiochemical protein Table-2 experiment block; same blocker. |
| `orx/claim-2-probe-tabpfn-and-tabicl-on-cpu-cost-and` | `audit/c2-foundation-model-cost` | CPU cost probe for TabPFN/TabICL; documents why the literal comparison exceeds budget. |
| `orx/claim-2-superconductivity-experiments-5-9` | `audit/c2-superconductivity` | Superconductivity Table-2 experiment block; same blocker. |
| `orx/claim-2-table-2-protocol-on-diamonds-five-cpu-cl` | `audit/c2-table2-diamonds` | Five-CPU-class Table-2 protocol route for Diamonds. |
| `orx/claim-2-table-2-protocol-on-food-delivery-time-f` | `audit/c2-table2-food-delivery` | Five-CPU-class Table-2 protocol route for Food Delivery Time. |
| `orx/claim-2-table-2-protocol-on-physiochemical-prote` | `audit/c2-table2-protein` | Five-CPU-class Table-2 protocol route for physiochemical protein. |
| `orx/claim-2-table-2-protocol-on-superconductivity-fi` | `audit/c2-table2-superconductivity` | Five-CPU-class Table-2 protocol route for superconductivity. |
| `orx/claim-3-covgap-versus-l1-ert-convergence-regress` | `audit/c3-covgap-convergence` | Synthetic L1-ERT versus CovGap convergence evidence; banked. |
| `orx/claim-3-rerun-under-the-tabpfn-pinned-environmen` | `audit/c3-tabpfn-rerun` | Claim-3 rerun under the pinned TabPFN-oriented environment. |
| `orx/claim-4-asymmetric-over-under-coverage-decomposi` | `audit/c4-asymmetric-decomposition` | Additivity and asymmetric over/under-coverage foundation; banked. |
| `orx/claim-5-table-4-classification-decomposition-on` | `audit/c5-table4-classification` | Full Table-4 classification decomposition route; blocked by compute. |
| `orx/claim-5-table-4-classification-on-cifar10` | `audit/c5-cifar10` | CIFAR10 Table-4 attempt; predictor fits completed without an ERT cell. |
| `orx/claim-5-table-4-classification-on-fashionmnist` | `audit/c5-fashionmnist` | FashionMNIST Table-4 attempt; predictor fits completed without an ERT cell. |
| `orx/claim-5-table-4-classification-on-mnist` | `audit/c5-mnist` | MNIST Table-4 attempt; predictor fits completed without an ERT cell. |
| `orx/claim-6-algorithm-1-cross-validation-audit-and-i` | `audit/c6-algorithm1-cross-validation` | Independent five-seed Algorithm-1 audit; verified candidate evidence. |
| `orx/claim-6-rerun-under-the-tabpfn-pinned-environmen` | `audit/c6-tabpfn-rerun` | Claim-6 rerun under the pinned TabPFN-oriented environment. |
| `orx/full-claim-stage-suite` | `integration/full-claim-stage-suite` | Full six-claim suite and honest publication gate. |

## Canonical evidence routes

- C1: `repro/pipeline/stage_principle.py`, `repro/verify/claims.py`
- C2: `repro/src/aggregate_cpu_comparator_claim.py`, `repro/pipeline/data.py`
- C3: `repro/pipeline/stage_convergence.py`, `repro/src/aggregate_synthetic_convergence.py`
- C4: `repro/pipeline/stage_decomposition.py`, `repro/pipeline/metrics.py`
- C5: `repro/pipeline/stage_table4.py` and the dataset-specific branches
- C6: `repro/src/audit_claim6.py`, `outputs/claim6_independent_audit.json`
- Gate and release contract: `repro/src/audit_publication_gate.py`, `repro/src/publication_gate.py`

Every published branch carries this file and the canonical README. Older
branches are historical snapshots; `main` is the source of truth for the
current evidence contract. No public branch uses an `orx/` name.
