# Conditional Coverage Diagnostics for Conformal Prediction

Source-faithful, CPU-first reproduction for ICML 2026 challenge paper
`vaApZm6MKM` (arXiv `2512.11779`).

## Pinned source

- Experiment repository: `ElSacho/Conditional_Coverage_Estimation` at
  `39a99dcad92205a15d93f2c5fec40c76540abf1c` (camera-ready).
- Metric package: `ElSacho/covmetrics` at
  `a5205aada6a0f39e3812daf087753217ef66b159`.
- The complete upstream checkouts are vendored under `upstream/`.

## Published reproduction evidence

The local publication gate passed with five distinct source-anchored claims:

1. ERT construction and the constant-target principle.
2. A direct CPU LightGBM-versus-PartitionWise L1-ERT comparison over the eight
   named TabArena datasets, ten seeds, ten test sizes, and five-fold ERT.
3. A ten-seed high-dimensional synthetic comparison in which L1-ERT separates
   standard and oracle conformal constructions more reliably than CovGap.
4. The over-/under-coverage decomposition.
5. Algorithm-1-style five-fold held-out ERT evaluation.

The full result summary and exact commands are in [`RESULTS.md`](RESULTS.md).
`repro/src/audit_publication_gate.py` is fail-closed and writes
`outputs/publication_gate.json` only after all five claims and their required
protocol evidence validate. The CPU comparator is deliberately scoped to two
released CPU blocks; it is **not** presented as a reproduction of the paper's
GPU-inclusive percentage table.
