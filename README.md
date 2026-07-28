# Conditional Coverage Diagnostics for Conformal Prediction

Source-faithful, CPU-first reproduction for ICML 2026 challenge paper
`vaApZm6MKM` (arXiv `2512.11779`).

## Pinned source

- Experiment repository: `ElSacho/Conditional_Coverage_Estimation` at
  `39a99dcad92205a15d93f2c5fec40c76540abf1c` (camera-ready).
- Metric package: `ElSacho/covmetrics` at
  `a5205aada6a0f39e3812daf087753217ef66b159`.
- The complete upstream checkouts are vendored under `upstream/`.

## Anchored claim map

1. ERT construction and its constant-predictor principle.
2. CPU LightGBM statistical-power comparison on the eight named TabArena
   datasets, with ten seeds and five-fold ERT estimation.
3. Synthetic reliability comparison with CovGap at the released test sizes.
4. Over-/under-coverage decomposition.
5. Source-faithful classification/CP-strategy decomposition results.
6. Algorithm 1 cross-validated ERT estimator.

No claim is verified by this scaffold. The exact TabArena data version and
row/column counts must be recovered before any full source execution.
