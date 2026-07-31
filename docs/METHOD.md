# Method

## One command, one environment, one contract

Every node of the experiment tree runs exactly:

```bash
bash repro/run.sh
```

It takes no arguments. What a node computes is decided by the committed file
`repro/config/stage.json` on that node's branch, so variants live in code and
configuration rather than in the command line or in environment variables.

The environment is a single repository-level `uv` project: Python 3.11 pinned
by `pyproject.toml`, all 86 packages pinned by `uv.lock`, `covmetrics` installed
directly from commit `a5205aa`. `run.sh` bootstraps `uv` itself inside the job
image (which ships Python but not `uv`) and then `uv sync --frozen` does the
rest. No `conda`, no unmanaged `pip`, no per-claim environments.

## Evidence transport

In orx local mode the run log is the only channel results come back through.
Every artifact is therefore written to `.openresearch/artifacts/` *and* printed
to stdout inside a delimited block carrying its SHA-256, and long stages stream
one row per finished cell so a truncated job still leaves everything it
completed. `repro/pipeline/collect.py` rebuilds the artifact tree from captured
logs and re-checks every digest.

## Two implementations of the same metric

`repro/pipeline/metrics.py` holds both the pinned `covmetrics` call and an
independent re-derivation of Algorithm 1 written from the paper's definitions
with no `covmetrics` import. The smoke stage runs both on identical folds and
fails the run unless they agree to 1e-9. On the baseline run they agreed
exactly (max |delta| = 0.0), so a number appearing in only one of them would be
caught rather than published.

## The four synthetic constructions

Claims 1, 3, 4 and 6 all run on the Section 4.2 generator — `f*(x) = 0`,
`σ(x) = 0.5 + |2x| + x²`, `x₁ ~ U[-1,1]` plus seven nuisance features, allocation
30,000 / 3,000 / 3,000 / 300,000, `α = 0.1` — because it is the one setting where
the true conditional coverage `p(x)` is known in closed form and every claim can
be scored against truth rather than against another estimate.

| Construction | Half-width | True `p(x)` | Role |
| --- | --- | --- | --- |
| `standard_cp` | split-conformal radius from 3,000 calibration points | 0.658 … 1.000 | two-sided violation; Claim 3 scenario A; Claim 1 negative control |
| `oracle` | `z_{1-α/2}·σ(x)` | exactly 0.900 | Claim 1's assumption regime; Claim 4 and 6 negative control |
| `conservative` | `(1.05 … 1.60)·z_{1-α/2}·σ(x)` | 0.916 … 0.992 | strictly over-covering, and *conditionally* so |
| `aggressive` | `(0.95 … 0.60)·z_{1-α/2}·σ(x)` | 0.676 … 0.882 | strictly under-covering, and conditionally so |

The scale factors ramp with `x₁` on purpose. A constant factor would produce a
constant `p(x) ≠ 1 − α`, which is a marginal miscalibration; nothing about
*localising* conservatism would then be under test.

## Handling the universal quantifier in Claim 1

"No classifier can achieve a lower risk" quantifies over all measurable
`h: X → [0,1]`, which no finite run can settle. Risk decomposes pointwise, so
under conditional coverage the population excess risk is

```
ERT(h) = E_X[ R_t(t) − R_t(h(X)) ] = − E_X[ d_ℓ(t, h(X)) ]
```

and `ERT(h) ≤ 0` for every `h` if and only if `d_ℓ(t, p) ≥ 0` on the whole unit
square. That is a two-dimensional statement over a complete bounded domain, so
it is swept exhaustively at 1,001² = 1,002,001 grid points, with the divergence
built by calling the released loss functions rather than a hand-simplified
formula, and cross-checked against closed forms derived by hand.

This settles the quantifier at the population level. The finite-sample arm then
shows that the paper's *estimator* respects it at full scale with the strongest
CPU classifiers available.

## Negative controls

Every claim carries a control that must fail for the intended reason, and each
verifier fails the run if its control passes:

| Claim | Control | Must show |
| --- | --- | --- |
| 1 | `standard_cp` under the same estimator | strictly positive L1-ERT, CI above 0 |
| 3 | the oracle scenario alongside the invalid one | CovGap unable to tell them apart |
| 4 | `oracle` under the same decomposition | neither component above 0.01 |
| 5 | the additivity identity, plus the empty-set mechanism | exact additivity and the stated direction |
| 6 | the same estimator with no cross-fitting | a large spurious violation on exactly-conditional data |

Claim 6's control is the sharpest: on data whose true ERT is exactly zero,
fitting and scoring the same rows reads L1-ERT ≈ +0.10 at n = 2,000 and still
+0.045 at n = 50,000, while five-fold cross-fitting reads +0.0008. A control
that passed for every implementation would be no control at all; this one fails
precisely when the cross-validation is removed.

## Thresholds

Fixed before the runs, and anchored to a scale the paper sets rather than to
observed values:

- `NEGLIGIBLE = 0.01` — one tenth of the true L1-ERT (0.0965) of the paper's own
  conditionally-invalid construction. Below this, an estimate cannot be read as
  detecting that violation.
- `DETECTED = 0.05` — half of it. A genuine violation must read at least this
  high or the estimator is not sensitive enough for any of these claims.
- `TABLE2_TOL = 5.0` percentage points — Table 2's own run-to-run standard
  deviations are 1.9 to 2.8, so about two of them is the most a ten-repeat
  rerun can assert.

## What is transcribed and what is reconstructed

The Table-2 classifier bodies, every `ERT(...)` keyword argument, the split
fractions, the conformal quantile index, the test-size ladders, the
classification drivers' epoch counts and the synthetic generator are copied
verbatim from the release. The driver scaffolding around them is rewritten,
because the release's own batch entry point is missing from the repository and
five of the seven Table-2 method blocks are commented out —
see [`SOURCE_AUDIT.md`](SOURCE_AUDIT.md). This is a release-repair
reproduction, not a one-command rerun of the authors' script, and it is
labelled that way everywhere it appears.
