# Source audit

Everything the reproduction asserts about the paper traces to one of the
artefacts below. Nothing is quoted from memory.

## Paper identity

The paper is [Conditional Coverage Diagnostics for Conformal Prediction](https://arxiv.org/abs/2512.11779)
by Sacha Braun, David Holzmüller, Michael I. Jordan, and Francis Bach. The ICML
submission identifier used by the evaluator is `vaApZm6MKM`.

```bibtex
@article{braun2025conditional,
  title         = {Conditional Coverage Diagnostics for Conformal Prediction},
  author        = {Braun, Sacha and Holzm{\"u}ller, David and Jordan, Michael I. and Bach, Francis},
  journal       = {arXiv preprint arXiv:2512.11779},
  year          = {2025},
  doi           = {10.48550/arXiv.2512.11779}
}
```

## Paper

| Field | Value |
| --- | --- |
| arXiv | `2512.11779` |
| OpenReview | `vaApZm6MKM` |
| Retrieved | 2026-07-31T05:06:34Z |
| URL | `https://arxiv.org/html/2512.11779v1` |
| SHA-256 | `1e13e76905dab485726ab80b671ca7140c73e58cdc6001832379f4a349da5a03` |
| Mirror | `https://ar5iv.labs.arxiv.org/html/2512.11779`, SHA-256 `cb2a1922e19a616f1ea47197a1c470a26674146c0a60c8cad215d6d9d6f849b2` |
| Abstract page | `https://arxiv.org/abs/2512.11779`, SHA-256 `e8d9e319637d05252d82eda22d8f35c3cda16db6ab8151456270a16d66b9fab9` |

Fetched with an explicit browser User-Agent; all three retrievals returned
HTTP 200.

## Code

| Repository | Commit | Role |
| --- | --- | --- |
| `ElSacho/covmetrics` | `a5205aada6a0f39e3812daf087753217ef66b159` | the released metric package; every ERT value in this reproduction comes from it |
| `ElSacho/Conditional_Coverage_Estimation` | `39a99dcad92205a15d93f2c5fec40c76540abf1c` | the experiment release; classifier definitions, generators and drivers are transcribed from it |

`covmetrics` is installed from that exact commit through `uv.lock`, so the
metric implementation is pinned rather than vendored.

## Exact quantifiers, by claim

### Claim 1 — Section 3.1

> "under conditional coverage, given a proper score ℓ, no classifier can
> achieve a lower risk than the constant predictor 1 − α."

Universal over classifiers and over the three proper scores. The assumption is
exact conditional coverage, `P(Y ∈ C_α(X) | X = x) = 1 − α` almost everywhere —
not marginal coverage.

### Claim 2 — Table 2, Section 4.1, Appendix H Table 5

The table caption defines the statistic: "ERT recovered by different methods,
relative to the highest value among all methods and number of samples, averaged
over all number of test samples and datasets. Experiments are repeated 10
times, and the index number is the standard deviation across those 10
experiments."

Section 4.1 fixes the protocol: the four largest TabArena regression datasets,
40 % train / 10 % calibration / 50 % test, `S(X,Y) = |Y − f(X)|`, `1 − α = 0.9`,
five-fold cross-validation, averaged over ten runs.

Appendix H Table 5 names the four datasets and their sizes:

| Dataset | Samples | Test samples | Features |
| --- | ---: | ---: | ---: |
| physiochemical_protein | 45,730 | 22,865 | 9 |
| Food_Delivery_Time | 45,593 | 22,797 | 10 |
| diamonds | 53,940 | 26,970 | 9 |
| superconductivity | 21,263 | 10,632 | 81 |

The seven methods are TabICLv1.1, RealTabPFN-2.5, CatBoost, LightGBM (medium),
ExtraTrees, RandomForest and PartitionWise. The two figures under discussion
are 68.4₂.₂ for LightGBM and 38.3₁.₉ for PartitionWise in the L1-ERT column.

The body text never writes the percentage's formula down. It is recovered from
`experiments_classifier_benchmark/results/see_pourcentage_improvment.ipynb` in
the release, which is the notebook that produced the table.

### Claim 3 — Figure 4, Section 4.2

> "Even with 5,000 points, they provide nearly identical diagnostics across
> these two very different scenarios."

The two scenarios are standard split conformal with `S(X,Y) = |Y|` on 3,000
calibration samples, and oracle sets from the true conditional `α/2` and
`1 − α/2` quantiles. `Y ~ N(0, σ(X¹))`, `X ~ U([−1,1]⁸)`. The release's
generator uses `σ(x) = 0.5 + |2x| + x²`; the paper's figure caption writes
`σ(x) = 0.5 + |x| + x²`. This reproduction follows the release code and records
the discrepancy.

Theoretical values are computed from the true `P(Y ∈ C_α(X) | X)` over 300,000
samples.

### Claim 4 — Section 3.3

The over- and under-coverage components are `ℓ₊-ERT` and `ℓ₋-ERT`. In the
release these are the `ERT_underconfident_*` and `ERT_overconfident_*` fields,
which clip predictions from below and above at `1 − α` respectively — that is,
covmetrics' `*_over` and `*_under` losses.

### Claim 5 — Table 4, Section 4.3.2

Two strategies: negative likelihood `S(X,Y) = −p(X)_Y` (sadinle2019) and
cumulative likelihood (romano2020, angelopoulos2020). Four datasets: MNIST,
FashionMNIST, CIFAR10, CIFAR100. Reported columns are L1-ERT, KL-ERT, KL₊-ERT
and KL₋-ERT.

The paper's stated mechanism: the likelihood strategy produces more empty
prediction sets, whose conditional coverage is zero, and KL weights those more
heavily than L1, so that strategy "yields a larger value of KL₋-ERT than
KL₊-ERT."

### Claim 6 — Algorithm 1

Inputs are `{(Xᵢ, Zᵢ)}`, a fold count `k ≥ 2`, a proper score, a level `α` and a
classification method. The stated purpose of the cross-validation is to avoid
overfitting the classifier used in the estimation.

## Release defects that shape what can be re-run

These are recorded because they determine how much of the reproduction can be a
direct rerun rather than a reconstruction.

1. The classifier benchmark's batch files call
   `_generate_simultaneous_experiments_csv.py`, which is not in the repository.
2. The driver that is present, `_generate_simultaneous_experiments.py`, imports
   a local `ERT` module that is absent from its directory; an equivalent exists
   under `experiments_general/code/ERT.py`.
3. Five of the seven Table-2 method blocks in that driver are commented out,
   including LightGBM and PartitionWise — the two methods this campaign's
   Claim 2 is about — even though the committed `results.csv` contains their
   output.
4. `data/` is git-ignored and no preprocessing script ships, so the raw sources
   have to be recovered independently.
5. The requirements omit `numba`, which the released `probmetrics` calibration
   import now needs.
6. `RealMLP_TD_S_Regressor(device=None)` auto-selects an accelerator. The CPU
   path is made explicit with `device="cpu"`.

Consequently the Table-2 and Table-4 protocols here are **transcriptions** of
the release's driver logic around the release's own class definitions and the
pinned `covmetrics` metric implementation — not a one-command rerun. The class
bodies and every ERT keyword argument are copied verbatim; only the driver
scaffolding is rewritten.

## Data provenance

All four Table-5 datasets are fetched from OpenML at run time, with no
credentials, so the whole protocol regenerates from the fixed command.

| Dataset | OpenML | Rows observed | Appendix H | Match |
| --- | ---: | ---: | ---: | --- |
| physiochemical_protein | 42903 | 45,730 | 45,730 | exact |
| diamonds | 42225 | 53,940 | 53,940 | exact |
| superconductivity | 43174 | 21,263 | 21,263 | exact |
| Food_Delivery_Time | 46928 | 45,451 | 45,593 | −142 rows (−0.31 %) |

Food_Delivery_Time is the only deviation. Appendix H used the original Kaggle
v1 file, which cannot be downloaded without an account; OpenML 46928 is the
TabArena curation of it and is 142 rows smaller with 9 predictors rather than
10. Every number this dataset contributes to carries that note.

## Known determinism limits

`ExtraTrees` and `RandomForest` are constructed with `n_jobs=-1`, so their
fitted trees depend on the number of worker processes. Runs on machines with
different core counts reproduce their ERT values to about 1e-3, not bitwise.
This affects the two weakest Table-2 rows and is recorded rather than hidden;
all other classifiers are deterministic given the seed.
