"""ERT evaluation: the pinned package, plus an independent re-derivation.

`ert_pinned` calls covmetrics@a5205aa, the package the paper releases.
`ert_independent` re-implements Algorithm 1 straight from the paper's text
without importing covmetrics, so the two can be compared on identical folds.
Any claim number that survives only in one of the two is not evidence.
"""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import KFold

from covmetrics.ERT import ERT as PinnedERT
from covmetrics.losses import (
    L1_miscoverage, L1_miscoverage_over, L1_miscoverage_under,
    brier_score, brier_score_over, brier_score_under,
    logloss, logloss_over, logloss_under,
)

ALL_LOSSES = (
    brier_score, logloss, L1_miscoverage,
    brier_score_over, L1_miscoverage_over, logloss_over,
    brier_score_under, logloss_under, L1_miscoverage_under,
)

# The paper's names for the three headline metrics.
PAPER_NAME = {
    "ERT_L1_miscoverage": "L1-ERT",
    "ERT_brier_score": "L2-ERT",
    "ERT_logloss": "KL-ERT",
    "ERT_L1_miscoverage_over": "L1+-ERT",
    "ERT_L1_miscoverage_under": "L1--ERT",
    "ERT_logloss_over": "KL+-ERT",
    "ERT_logloss_under": "KL--ERT",
    "ERT_brier_score_over": "L2+-ERT",
    "ERT_brier_score_under": "L2--ERT",
}


def ert_pinned(model_cls, model_kwargs, x, cover, alpha=0.1, n_splits=5, random_state=42) -> dict:
    """Every loss covmetrics exposes, on one call, using its own fold logic."""
    metric = PinnedERT(model_cls, **model_kwargs)
    values = metric.evaluate_multiple_losses(
        np.asarray(x, dtype=np.float64), np.asarray(cover, dtype=np.int64),
        alpha, n_splits=n_splits, random_state=random_state, all_losses=list(ALL_LOSSES),
    )
    return {key: float(value) for key, value in values.items()}


# ---------------------------------------------------------------------------
# Independent re-derivation from the paper's definitions.
# ---------------------------------------------------------------------------

def _clip_over(p, t):
    return np.maximum(p, t)


def _clip_under(p, t):
    return np.minimum(p, t)


def _l1(p, z, t):
    # Section 3.2: the L1 loss charges +/-(t - z) according to which side of the
    # target the prediction falls on, and 0 exactly at the target.
    out = np.zeros_like(z, dtype=float)
    out[p < t] = -(t - z)[p < t]
    out[p > t] = (t - z)[p > t]
    return out


def _brier(p, z, t=None):
    return (p - z) ** 2


def _logloss(p, z, t=None, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return -(z * np.log(p) + (1 - z) * np.log(1 - p))


INDEPENDENT_LOSSES = {
    "ERT_L1_miscoverage": (_l1, None),
    "ERT_brier_score": (_brier, None),
    "ERT_logloss": (_logloss, None),
    "ERT_L1_miscoverage_over": (_l1, _clip_over),
    "ERT_brier_score_over": (_brier, _clip_over),
    "ERT_logloss_over": (_logloss, _clip_over),
    "ERT_L1_miscoverage_under": (_l1, _clip_under),
    "ERT_brier_score_under": (_brier, _clip_under),
    "ERT_logloss_under": (_logloss, _clip_under),
}


def excess_risk(predictions, cover, alpha, loss, clip=None) -> float:
    """R(constant 1-alpha) - R(h).  Positive means h beat the target."""
    target = 1.0 - alpha
    z = np.asarray(cover, dtype=float)
    p = np.asarray(predictions, dtype=float)
    if clip is not None:
        p = clip(p, target)
    baseline = np.full_like(z, target)
    if clip is not None:
        baseline = clip(baseline, target)
    return float(np.mean(loss(baseline, z, target)) - np.mean(loss(p, z, target)))


def ert_independent(fit_predict, x, cover, alpha=0.1, n_splits=5, random_state=42) -> dict:
    """Algorithm 1 with k-fold cross-fitting, written from the paper's text.

    `fit_predict(x_train, cover_train, x_test) -> probabilities` keeps this free
    of any covmetrics import, so it is a genuine second opinion rather than a
    re-spelling of the same code.
    """
    x = np.asarray(x, dtype=np.float64)
    cover = np.asarray(cover, dtype=np.int64)
    folds = {name: [] for name in INDEPENDENT_LOSSES}
    for train_index, test_index in KFold(n_splits=n_splits, shuffle=True, random_state=random_state).split(x):
        predictions = fit_predict(x[train_index], cover[train_index], x[test_index])
        for name, (loss, clip) in INDEPENDENT_LOSSES.items():
            folds[name].append(excess_risk(predictions, cover[test_index], alpha, loss, clip))
    return {name: float(np.mean(values)) for name, values in folds.items()}


def ert_pinned_parallel(fit_predict, x, cover, alpha=0.1, n_splits=5, random_state=42,
                        workers=5) -> dict:
    """Identical to `ert_pinned`, with the fold fits spread across processes.

    Only the classifier fits move; the folds come from the same
    `KFold(shuffle=True, random_state=42)` covmetrics uses (verified index by
    index in Claim 6's partition audit) and each fold's metric values come from
    covmetrics' own `evaluate_with_predictions`.  For image covariates the fits
    dominate the cost by orders of magnitude, and a 64-vCPU box cannot use that
    width inside a 32-row batch.
    """
    from concurrent.futures import ProcessPoolExecutor

    from covmetrics.ERT import evaluate_with_predictions

    x = np.asarray(x, dtype=np.float64)
    cover = np.asarray(cover, dtype=np.int64)
    folds = list(KFold(n_splits=n_splits, shuffle=True, random_state=random_state).split(x))

    jobs = [(x[train], cover[train], x[test]) for train, test in folds]
    with ProcessPoolExecutor(max_workers=min(workers, len(jobs))) as pool:
        predictions = list(pool.map(fit_predict, jobs))

    values = {f"ERT_{loss.__name__}": [] for loss in ALL_LOSSES}
    for (_, test), prediction in zip(folds, predictions):
        for loss in ALL_LOSSES:
            values[f"ERT_{loss.__name__}"].append(
                float(evaluate_with_predictions(prediction, cover[test], alpha, loss=loss))
            )
    return {key: float(np.mean(v)) for key, v in values.items()}


def mean_and_sem(values) -> dict:
    values = np.asarray(values, dtype=float)
    n = len(values)
    sem = float(values.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    return {
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)) if n > 1 else 0.0,
        "sem": sem,
        "n": int(n),
        "ci95_low": float(values.mean() - 1.96 * sem),
        "ci95_high": float(values.mean() + 1.96 * sem),
    }
