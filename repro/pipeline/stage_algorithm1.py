"""Claim 6 — Algorithm 1's k-fold cross-validation, and what it is there for.

Paper (Algorithm 1): the ERT metrics are estimated from finite samples using
k-fold cross-validation *to avoid overfitting the classifier used in the
estimation*.

Two separable assertions, tested separately.

Partition audit.  The released estimator really does k-fold cross-fit: the
folds it hands the classifier are recorded through an instrumented model and
compared, index by index, with an independent `KFold(shuffle=True,
random_state=42)`; test folds must be disjoint and cover every row once, and
no row may appear in both the fit and the score set of the same fold.

Overfitting audit — the part that carries the claim's *purpose*.  On the
`oracle` construction the true ERT is 0, so any positive estimate is pure
overfitting.  Estimating without cross-fitting (fit and score the same rows)
must produce a large spurious violation, and cross-fitting must remove it.  We
sweep k in {2, 3, 5, 10} and several sample sizes so the effect is shown to be
a property of cross-fitting rather than of one configuration.  The no-CV arm is
the negative control: if it did *not* inflate, the claim would have no content.
"""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import KFold

from . import synthetic
from .classifiers import method_spec
from .emit import artifact, note
from .metrics import ert_pinned, excess_risk, mean_and_sem
from .metrics import INDEPENDENT_LOSSES

FOLD_COUNTS = (2, 3, 5, 10)


class _RecordingClassifier:
    """A constant 1-alpha predictor that reports the exact rows it was given."""

    records: list[dict] = []

    def fit(self, x, cover):
        self._train = sorted(int(v) for v in np.asarray(x)[:, 0])
        return self

    def predict_proba(self, x):
        rows = sorted(int(v) for v in np.asarray(x)[:, 0])
        self.records.append({"train": self._train, "test": rows})
        p = np.full(len(np.asarray(x)), 1.0 - synthetic.ALPHA)
        return np.column_stack([1.0 - p, p])


def partition_audit(n_rows: int = 5_000, n_splits: int = 5) -> dict:
    x = np.column_stack([np.arange(n_rows, dtype=float), np.linspace(-1.0, 1.0, n_rows)])
    cover = np.tile(np.array([1] * 9 + [0], dtype=np.int64), n_rows // 10)
    _RecordingClassifier.records = []
    values = ert_pinned(_RecordingClassifier, {}, x, cover, alpha=synthetic.ALPHA, n_splits=n_splits)
    observed = _RecordingClassifier.records
    expected = [
        {"train": sorted(train.tolist()), "test": sorted(test.tolist())}
        for train, test in KFold(n_splits=n_splits, shuffle=True, random_state=42).split(x)
    ]
    test_union: list[int] = []
    for fold in observed:
        test_union.extend(fold["test"])
        if set(fold["train"]) & set(fold["test"]):
            raise AssertionError("a fold scored rows it was fitted on")
    return {
        "n_rows": n_rows,
        "n_splits_requested": n_splits,
        "n_splits_observed": len(observed),
        "matches_independent_kfold": observed == expected,
        "test_folds_disjoint_and_complete": sorted(test_union) == list(range(n_rows)),
        "constant_target_cross_validated_ert": {k: v for k, v in values.items()},
    }


def _fit_score_no_cv(model_cls, kwargs, x, cover, alpha) -> dict:
    """The estimator Algorithm 1 replaces: fit and score the same rows."""
    model = model_cls(**kwargs)
    model.fit(x, cover)
    predictions = model.predict_proba(x)[:, 1]
    return {
        name: excess_risk(predictions, cover, alpha, loss, clip)
        for name, (loss, clip) in INDEPENDENT_LOSSES.items()
    }


def run(config: dict) -> dict:
    seeds = list(config.get("seeds", range(5)))
    sizes = list(config.get("sizes", [2_000, 10_000, 50_000]))
    method = config.get("method", "CheapBetterLGBMClassifier")
    alpha = synthetic.ALPHA

    note("claim 6: auditing the released estimator's cross-validation partitions")
    audit = partition_audit()

    per_seed: dict[str, dict] = {}
    for seed in seeds:
        payload = synthetic.constructions(seed)["oracle"]
        model_cls, kwargs = method_spec(method, len(payload["cover"]))
        per_size = {}
        for size_index, size in enumerate(sizes):
            indices = synthetic.subsample(seed, "algorithm1", size_index, len(payload["cover"]), size)
            x, cover = payload["features"][indices], payload["cover"][indices]
            arm = {"no_cv": _fit_score_no_cv(model_cls, kwargs, x, cover, alpha)}
            for k in FOLD_COUNTS:
                arm[f"kfold_{k}"] = ert_pinned(model_cls, kwargs, x, cover, alpha=alpha, n_splits=k)
            per_size[str(size)] = arm
            note(f"claim6 seed={seed} n={size} no-CV L1-ERT={arm['no_cv']['ERT_L1_miscoverage']:+.6f} "
                 f"5-fold L1-ERT={arm['kfold_5']['ERT_L1_miscoverage']:+.6f}")
        per_seed[str(seed)] = per_size

    summary = {}
    for size in sizes:
        summary[str(size)] = {}
        for arm in ("no_cv", *[f"kfold_{k}" for k in FOLD_COUNTS]):
            summary[str(size)][arm] = {
                loss_key: mean_and_sem([per_seed[str(s)][str(size)][arm][loss_key] for s in seeds])
                for loss_key in ("ERT_L1_miscoverage", "ERT_brier_score", "ERT_logloss")
            }

    result = {
        "claim": "6",
        "protocol": synthetic.PROTOCOL | {"seeds": seeds, "sizes": sizes, "method": method,
                                          "fold_counts": list(FOLD_COUNTS),
                                          "construction": "oracle (true ERT is exactly 0)"},
        "partition_audit": audit,
        "per_seed": per_seed,
        "summary": summary,
    }
    artifact("claim6_algorithm1/raw.json", result)
    return result
