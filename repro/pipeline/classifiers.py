"""The seven Table-2 ERT classifiers, transcribed from the pinned release.

`experiments_classifier_benchmark/code/networks.py` at
`39a99dcad92205a15d93f2c5fec40c76540abf1c` defines these classes, and
`_generate_simultaneous_experiments.py` constructs the `ERT(...)` wrappers.
Five of the seven construction blocks are commented out in the release and its
batch scripts call a script that is absent from the repository (see
docs/RELEASE_AUDIT.md), so the benchmark cannot be re-run unmodified.  The
class bodies and the ERT keyword arguments below are copied verbatim from the
release; only the surrounding driver is re-implemented.
"""

from __future__ import annotations

import multiprocessing as mp

import numpy as np
import sklearn
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from probmetrics.calibrators import get_calibrator
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.cluster import KMeans
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier


class CheapBetterLGBMClassifier(sklearn.base.BaseEstimator, sklearn.base.ClassifierMixin):
    """networks.py lines 65-90 — Table 2 row "LightGBM (medium)"."""

    def _fit_model(self, idxs):
        m = LGBMClassifier(n_estimators=1_000, learning_rate=0.04, subsample=0.75, subsample_freq=1,
                           num_leaves=50, random_state=0, early_stopping_round=100, min_child_samples=40,
                           min_child_weight=1e-7, n_jobs=1, verbosity=-1)
        return m.fit(self.X_.take(idxs[0], 0), self.y_[idxs[0]],
                     eval_set=(self.X_.take(idxs[1], 0), self.y_[idxs[1]]))

    def fit(self, X, y):
        self.le_ = sklearn.preprocessing.LabelEncoder().fit(y)
        self.X_, self.y_, self.classes_ = X, self.le_.transform(y), self.le_.classes_
        splits = list(sklearn.model_selection.StratifiedKFold(n_splits=8, shuffle=True, random_state=0).split(X, y))
        with mp.Pool(processes=min(len(splits), mp.cpu_count())) as pool:
            self.models_ = pool.map(self._fit_model, splits)
        oof_preds = np.concatenate([m.predict_proba(X.take(idxs[1], 0)) for m, idxs in zip(self.models_, splits)], axis=0)
        oof_labels = np.concatenate([self.y_[idxs[1]] for idxs in splits], axis=0)
        self.calib_ = get_calibrator('logistic', calibrate_with_mixture=True,
                                     logistic_binary_type='quadratic').fit(oof_preds, oof_labels)
        return self

    def predict_proba(self, X):
        return self.calib_.predict_proba(np.mean([m.predict_proba(X) for m in self.models_], axis=0))

    def predict(self, X):
        return self.le_.inverse_transform(np.argmax(self.predict_proba(X), axis=1))


class BetterCatBoostClassifier(sklearn.base.BaseEstimator, sklearn.base.ClassifierMixin):
    """networks.py lines 116-160 — Table 2 row "CatBoost"."""

    def __init__(self, iterations=10000, early_stopping_rounds=300, thread_count=1, random_state=0):
        self.iterations = iterations
        self.early_stopping_rounds = early_stopping_rounds
        self.thread_count = thread_count
        self.random_state = random_state

    def _fit_model(self, idxs):
        m = CatBoostClassifier(iterations=self.iterations, learning_rate=None, random_state=self.random_state,
                               early_stopping_rounds=self.early_stopping_rounds,
                               thread_count=self.thread_count, verbose=False)
        return m.fit(self.X_.take(idxs[0], 0), self.y_[idxs[0]],
                     eval_set=(self.X_.take(idxs[1], 0), self.y_[idxs[1]]))

    def fit(self, X, y):
        self.le_ = sklearn.preprocessing.LabelEncoder().fit(y)
        self.X_, self.y_, self.classes_ = X, self.le_.transform(y), self.le_.classes_
        splits = list(sklearn.model_selection.StratifiedKFold(n_splits=8, shuffle=True, random_state=0).split(X, y))
        with mp.Pool(processes=min(len(splits), mp.cpu_count())) as pool:
            self.models_ = pool.map(self._fit_model, splits)
        oof_preds = np.concatenate([m.predict_proba(X.take(idxs[1], 0)) for m, idxs in zip(self.models_, splits)], axis=0)
        oof_labels = np.concatenate([self.y_[idxs[1]] for idxs in splits], axis=0)
        self.calib_ = get_calibrator('logistic', calibrate_with_mixture=True,
                                     logistic_binary_type='quadratic').fit(oof_preds, oof_labels)
        return self

    def predict_proba(self, X):
        return self.calib_.predict_proba(np.mean([m.predict_proba(X) for m in self.models_], axis=0))

    def predict(self, X):
        return self.le_.inverse_transform(np.argmax(self.predict_proba(X), axis=1))


class PartitionWisePredictor(BaseEstimator, ClassifierMixin):
    """networks.py lines 364-454 — the estimator underlying CovGap."""

    def __init__(self, n_clusters=3, verbose=False, **kwargs):
        self.n_clusters = n_clusters
        self.clusterer = KMeans(n_clusters=self.n_clusters, random_state=42, n_init='auto')
        self.cluster_means_ = {}
        self.verbose = verbose

    def fit(self, X, y):
        X, y = np.asarray(X), np.asarray(y)
        if X.shape[0] < self.n_clusters:
            self.global_mean_ = y.mean()
            self.clusterer = None
            return self
        cluster_labels = self.clusterer.fit_predict(X)
        self.cluster_means_ = {}
        for i in range(self.n_clusters):
            y_partition = y[np.where(cluster_labels == i)[0]]
            self.cluster_means_[i] = y_partition.mean() if len(y_partition) else 0.5
        return self

    def predict(self, X):
        if self.clusterer is None:
            return np.full(np.asarray(X).shape[0], self.global_mean_)
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)

    def predict_proba(self, X):
        X = np.asarray(X)
        if self.clusterer is None:
            return np.full((X.shape[0], 2), [1 - self.global_mean_, self.global_mean_])
        assignments = self.clusterer.predict(X)
        proba_1 = np.zeros(X.shape[0], dtype=np.float64)
        for i in range(self.n_clusters):
            index = np.where(assignments == i)[0]
            if len(index):
                proba_1[index] = self.cluster_means_.get(i, 0.5)
        return np.column_stack([1.0 - proba_1, proba_1])


# Exactly the keyword arguments the release driver passes to ERT(...).  The
# PartitionWise cluster count follows the release's fourth-root rule, which is
# taken over the size of the *full* test split rather than the subsample.
CPU_METHODS = ("CheapBetterLGBMClassifier", "BetterCatBoost", "RF", "XT", "PartitionWise")
FOUNDATION_MODEL_METHODS = ("TabPFN", "tabICL")
ALL_METHODS = CPU_METHODS + FOUNDATION_MODEL_METHODS


def method_spec(name: str, n_test_full: int) -> tuple[type, dict]:
    if name == "CheapBetterLGBMClassifier":
        return CheapBetterLGBMClassifier, {}
    if name == "BetterCatBoost":
        return BetterCatBoostClassifier, dict(iterations=1000, early_stopping_rounds=200,
                                              thread_count=2, random_state=42)
    if name == "XT":
        return ExtraTreesClassifier, dict(n_estimators=300, max_depth=None, random_state=0, n_jobs=-1)
    if name == "RF":
        return RandomForestClassifier, dict(n_estimators=300, max_depth=None, random_state=42, n_jobs=-1)
    if name == "PartitionWise":
        return PartitionWisePredictor, dict(n_clusters=max(1, int(n_test_full ** (1 / 4))))
    if name == "TabPFN":
        from tabpfn import TabPFNClassifier
        return TabPFNClassifier, dict(device="cpu", ignore_pretraining_limits=True)
    if name == "tabICL":
        from tabicl import TabICLClassifier
        return TabICLClassifier, dict(device="cpu")
    raise KeyError(name)
