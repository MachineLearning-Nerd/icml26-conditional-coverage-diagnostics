"""Minimal runtime repair for the released CPU ERT comparison blocks.

The camera-ready classifier driver cannot import because it loads every
GPU-oriented optional comparator and a missing local ``ERT.py``.  This module
implements only the two CPU class definitions that are present-but-commented
in that driver, and imports the unmodified in-repository ERT implementation from
``experiments_general/code``.  It is a disclosed release-repair wrapper, not a
claim that the released command runs unchanged.
"""

from __future__ import annotations

import multiprocessing as mp
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import sklearn
from lightgbm import LGBMClassifier
from probmetrics.calibrators import get_calibrator
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.cluster import KMeans


UPSTREAM = Path(__file__).resolve().parents[2] / "upstream" / "Conditional_Coverage_Estimation"
ERT_DIR = UPSTREAM / "experiments" / "experiments_general" / "code"
if str(ERT_DIR) not in sys.path:
    sys.path.insert(0, str(ERT_DIR))
from ERT import ERT  # noqa: E402


class CheapBetterLGBMClassifier(sklearn.base.BaseEstimator, sklearn.base.ClassifierMixin):
    """Source-derived CPU class from upstream classifier ``networks.py`` lines 65-91."""

    def _fit_model(self, idxs):
        m = LGBMClassifier(n_estimators=1_000, learning_rate=0.04, subsample=0.75, subsample_freq=1, num_leaves=50,
                           random_state=0, early_stopping_round=100, min_child_samples=40, min_child_weight=1e-7,
                           n_jobs=1, verbosity=0)
        return m.fit(self.X_.take(idxs[0], 0), self.y_[idxs[0]], eval_set=(self.X_.take(idxs[1], 0), self.y_[idxs[1]]))

    def fit(self, X, y):
        self.le_ = sklearn.preprocessing.LabelEncoder().fit(y)
        self.X_, self.y_, self.classes_ = X, self.le_.transform(y), self.le_.classes_
        splits = list(sklearn.model_selection.StratifiedKFold(n_splits=8, shuffle=True, random_state=0).split(X, y))
        with mp.Pool(processes=min(len(splits), mp.cpu_count())) as pool:
            self.models_ = pool.map(self._fit_model, splits)
        oof_preds = np.concatenate([m.predict_proba(X.take(idxs[1], 0)) for m, idxs in zip(self.models_, splits)], axis=0)
        oof_labels = np.concatenate([y[idxs[1]] for idxs in splits], axis=0)
        self.calib_ = get_calibrator('logistic', calibrate_with_mixture=True,
                                     logistic_binary_type='quadratic').fit(oof_preds, oof_labels)
        return self

    def predict_proba(self, X):
        return self.calib_.predict_proba(np.mean([m.predict_proba(X) for m in self.models_], axis=0))

    def predict(self, X):
        return self.le_.inverse_transform(np.argmax(self.predict_proba(X), axis=1))


class PartitionWisePredictor(BaseEstimator, ClassifierMixin):
    """Source-derived CPU class from upstream classifier ``networks.py`` lines 364-454."""

    def __init__(self, n_clusters=3, verbose=False, **kwargs):
        self.n_clusters = n_clusters
        self.clusterer = KMeans(n_clusters=self.n_clusters, random_state=42, n_init='auto')
        self.cluster_means_ = {}
        self.verbose = verbose

    def fit(self, X, y):
        if X.shape[0] < self.n_clusters:
            self.global_mean_ = y.mean()
            self.clusterer = None
            return self
        cluster_labels = self.clusterer.fit_predict(X)
        self.cluster_means_ = {}
        for i in range(self.n_clusters):
            partition_indices = np.where(cluster_labels == i)[0]
            y_partition = y[partition_indices]
            self.cluster_means_[i] = y_partition.mean() if len(y_partition) else 0.5
        return self

    def predict(self, X):
        if self.clusterer is None:
            return np.full(X.shape[0], self.global_mean_)
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)

    def predict_proba(self, X):
        if self.clusterer is None:
            return np.full((X.shape[0], 2), [1 - self.global_mean_, self.global_mean_])
        assignments = self.clusterer.predict(X)
        y_proba_1 = np.zeros(X.shape[0], dtype=np.float64)
        for i in range(self.n_clusters):
            index = np.where(assignments == i)[0]
            if len(index):
                y_proba_1[index] = self.cluster_means_.get(i, 0.5)
        return np.column_stack([1.0 - y_proba_1, y_proba_1])


def evaluate_cpu_methods(features: pd.DataFrame, coverage: pd.Series, alpha: float = 0.1) -> dict:
    """Run the source ERT five-fold protocol for the two CPU Table-2 methods."""
    # The upstream driver generates an array-like coverage vector. Preserve that
    # positional contract even when callers supply a pandas Series after a split.
    coverage = np.asarray(coverage)
    methods = {
        "CheapBetterLGBMClassifier": ERT(CheapBetterLGBMClassifier),
        "PartitionWise": ERT(PartitionWisePredictor, n_clusters=max(1, int(len(coverage) ** 0.25))),
    }
    return {name: method.evaluate_all(features, coverage, alpha, n_splits=5) for name, method in methods.items()}
