"""Claim 2 — the Table-2 relative-power percentages for LightGBM and PartitionWise.

Paper (Table 2, Section 4.1): with L1-ERT, LightGBM (medium) recovers
68.4 +/- 2.2 percent of the maximum and PartitionWise 38.3 +/- 1.9 percent.

The percentage is not defined in the body text, so it is taken from the release
that produced the table, `results/see_pourcentage_improvment.ipynb`:

    1. negative ERT values are clipped to 0 and missing values filled with 0;
    2. within each (dataset, experiment) the maximum is taken over *all methods
       and all test-set sizes* jointly;
    3. each cell becomes 100 * value / that maximum;
    4. average over datasets and sizes within each (method, experiment);
    5. report mean and standard deviation of those ten per-experiment averages.

Running that notebook's formula over the release's own committed
`results.csv`, restricted to the four Table-5 datasets, returns LightGBM 68.2
and PartitionWise 38.6 against the table's 68.4 and 38.3 — which is how the
formula and the dataset restriction were confirmed before any new compute was
spent.  That check uses the authors' numbers and is calibration of the
statistic, not evidence; the evidence is the regeneration below.

Protocol, from `_generate_simultaneous_experiments.py`:
40/10/50 split at `random_state=experiment`, RealMLP-TD-S as the mean
predictor, nonconformity |Y - f(X)|, alpha = 0.1, conformal threshold at index
ceil((n+1)(1-alpha)) of the sorted calibration scores, ten log-spaced test
sizes from 300 to the full test split, and five-fold ERT.

The normalisation is over whichever methods a run actually evaluates, so a
CPU-only run and the paper's seven-method run are not the same statistic.  The
`methods` list is recorded with every number and the verifier refuses to
compare against 68.4/38.3 unless the seven-method set was used.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import data
from .classifiers import ALL_METHODS, method_spec
from .emit import artifact, note, row
from .metrics import ert_pinned
from .provenance import Timer

ALPHA = 0.1
N_SIZES = 10
PAPER_TABLE2_L1 = {
    "CheapBetterLGBMClassifier": {"label": "LightGBM (medium)", "mean": 68.4, "std": 2.2},
    "PartitionWise": {"label": "PartitionWise", "mean": 38.3, "std": 1.9},
    "BetterCatBoost": {"label": "CatBoost", "mean": 68.7, "std": 2.8},
    "XT": {"label": "ExtraTrees", "mean": 65.9, "std": 2.4},
    "RF": {"label": "RandomForest", "mean": 65.9, "std": 2.8},
    "TabPFN": {"label": "RealTabPFN-2.5", "mean": 71.6, "std": 1.7},
    "tabICL": {"label": "TabICLv1.1", "mean": 71.9, "std": 1.9},
}


def split_and_conformalize(x, y, experiment: int) -> dict:
    """The release's 40/10/50 split, RealMLP predictor and split-conformal step."""
    from pytabkit import RealMLP_TD_S_Regressor
    from sklearn.model_selection import train_test_split

    x_train, x_rest, y_train, y_rest = train_test_split(x, y, test_size=0.6, random_state=experiment)
    x_calibration, x_test, y_calibration, y_test = train_test_split(
        x_rest, y_rest, test_size=5 / 6, random_state=experiment
    )

    with Timer() as timer:
        # device="cpu" is made explicit; leaving it unset makes pytabkit pick an
        # accelerator, which this campaign is not permitted to use.
        model = RealMLP_TD_S_Regressor(n_cv=5, val_metric_name="brier", verbosity=False, device="cpu")
        model.fit(x_train, y_train)

    def scores(features, targets):
        predictions = np.asarray(model.predict(features)).reshape(len(features), -1)
        return np.linalg.norm(np.asarray(targets).reshape(len(targets), -1) - predictions, ord=2, axis=1)

    calibration_scores = np.sort(scores(x_calibration, y_calibration))
    index = int(np.ceil((len(y_calibration) + 1) * (1 - ALPHA)))
    threshold = float(calibration_scores[index])
    cover = (scores(x_test, y_test) < threshold).astype(np.int64)

    return {
        "x_test": x_test, "cover": cover, "threshold": threshold,
        "n_train": int(len(y_train)), "n_calibration": int(len(y_calibration)),
        "n_test": int(len(y_test)), "marginal_coverage": float(cover.mean()),
        "predictor_fit_s": round(timer.wall_s, 2),
    }


def test_sizes(n_test: int) -> list[int]:
    return np.round(np.logspace(np.log10(300), np.log10(n_test), N_SIZES)).astype(int).tolist()


class _Experiment:
    """One (dataset, experiment) cell, run start to finish in its own process.

    Experiments are fully independent - their own split, their own predictor,
    their own subsamples - and the RealMLP predictor takes about twenty minutes
    per experiment on this hardware, so running several at once is the
    difference between a job that fits its timeout and one that does not.
    Threads stay capped inside each worker so the total stays near the core
    count.
    """

    def __init__(self, dataset: str, methods: list[str], size_limit, threads: int):
        self.dataset, self.methods, self.size_limit, self.threads = \
            dataset, methods, size_limit, threads

    def __call__(self, experiment: int) -> list[dict]:
        import torch
        torch.set_num_threads(self.threads)

        features, targets = _DATASET_CACHE[self.dataset]
        np.random.seed(experiment)
        fitted = split_and_conformalize(features, targets, experiment)
        note(f"table2 {self.dataset} exp={experiment} split="
             f"{fitted['n_train']}/{fitted['n_calibration']}/{fitted['n_test']} "
             f"coverage={fitted['marginal_coverage']:.4f} "
             f"predictor={fitted['predictor_fit_s']}s")

        sizes = test_sizes(fitted["n_test"])
        if self.size_limit:
            sizes = [s for s in sizes if s <= int(self.size_limit)]

        records = []
        for size in sizes:
            # The release draws each subsample from the global numpy RNG, seeded
            # once per experiment, so the draw order is preserved.
            idx = np.random.choice(fitted["n_test"], size=size, replace=False)
            x_subset, cover_subset = fitted["x_test"][idx], fitted["cover"][idx]
            for method in self.methods:
                model_cls, kwargs = method_spec(method, fitted["n_test"])
                with Timer() as timer:
                    values = ert_pinned(model_cls, kwargs, x_subset, cover_subset, alpha=ALPHA)
                record = {"dataset": self.dataset, "experiment": int(experiment),
                          "nsamples": int(size), "method": method,
                          "time": round(timer.wall_s, 4),
                          "marginal_coverage": fitted["marginal_coverage"], **values}
                records.append(record)
                row("table2", record)
                note(f"table2 {self.dataset} exp={experiment} n={size} {method} "
                     f"L1-ERT={values['ERT_L1_miscoverage']:+.6f} ({timer.wall_s:.1f}s)")
        return records


_DATASET_CACHE: dict[str, tuple] = {}


def run(config: dict) -> dict:
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor

    datasets = list(config.get("datasets", data.TABLE2_DATASETS))
    experiments = list(config.get("experiments", range(10)))
    methods = list(config.get("methods", ALL_METHODS))
    size_limit = config.get("max_test_size")
    workers = int(config.get("experiment_workers", 5))
    threads = int(config.get("worker_threads", 8))

    rows: list[dict] = []
    integrity: dict[str, dict] = {}
    for name in datasets:
        loaded = data.load(name)
        integrity[name] = loaded["integrity"]
        note(f"table2 dataset={name} {loaded['integrity']['rows']} rows "
             f"(Appendix H {loaded['integrity']['appendix_h_rows']})")
        _DATASET_CACHE[name] = (data.encode(loaded["x"]), loaded["y"].to_numpy())

        worker = _Experiment(name, methods, size_limit, threads)
        if workers <= 1:
            for experiment in experiments:
                rows.extend(worker(experiment))
        else:
            context = mp.get_context("fork")
            with ProcessPoolExecutor(max_workers=min(workers, len(experiments)),
                                     mp_context=context) as pool:
                for records in pool.map(worker, experiments):
                    rows.extend(records)

    result = {"claim": "2", "protocol": {
        "alpha": ALPHA, "split": "40/10/50", "predictor": "RealMLP_TD_S_Regressor(n_cv=5, "
        "val_metric_name='brier', device='cpu')", "score": "|Y - f(X)|",
        "ert_folds": 5, "n_sizes": N_SIZES, "datasets": datasets,
        "experiments": experiments, "methods": methods,
        "source_driver": "_generate_simultaneous_experiments.py",
        "statistic_source": "results/see_pourcentage_improvment.ipynb",
    }, "dataset_integrity": integrity, "rows": rows}
    artifact(f"claim2_table2/raw_{'_'.join(datasets)}.json", result)
    return result


# ---------------------------------------------------------------------------
# The Table-2 statistic itself, kept separate so it can be applied to rows from
# several runs and to the release's own CSV with identical code.
# ---------------------------------------------------------------------------

METRIC_COLUMNS = ("ERT_L1_miscoverage", "ERT_brier_score", "ERT_logloss")


def table2_statistic(frame: pd.DataFrame) -> pd.DataFrame:
    """Reproduce `see_pourcentage_improvment.ipynb` exactly."""
    frame = frame.copy()
    for column in METRIC_COLUMNS:
        frame[column] = frame[column].fillna(0).clip(lower=0)
    maxima = (frame.groupby(["dataset", "experiment"])[list(METRIC_COLUMNS)].max()
              .rename(columns={c: f"max_{c}" for c in METRIC_COLUMNS}))
    frame = frame.merge(maxima, on=["dataset", "experiment"])
    percent_columns = []
    for column in METRIC_COLUMNS:
        frame[f"pct_{column}"] = frame[column] / frame[f"max_{column}"] * 100
        percent_columns.append(f"pct_{column}")
    per_experiment = frame.groupby(["method", "experiment"], as_index=False)[percent_columns].mean()
    return per_experiment.groupby("method")[percent_columns].agg(["mean", "std"])
