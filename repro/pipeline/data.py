"""The four Table-5 regression datasets, fetched from OpenML without credentials.

The release ignores `data/` and ships no preprocessing script, so the raw
sources have to be recovered.  Every dataset here is public on OpenML, so the
whole Table-2 protocol re-runs from the fixed command with no Kaggle account
and no manual download.  The only transformation applied is target selection
and renaming to the `target` column the release driver expects.

Row counts are checked against Appendix H Table 5 and the observed count is
recorded either way, because one source does not match exactly:

    physiochemical_protein  45730   OpenML 42903 (CASP)              exact
    diamonds                53940   OpenML 42225                     exact
    superconductivity       21263   OpenML 43174                     exact
    Food_Delivery_Time      45593   OpenML 46928 (TabArena curation) 45451

The Food Delivery source is the TabArena curation of the original Kaggle file.
It is 142 rows (0.31%) smaller than Appendix H, which used the raw Kaggle v1
file; that file is not downloadable without credentials, so a credential-free
reproduction cannot use it.  The deviation is reported with every number this
dataset contributes to.
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path

import numpy as np
import pandas as pd

CACHE = Path("data/openml")

DATASETS = {
    "physiochemical_protein": {"openml_id": 42903, "target": "RMSD",
                               "appendix_h_rows": 45730, "appendix_h_features": 9},
    "Food_Delivery_Time": {"openml_id": 46928, "target": "Time_taken(min)",
                           "appendix_h_rows": 45593, "appendix_h_features": 10},
    "diamonds": {"openml_id": 42225, "target": "price",
                 "appendix_h_rows": 53940, "appendix_h_features": 9},
    "superconductivity": {"openml_id": 43174, "target": "criticaltemp",
                          "appendix_h_rows": 21263, "appendix_h_features": 81},
}

# The four largest TabArena regression datasets, in the order Table 5 lists them.
TABLE2_DATASETS = tuple(DATASETS)


def _frame_digest(frame: pd.DataFrame) -> str:
    payload = pd.util.hash_pandas_object(frame, index=False).values.tobytes()
    payload += "|".join(map(str, frame.columns)).encode()
    return hashlib.sha256(payload).hexdigest()


def load(name: str) -> dict:
    """Return the release's `target`-column contract plus an integrity record."""
    from sklearn.datasets import fetch_openml

    spec = DATASETS[name]
    CACHE.mkdir(parents=True, exist_ok=True)
    cached = CACHE / f"{name}.parquet"
    if cached.exists():
        frame = pd.read_parquet(cached)
    else:
        # Several dataset shards start at once and OpenML rate-limits them with
        # a 503, which is a transport failure rather than a result.  Back off
        # rather than letting it end a ten-hour job in four seconds.
        last_error: Exception | None = None
        for attempt in range(8):
            try:
                frame = fetch_openml(data_id=spec["openml_id"], as_frame=True, parser="auto").frame
                break
            except Exception as error:  # noqa: BLE001 - re-raised below if terminal
                last_error = error
                delay = min(300, 15 * 2 ** attempt)
                print(f"[repro] openml {spec['openml_id']} attempt {attempt + 1}/8 failed "
                      f"({type(error).__name__}); retrying in {delay}s", flush=True)
                time.sleep(delay)
        else:
            raise RuntimeError(f"OpenML {spec['openml_id']} unreachable after 8 attempts") from last_error
        frame.to_parquet(cached)

    target_column = spec["target"]
    if target_column not in frame.columns:
        candidates = [c for c in frame.columns if c.lower().replace("_", "") ==
                      target_column.lower().replace("_", "")]
        if not candidates:
            raise KeyError(f"{name}: target {target_column!r} not among {list(frame.columns)}")
        target_column = candidates[0]

    frame = frame.rename(columns={target_column: "target"})
    frame = frame.dropna().reset_index(drop=True)
    y = pd.to_numeric(frame["target"], errors="raise").astype(float)
    x = frame.drop(columns=["target"])

    integrity = {
        "dataset": name,
        "openml_id": spec["openml_id"],
        "target_column": target_column,
        "rows": int(len(frame)),
        "predictors": int(x.shape[1]),
        "appendix_h_rows": spec["appendix_h_rows"],
        "appendix_h_features": spec["appendix_h_features"],
        "rows_match_appendix_h": int(len(frame)) == spec["appendix_h_rows"],
        "features_match_appendix_h": int(x.shape[1]) == spec["appendix_h_features"],
        "row_deviation_pct": round(100 * (len(frame) - spec["appendix_h_rows"])
                                   / spec["appendix_h_rows"], 4),
        "sha256": _frame_digest(frame),
    }
    return {"x": x, "y": y, "integrity": integrity}


def encode(x: pd.DataFrame) -> np.ndarray:
    """The release's preprocessing: ordinal-encode object columns, nothing else."""
    from sklearn.preprocessing import OrdinalEncoder

    x = x.copy()
    categorical = x.select_dtypes(include=["object", "category"]).columns
    if len(categorical):
        x[categorical] = OrdinalEncoder(dtype=float).fit_transform(x[categorical].astype(str))
    return x.astype(float).to_numpy()
