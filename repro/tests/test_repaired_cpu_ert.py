import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "repro" / "src"))

from repaired_cpu_ert import PartitionWisePredictor


def test_partitionwise_returns_binary_probability_pairs():
    features = pd.DataFrame({"x": np.arange(16), "z": np.arange(16) % 3})
    labels = np.array([0, 1] * 8)
    model = PartitionWisePredictor(n_clusters=2).fit(features, labels)
    probabilities = model.predict_proba(features)
    assert probabilities.shape == (16, 2)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert set(model.predict(features)).issubset({0, 1})
