"""The paper's Section 4.2 heteroscedastic generator and its four constructions.

Transcribed from `experiments_general/code/_generate_fig_evol.py` and
`_generate_fig_evol_under_over.py` at 39a99dc: f*(x) = 0, sigma(x) = 0.5 +
|2x| + x^2, x1 ~ U[-1,1] plus seven U[-1,1] nuisance features, allocation
30,000 / 3,000 / 3,000 / 300,000, alpha = 0.1.

Four prediction-set constructions share that data:

  standard_cp   score |Y - f*(X)| calibrated on 3,000 points.  Marginally
                valid, conditionally invalid; over-covers where sigma is small
                and under-covers where sigma is large.
  oracle        the true conditional alpha/2 and 1-alpha/2 quantiles.  Exactly
                conditionally valid, so p(x) == 1 - alpha for every x.
  conservative  the oracle interval scaled by a factor > 1.  Over-covers
                everywhere, under-covers nowhere.
  aggressive    the oracle interval scaled by a factor < 1.  Under-covers
                everywhere, over-covers nowhere.

`oracle` is the assumption-satisfying regime for the Claim 1 principle and the
negative control for Claim 4; `conservative` and `aggressive` are the one-sided
regimes that make the Claim 4 decomposition falsifiable; `standard_cp` is the
two-sided regime and the Claim 1 negative control.
"""

from __future__ import annotations

import zlib

import numpy as np
from scipy.stats import norm
import torch

ALPHA = 0.1
TARGET = 1.0 - ALPHA
N_TRAIN, N_STOP, N_CALIBRATION, N_TEST = 30_000, 3_000, 3_000, 300_000
N_FAKE_FEATURES = 7
CONSERVATIVE_SCALE = 1.60
AGGRESSIVE_SCALE = 0.60

PROTOCOL = {
    "generator": "f*(x)=0; sigma(x)=0.5+|2*x1|+x1^2; x1~U[-1,1] plus 7 U[-1,1] nuisance features",
    "allocation": {"train": N_TRAIN, "stop": N_STOP, "calibration": N_CALIBRATION, "test": N_TEST},
    "alpha": ALPHA,
    "ert_folds": 5,
    "ert_random_state": 42,
    "source_drivers": ["_generate_fig_evol.py", "_generate_fig_evol_under_over.py"],
    "conservative_scale": CONSERVATIVE_SCALE,
    "aggressive_scale": AGGRESSIVE_SCALE,
}


def seed_everything(seed: int) -> None:
    """The release's seeding, reproduced so the draw order matches."""
    np.random.seed(seed)
    torch.manual_seed(seed)


def _sigma(x1: np.ndarray) -> np.ndarray:
    return 0.5 + np.abs(2 * x1) + x1 ** 2


def _generate(n: int) -> tuple[np.ndarray, np.ndarray]:
    x = ((torch.rand(n, 1) - 0.5) * 2).numpy()
    sigma = _sigma(x[:, 0])
    y = (sigma * torch.randn(n).numpy())
    return x, y


def draw(seed: int) -> dict:
    """Reproduce the release's draw order, including the unused train/stop sets."""
    seed_everything(seed)
    x_train, _ = _generate(N_TRAIN)
    x_stop, _ = _generate(N_STOP)
    x_calibration, y_calibration = _generate(N_CALIBRATION)
    x_test, y_test = _generate(N_TEST)

    def widen(x: np.ndarray) -> np.ndarray:
        return np.hstack([x, np.random.uniform(-1, 1, size=(len(x), N_FAKE_FEATURES))])

    # The train/stop widenings are retained because they advance the numpy RNG
    # exactly as the release does before the calibration and test features.
    widen(x_train)
    widen(x_stop)
    return {
        "x_calibration": widen(x_calibration), "y_calibration": y_calibration,
        "x_test": widen(x_test), "y_test": y_test,
    }


def constructions(seed: int) -> dict[str, dict]:
    """Cover vectors and the analytically known true conditional coverage p(x)."""
    data = draw(seed)
    y_calibration, x_test, y_test = data["y_calibration"], data["x_test"], data["y_test"]
    sigma_test = _sigma(x_test[:, 0])

    # Split conformal with the standard finite-sample quantile index.
    index = int(np.ceil((len(y_calibration) + 1) * TARGET)) - 1
    radius = float(np.sort(np.abs(y_calibration))[index])
    z_high = float(norm.ppf(1 - ALPHA / 2))

    def band(half_width: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        cover = (np.abs(y_test) <= half_width).astype(np.int8)
        p_true = 2.0 * norm.cdf(half_width / sigma_test) - 1.0
        return cover, p_true

    # The one-sided regimes scale the oracle half-width by a factor that varies
    # with x1, so their miscoverage is genuinely conditional rather than a
    # constant marginal shift a single number could summarise.
    ramp = (x_test[:, 0] + 1.0) / 2.0
    conservative_factor = 1.05 + (CONSERVATIVE_SCALE - 1.05) * ramp
    aggressive_factor = 0.95 - (0.95 - AGGRESSIVE_SCALE) * ramp

    out = {}
    for name, half_width in (
        ("standard_cp", np.full(len(y_test), radius)),
        ("oracle", z_high * sigma_test),
        ("conservative", conservative_factor * z_high * sigma_test),
        ("aggressive", aggressive_factor * z_high * sigma_test),
    ):
        cover, p_true = band(half_width)
        out[name] = {
            "features": x_test,
            "cover": cover,
            "p_true": p_true,
            "true_l1_ert": float(np.mean(np.abs(TARGET - p_true))),
            "true_l1_ert_over": float(np.mean(np.clip(p_true - TARGET, 0, None))),
            "true_l1_ert_under": float(np.mean(np.clip(TARGET - p_true, 0, None))),
            "marginal_coverage": float(cover.mean()),
        }
    out["standard_cp"]["conformal_radius"] = radius
    return out


def subsample(seed: int, tag: str, size_index: int, n: int, size: int) -> np.ndarray:
    """Deterministic uniform draw without replacement, stable across resumes."""
    generator = torch.Generator(device="cpu")
    offset = zlib.crc32(tag.encode()) % 997
    generator.manual_seed((seed * 1_000_003 + size_index * 1_009 + offset) % (2 ** 31))
    return torch.randperm(n, generator=generator)[:size].numpy()
