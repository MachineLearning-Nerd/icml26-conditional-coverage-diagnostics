"""Claim 4 — the asymmetric over/under-coverage decomposition, at full scale.

Paper (Section 3.3): the metrics decompose conditional coverage error into
l+-ERT and l--ERT, separating unnecessary conservatism (over-coverage) from
excessive aggressiveness (under-coverage).

Three things have to hold for that sentence to be true, and this stage tests
all three at full sample scale rather than on a constructed pair of vectors.

1. Exact additivity.  l-ERT = l+-ERT + l--ERT for every proper score, not
   approximately.  Writing A = {h > 1-alpha} and B = {h < 1-alpha},
   R(clip_over h) = R_A(h) + R_B(1-alpha) and R(clip_under h) = R_A(1-alpha) +
   R_B(h), so the two excess risks sum to R(1-alpha) - R(h) identically.  We
   check this to machine precision on every value the run produces.

2. Correct one-sided response.  On a construction that over-covers everywhere
   the positive part must carry the signal and the negative part must vanish,
   and vice versa.  `conservative` (oracle interval scaled by 1.35) and
   `aggressive` (scaled by 0.80) are exactly those regimes, at full scale with
   an analytically known true decomposition to compare against.

3. Genuine localisation, not just a sign.  On `standard_cp` both parts are
   non-zero, and the claim is that they separate *regions*.  Because the true
   conditional coverage p(x) is known in closed form here, we can score how
   often sign(h(x) - (1-alpha)) agrees with sign(p(x) - (1-alpha)), which is
   the property "separating conservatism from aggressiveness" actually asserts.

Negative control: on `oracle`, which is conditionally exact, both parts must be
statistically indistinguishable from zero.  A run where the control also shows
a large one-sided component would mean the decomposition is reading noise.
"""

from __future__ import annotations

import numpy as np

from . import synthetic
from .classifiers import CPU_METHODS, method_spec
from .emit import artifact, note
from .metrics import ert_pinned, mean_and_sem
from .provenance import Timer

CONSTRUCTIONS = ("standard_cp", "conservative", "aggressive", "oracle")

ADDITIVITY_PAIRS = (
    ("ERT_L1_miscoverage", "ERT_L1_miscoverage_over", "ERT_L1_miscoverage_under"),
    ("ERT_brier_score", "ERT_brier_score_over", "ERT_brier_score_under"),
    ("ERT_logloss", "ERT_logloss_over", "ERT_logloss_under"),
)


def _localisation(model_cls, kwargs, x, cover, p_true, alpha) -> dict:
    """How well the sign of h - (1-alpha) recovers the true over/under regions."""
    from sklearn.model_selection import KFold

    target = 1.0 - alpha
    predictions = np.empty(len(cover), dtype=float)
    for train_index, test_index in KFold(n_splits=5, shuffle=True, random_state=42).split(x):
        model = model_cls(**kwargs)
        model.fit(x[train_index], cover[train_index])
        predictions[test_index] = model.predict_proba(x[test_index])[:, 1]

    truth_over = p_true > target
    truth_under = p_true < target
    called_over = predictions > target
    decided = truth_over | truth_under
    agree = (called_over == truth_over)[decided]
    return {
        "n_points": int(len(cover)),
        "fraction_truly_over_covered": float(truth_over.mean()),
        "fraction_called_over_covered": float(called_over.mean()),
        "sign_agreement": float(agree.mean()) if decided.any() else float("nan"),
        "sign_agreement_on_over": float(called_over[truth_over].mean()) if truth_over.any() else float("nan"),
        "sign_agreement_on_under": float((~called_over)[truth_under].mean()) if truth_under.any() else float("nan"),
        "mean_h": float(predictions.mean()),
    }


def run(config: dict) -> dict:
    seeds = list(config.get("seeds", range(10)))
    n_points = int(config.get("n_points", 50_000))
    methods = list(config.get("methods", CPU_METHODS))
    localisation_method = config.get("localisation_method", "CheapBetterLGBMClassifier")
    alpha = synthetic.ALPHA

    per_seed: dict[str, dict] = {name: {} for name in CONSTRUCTIONS}
    additivity_residuals: list[float] = []

    for seed in seeds:
        built = synthetic.constructions(seed)
        for name in CONSTRUCTIONS:
            payload = built[name]
            indices = synthetic.subsample(seed, f"decomposition:{name}", 0,
                                          len(payload["cover"]), n_points)
            x = payload["features"][indices]
            cover = payload["cover"][indices]
            p_true = payload["p_true"][indices]

            record = {
                "_meta": {
                    "n_points": int(n_points),
                    "true_l1_ert": float(np.mean(np.abs(1 - alpha - p_true))),
                    "true_l1_ert_over": float(np.mean(np.clip(p_true - (1 - alpha), 0, None))),
                    "true_l1_ert_under": float(np.mean(np.clip((1 - alpha) - p_true, 0, None))),
                    "marginal_coverage": float(cover.mean()),
                }
            }
            for method in methods:
                model_cls, kwargs = method_spec(method, len(payload["cover"]))
                with Timer() as timer:
                    values = ert_pinned(model_cls, kwargs, x, cover, alpha=alpha)
                for total, over, under in ADDITIVITY_PAIRS:
                    additivity_residuals.append(abs(values[total] - values[over] - values[under]))
                record[method] = {"ert": values, "wall_s": round(timer.wall_s, 3)}
                note(f"claim4 seed={seed} construction={name} method={method} "
                     f"L1+={values['ERT_L1_miscoverage_over']:+.6f} "
                     f"L1-={values['ERT_L1_miscoverage_under']:+.6f} ({timer.wall_s:.1f}s)")

            model_cls, kwargs = method_spec(localisation_method, len(payload["cover"]))
            record["localisation"] = _localisation(model_cls, kwargs, x, cover, p_true, alpha)
            per_seed[name][str(seed)] = record

    summary = {}
    for name, seed_map in per_seed.items():
        summary[name] = {"truth": {
            key: mean_and_sem([seed_map[str(s)]["_meta"][key] for s in seeds])
            for key in ("true_l1_ert", "true_l1_ert_over", "true_l1_ert_under", "marginal_coverage")
        }, "localisation": {
            key: mean_and_sem([seed_map[str(s)]["localisation"][key] for s in seeds])
            for key in ("sign_agreement", "sign_agreement_on_over", "sign_agreement_on_under",
                        "fraction_truly_over_covered", "fraction_called_over_covered")
        }}
        for method in methods:
            summary[name][method] = {
                loss_key: mean_and_sem([seed_map[str(s)][method]["ert"][loss_key] for s in seeds])
                for loss_key in seed_map[str(seeds[0])][method]["ert"]
            }

    result = {
        "claim": "4",
        "protocol": synthetic.PROTOCOL | {"n_points_per_seed": n_points, "seeds": seeds,
                                          "methods": methods,
                                          "localisation_method": localisation_method},
        "additivity": {
            "checked_values": len(additivity_residuals),
            "max_abs_residual": float(max(additivity_residuals)) if additivity_residuals else 0.0,
            "identity": "l-ERT == l+-ERT + l--ERT for L1, Brier and logloss",
        },
        "per_seed": per_seed,
        "summary": summary,
    }
    artifact("claim4_decomposition/raw.json", result)
    return result
