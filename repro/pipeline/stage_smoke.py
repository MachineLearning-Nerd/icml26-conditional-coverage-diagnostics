"""A cheap end-to-end check that the pinned environment can produce evidence.

It is not evidence for any claim.  It exists so a broken environment fails in
two minutes instead of six hours, and so the pinned covmetrics implementation
and the independent re-derivation in `metrics.py` are shown to agree on
identical folds before any long run trusts either of them.
"""

from __future__ import annotations

import numpy as np

from . import synthetic
from .classifiers import method_spec
from .emit import artifact, note
from .metrics import ert_independent, ert_pinned


def _cross_check(method: str, x, cover, alpha) -> dict:
    model_cls, kwargs = method_spec(method, len(cover))
    pinned = ert_pinned(model_cls, kwargs, x, cover, alpha=alpha)

    def fit_predict(x_train, cover_train, x_test):
        model = model_cls(**kwargs)
        model.fit(x_train, cover_train)
        return model.predict_proba(x_test)[:, 1]

    independent = ert_independent(fit_predict, x, cover, alpha=alpha)
    deltas = {key: abs(pinned[key] - independent[key]) for key in independent}
    return {"pinned": pinned, "independent": independent,
            "max_abs_delta": max(deltas.values()), "deltas": deltas}


def run(config: dict) -> dict:
    n = int(config.get("n_points", 4_000))
    methods = list(config.get("methods", ["PartitionWise", "XT"]))
    alpha = synthetic.ALPHA

    built = synthetic.constructions(0)
    out = {}
    for construction in ("standard_cp", "oracle"):
        payload = built[construction]
        indices = synthetic.subsample(0, f"smoke:{construction}", 0, len(payload["cover"]), n)
        x, cover = payload["features"][indices], payload["cover"][indices]
        out[construction] = {
            "true_l1_ert": payload["true_l1_ert"],
            "marginal_coverage": float(cover.mean()),
            "methods": {m: _cross_check(m, x, cover, alpha) for m in methods},
        }
        for m in methods:
            r = out[construction]["methods"][m]
            note(f"smoke {construction} {m}: pinned L1-ERT={r['pinned']['ERT_L1_miscoverage']:+.6f} "
                 f"independent={r['independent']['ERT_L1_miscoverage']:+.6f} "
                 f"max|delta|={r['max_abs_delta']:.2e}")

    worst = max(out[c]["methods"][m]["max_abs_delta"] for c in out for m in methods)
    result = {"claim": "none (environment check)", "n_points": n, "methods": methods,
              "constructions": out, "worst_pinned_vs_independent_delta": worst,
              "true_l1_ert_check": {c: out[c]["true_l1_ert"] for c in out}}
    artifact("run/smoke.json", result)
    if not np.isfinite(worst) or worst > 1e-9:
        raise AssertionError(
            f"pinned covmetrics and the independent re-derivation disagree by {worst:.3e}"
        )
    note("smoke: pinned and independent ERT agree to 1e-9 on identical folds")
    return result
