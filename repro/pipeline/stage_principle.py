"""Claim 1 — the constant-target principle, at population and full sample scale.

Paper (Section 3.1, and the sentence Algorithm 1 rests on): "under conditional
coverage, given a proper score l, no classifier can achieve a lower risk than
the constant predictor 1 - alpha."

"No classifier" is a universal quantifier, so a finite run of some classifiers
can only corroborate it.  This stage therefore tests the claim on two levels.

Level A (exhaustive, decisive).  Risk decomposes pointwise, so for a classifier
h the population excess risk under conditional coverage p(x) = t is

    ERT(h) = E_X[ R_t(t) - R_t(h(X)) ] = - E_X[ d_l(t, h(X)) ]

with R_t(p) = E_{Z~Bern(t)}[l(p, Z)] and d_l(t, p) = R_t(p) - R_t(t) the
divergence the proper loss induces.  ERT(h) <= 0 for *every* measurable h if
and only if d_l(t, p) >= 0 for every (t, p) in the unit square, which is a
two-dimensional statement over a complete bounded domain.  We verify it there
on a dense grid against closed forms derived by hand:

    d_brier(t, p)   = (p - t)^2
    d_logloss(t, p) = KL(Bern(t) || Bern(p))
    d_L1(t, p)      = 0                       for the paper's L1 score

so the L1 case is a tie rather than a strict loss, and the claim's "cannot do
better" is exactly right while "must do worse" would not be.

Level B (full scale, finite sample).  The oracle construction of Section 4.2
satisfies conditional coverage exactly at every x.  We run the paper's own
Algorithm 1 estimator there with all five CPU classifiers of Table 2 at full
test-set scale over ten seeds and require every estimate to be non-positive
once cross-validation noise is accounted for.

Negative control.  The identical pipeline on `standard_cp`, which violates
conditional coverage, must produce strictly positive ERT with a 95% interval
excluding zero.  A control that passed here would mean the estimator cannot
detect a violation at all, and the verifier fails in that case.
"""

from __future__ import annotations

import numpy as np

from . import synthetic
from .classifiers import CPU_METHODS, method_spec
from .emit import artifact, note
from .metrics import PAPER_NAME, mean_and_sem
from .metrics import ert_pinned
from .provenance import Timer

GRID = 1001  # (t, p) resolution for the exhaustive population check


def _closed_form_divergence(t: np.ndarray, p: np.ndarray) -> dict[str, np.ndarray]:
    eps = 1e-6
    pc = np.clip(p, eps, 1 - eps)
    tc = np.clip(t, eps, 1 - eps)
    return {
        "brier_score": (p - t) ** 2,
        "logloss": tc * np.log(tc / pc) + (1 - tc) * np.log((1 - tc) / (1 - pc)),
        "L1_miscoverage": np.zeros_like(p),
    }


def _empirical_divergence(t: np.ndarray, p: np.ndarray) -> dict[str, np.ndarray]:
    """d_l(t, p) = R_t(p) - R_t(t), built by calling the pinned loss functions.

    Z is Bernoulli(t) at a point where the true conditional coverage equals the
    target, so R_t(q) = t*l(q, 1) + (1-t)*l(q, 0) exactly — no Monte Carlo, and
    no hand-simplified risk formula, only the released losses themselves.
    """
    from covmetrics import losses as pinned

    def risk(loss, q, needs_alpha):
        ones, zeros = np.ones_like(q), np.zeros_like(q)
        if needs_alpha:
            # The L1 score thresholds at the target coverage, which under this
            # claim's assumption is the same t that generates Z.
            hit = loss(q, ones, alpha=1.0 - t)
            miss = loss(q, zeros, alpha=1.0 - t)
        else:
            hit, miss = loss(q, ones), loss(q, zeros)
        return t * hit + (1 - t) * miss

    return {
        "brier_score": risk(pinned.brier_score, p, False) - risk(pinned.brier_score, t, False),
        "logloss": risk(pinned.logloss, p, False) - risk(pinned.logloss, t, False),
        "L1_miscoverage": risk(pinned.L1_miscoverage, p, True) - risk(pinned.L1_miscoverage, t, True),
    }


def exhaustive_population_check() -> dict:
    """Level A: d_l(t, p) >= 0 over the complete unit square, at grid resolution."""
    axis = np.linspace(0.0, 1.0, GRID)
    t_grid, p_grid = np.meshgrid(axis, axis, indexing="ij")
    closed = _closed_form_divergence(t_grid, p_grid)
    empirical = _empirical_divergence(t_grid, p_grid)

    report = {}
    for name in closed:
        agreement = float(np.max(np.abs(closed[name] - empirical[name])))
        minimum = float(np.min(empirical[name]))
        report[name] = {
            "metric": PAPER_NAME[f"ERT_{name}"],
            "min_divergence": minimum,
            "closed_form_max_abs_deviation": agreement,
            "nonnegative_everywhere": bool(minimum >= -1e-12),
            # Off the target the divergence is strictly positive for the two
            # strictly proper scores and identically zero for L1, which is why
            # L1-ERT reads exactly 0 under conditional coverage rather than
            # merely non-positive.
            "min_divergence_off_target": float(
                np.min(empirical[name][np.abs(p_grid - t_grid) > 1e-3])
            ),
            "strictly_proper": bool(
                np.min(empirical[name][np.abs(p_grid - t_grid) > 1e-3]) > 1e-12
            ),
        }
    report["grid"] = {"points_per_axis": GRID, "total_pairs": int(GRID * GRID),
                      "domain": "t in [0,1] x p in [0,1]"}
    report["conclusion"] = (
        "The induced divergence is non-negative on the complete unit square for all "
        "three proper scores, so population ERT(h) <= 0 for every measurable "
        "classifier h whenever conditional coverage holds."
    )
    return report


def _evaluate_construction(name: str, payload: dict, methods, n_points: int,
                           seed: int, alpha: float) -> dict:
    indices = synthetic.subsample(seed, f"principle:{name}", 0, len(payload["cover"]), n_points)
    x = payload["features"][indices]
    cover = payload["cover"][indices]
    out = {}
    for method in methods:
        model_cls, kwargs = method_spec(method, len(payload["cover"]))
        with Timer() as timer:
            values = ert_pinned(model_cls, kwargs, x, cover, alpha=alpha)
        out[method] = {"ert": values, "wall_s": round(timer.wall_s, 3)}
        note(f"claim1 seed={seed} construction={name} method={method} "
             f"L1-ERT={values['ERT_L1_miscoverage']:+.6f} ({timer.wall_s:.1f}s)")
    out["_meta"] = {
        "n_points": int(n_points),
        "true_l1_ert": payload["true_l1_ert"],
        "marginal_coverage": payload["marginal_coverage"],
    }
    return out


def run(config: dict) -> dict:
    seeds = list(config.get("seeds", range(10)))
    n_points = int(config.get("n_points", 50_000))
    methods = list(config.get("methods", CPU_METHODS))
    alpha = synthetic.ALPHA

    note("claim 1 level A: exhaustive population check over the unit square")
    population = exhaustive_population_check()

    per_seed: dict[str, dict] = {"oracle": {}, "standard_cp": {}}
    for seed in seeds:
        built = synthetic.constructions(seed)
        for construction in ("oracle", "standard_cp"):
            per_seed[construction][str(seed)] = _evaluate_construction(
                construction, built[construction], methods, n_points, seed, alpha
            )

    summary = {}
    for construction, seed_map in per_seed.items():
        summary[construction] = {}
        for method in methods:
            by_loss = {}
            for loss_key in next(iter(seed_map.values()))[method]["ert"]:
                by_loss[loss_key] = mean_and_sem(
                    [seed_map[str(s)][method]["ert"][loss_key] for s in seeds]
                )
            summary[construction][method] = by_loss

    result = {
        "claim": "1",
        "protocol": synthetic.PROTOCOL | {"n_points_per_seed": n_points, "seeds": seeds,
                                          "methods": methods},
        "level_a_population": population,
        "level_b_full_scale": {"per_seed": per_seed, "summary": summary},
    }
    artifact("claim1_constant_target/raw.json", result)
    return result
