"""Claim 3 — CovGap stays unreliable at 5,000 test points where L1-ERT converges.

Paper (Section 4.2, Figure 4): "group-based metrics are extremely unaligned
with their theoretical values and require large sample sizes to converge.  Even
with 5,000 points, they provide nearly identical diagnostics across these two
very different scenarios ... By contrast, L1-ERT stabilizes very quickly."

The quantified content is therefore a comparison, at matched sample size,
between (a) each metric's distance from its own theoretical value and (b) each
metric's ability to tell the two scenarios apart.  Both are measurable here
because the true conditional coverage is known in closed form: the theoretical
L1-ERT is E_X|1-alpha-p(X)|, and the theoretical CovGap is 0 for the oracle
construction.  Fifteen log-spaced test sizes from 500 to 100,000 bracket the
5,000 the paper singles out.

This claim already held in the previous judged revision; it is re-run here so
the cumulative regression suite covers it on the current code.
"""

from __future__ import annotations

import numpy as np

from . import synthetic
from .classifiers import method_spec
from .emit import artifact, note
from .metrics import ert_pinned, mean_and_sem

SIZES = tuple(np.round(np.logspace(np.log10(500), np.log10(100_000), 15)).astype(int).tolist())
CONSTRUCTIONS = ("standard_cp", "oracle")


def covgap(features: np.ndarray, cover: np.ndarray, alpha: float) -> tuple[float, int]:
    """CovGap over KMeans feature-space groups, with the release's group count."""
    from sklearn.cluster import KMeans

    n_groups = max(1, int(len(cover) ** 0.25))
    if n_groups < 2:
        return float(abs(cover.mean() - (1 - alpha))), 1
    labels = KMeans(n_clusters=n_groups, random_state=42, n_init="auto").fit_predict(features)
    gaps = [abs(cover[labels == g].mean() - (1 - alpha)) for g in range(n_groups) if (labels == g).any()]
    return float(np.mean(gaps)), int(n_groups)


def run(config: dict) -> dict:
    seeds = list(config.get("seeds", range(10)))
    sizes = list(config.get("sizes", SIZES))
    method = config.get("method", "CheapBetterLGBMClassifier")
    alpha = synthetic.ALPHA

    per_seed: dict[str, dict] = {name: {} for name in CONSTRUCTIONS}
    for seed in seeds:
        built = synthetic.constructions(seed)
        for name in CONSTRUCTIONS:
            payload = built[name]
            model_cls, kwargs = method_spec(method, len(payload["cover"]))
            record = {"true_l1_ert": payload["true_l1_ert"], "true_covgap": 0.0
                      if name == "oracle" else None, "sizes": {}}
            for size_index, size in enumerate(sizes):
                indices = synthetic.subsample(seed, f"convergence:{name}", size_index,
                                              len(payload["cover"]), size)
                x, cover = payload["features"][indices], payload["cover"][indices]
                values = ert_pinned(model_cls, kwargs, x, cover, alpha=alpha)
                gap, n_groups = covgap(x, cover, alpha)
                record["sizes"][str(size)] = {
                    "l1_ert": values["ERT_L1_miscoverage"],
                    "l2_ert": values["ERT_brier_score"],
                    "kl_ert": values["ERT_logloss"],
                    "covgap": gap,
                    "covgap_groups": n_groups,
                }
                note(f"claim3 seed={seed} {name} n={size} L1-ERT={values['ERT_L1_miscoverage']:+.6f} "
                     f"CovGap={gap:.6f}")
            per_seed[name][str(seed)] = record

    summary = {}
    for name in CONSTRUCTIONS:
        truth = mean_and_sem([per_seed[name][str(s)]["true_l1_ert"] for s in seeds])
        by_size = {}
        for size in sizes:
            l1 = [per_seed[name][str(s)]["sizes"][str(size)]["l1_ert"] for s in seeds]
            gaps = [per_seed[name][str(s)]["sizes"][str(size)]["covgap"] for s in seeds]
            truths = [per_seed[name][str(s)]["true_l1_ert"] for s in seeds]
            by_size[str(size)] = {
                "l1_ert": mean_and_sem(l1),
                "covgap": mean_and_sem(gaps),
                "l1_ert_abs_error": mean_and_sem(np.abs(np.array(l1) - np.array(truths))),
                # CovGap's theoretical value is 0 on the oracle construction; on
                # standard_cp the partition estimator has no closed form, so only
                # the oracle arm has an absolute-error column.
                "covgap_abs_error": mean_and_sem(np.abs(gaps)) if name == "oracle" else None,
            }
        summary[name] = {"true_l1_ert": truth, "by_size": by_size}

    # The separation test: at each size, how far apart do the two scenarios read?
    separation = {}
    for size in sizes:
        l1_gap = [per_seed["standard_cp"][str(s)]["sizes"][str(size)]["l1_ert"]
                  - per_seed["oracle"][str(s)]["sizes"][str(size)]["l1_ert"] for s in seeds]
        cg_gap = [per_seed["standard_cp"][str(s)]["sizes"][str(size)]["covgap"]
                  - per_seed["oracle"][str(s)]["sizes"][str(size)]["covgap"] for s in seeds]
        separation[str(size)] = {"l1_ert_separation": mean_and_sem(l1_gap),
                                 "covgap_separation": mean_and_sem(cg_gap)}

    result = {
        "claim": "3",
        "protocol": synthetic.PROTOCOL | {"seeds": seeds, "sizes": sizes, "method": method},
        "per_seed": per_seed,
        "summary": summary,
        "separation": separation,
    }
    artifact("claim3_convergence/raw.json", result)
    return result
