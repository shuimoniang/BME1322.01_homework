"""End-to-end falsification of the Builder D candidate against the incumbent.

Uses the REAL modules (``assignment2/bayes_net.py`` and
``candidates/builder_d_serpentine.py``), i.e. the full fit -> predict path with
no cached features, so the reported numbers are the numbers a critic can
reproduce by importing the two modules.

Outputs JSON to ``../results/builder_d_final.json`` and prints a table.

Run:  python builder_d_final_validation.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
V4 = HERE.parent.parent
ROOT = V4.parent
for path in (str(HERE), str(V4), str(ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from v4_common import (  # noqa: E402
    EXTRA_SEEDS,
    accuracy,
    check_bn_bounds,
    compare_candidates,
    load_split,
    macro_f1,
    score_model,
    top_confusions,
)
from bayes_net import BayesNetClassifier as Incumbent  # noqa: E402
from builder_d_serpentine import (  # noqa: E402
    BayesNetClassifier as Candidate,
    FEATURE_CARDINALITIES,
    FEATURE_NAMES,
    FEATURE_PARENTS,
)

RESULTS = HERE.parent / "results"
ALPHA = 1.0


def mcnemar_pooled(b: int, c: int) -> dict:
    """Exact two-sided binomial McNemar on pooled discordant pairs (3 seeds).

    Pooling the three independent discordant counts is valid because each
    seed's split is an independent sample of the 60,000 training images.
    """
    from math import comb

    n = b + c
    if n == 0:
        return {"b": 0, "c": 0, "n": 0, "p_value": 1.0, "chi2": 0.0}
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(k + 1)) / (2 ** n)
    p = float(min(1.0, 2 * tail))
    chi2 = float((abs(b - c) - 1) ** 2 / n)
    return {"b": b, "c": c, "n": n, "p_value": p, "chi2": chi2}


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = {}

    print("== bounds audit ==")
    check_bn_bounds(FEATURE_NAMES, FEATURE_CARDINALITIES, FEATURE_PARENTS)
    out["structure"] = {
        "feature_parents": list(FEATURE_PARENTS),
        "edges": [
            [FEATURE_NAMES[p], FEATURE_NAMES[c]]
            for c, p in enumerate(FEATURE_PARENTS) if p >= 0
        ],
    }

    # ---- seed 1322 head-to-head -------------------------------------
    Xtr, ytr, Xva, yva = load_split(seed=1322)
    report = compare_candidates(
        Xtr, ytr, Xva, yva,
        lambda: Incumbent(alpha=ALPHA),
        lambda: Candidate(alpha=ALPHA),
    )
    inc = Incumbent(alpha=ALPHA).fit(Xtr, ytr)
    cand = Candidate(alpha=ALPHA).fit(Xtr, ytr)
    inc_metrics = score_model(inc, Xva, yva)
    cand_metrics = score_model(cand, Xva, yva)
    report["incumbent_macro_f1"] = inc_metrics["macro_f1"]
    report["candidate_macro_f1"] = cand_metrics["macro_f1"]
    report["candidate_top_confusions"] = top_confusions(
        np.array(cand_metrics["confusion_matrix"]), limit=6)
    report["incumbent_top_confusions"] = top_confusions(
        np.array(inc_metrics["confusion_matrix"]), limit=6)
    print("\n== seed 1322 head-to-head (alpha=1.0) ==")
    print(json.dumps(report, indent=2, default=str))
    out["seed1322"] = report

    # ---- cross-seed -------------------------------------------------
    print("\n== cross-seed (independent re-splits) ==")
    cross = {}
    for seed in (1322,) + tuple(EXTRA_SEEDS):
        Xtr, ytr, Xva, yva = load_split(seed=seed)
        pred_i = Incumbent(alpha=ALPHA).fit(Xtr, ytr).predict(Xva)
        pred_c = Candidate(alpha=ALPHA).fit(Xtr, ytr).predict(Xva)
        from v4_common import mcnemar, paired_bootstrap
        a_i = accuracy(yva, pred_i)
        a_c = accuracy(yva, pred_c)
        cross[seed] = {
            "incumbent": a_i, "candidate": a_c, "delta": a_c - a_i,
            "mcnemar": mcnemar(yva, pred_i, pred_c),
            "bootstrap": paired_bootstrap(yva, pred_i, pred_c),
        }
        mc = cross[seed]["mcnemar"]
        print(f"  seed {seed:>5}: incumbent={a_i:.4f}  candidate={a_c:.4f}  "
              f"delta={a_c - a_i:+.4f}  mcnemar p={mc['p_value']:.4f} "
              f"(b={mc['b']} c={mc['c']})")
    out["cross_seed"] = cross
    deltas = [v["delta"] for v in cross.values()]
    pooled_b = sum(v["mcnemar"]["b"] for v in cross.values())
    pooled_c = sum(v["mcnemar"]["c"] for v in cross.values())
    pooled = mcnemar_pooled(pooled_b, pooled_c)
    out["cross_seed_summary"] = {
        "mean_delta": float(np.mean(deltas)),
        "min_delta": float(np.min(deltas)),
        "all_positive": bool(all(d > 0 for d in deltas)),
        "pooled_mcnemar": pooled,
    }
    print(f"  mean delta={np.mean(deltas):+.4f}  min delta={np.min(deltas):+.4f}  "
          f"all positive={all(d > 0 for d in deltas)}")
    print(f"  pooled McNemar over 3 seeds: b={pooled_b} c={pooled_c} "
          f"p={pooled['p_value']:.2e} chi2={pooled['chi2']:.2f}")

    path = RESULTS / "builder_d_final.json"
    path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
