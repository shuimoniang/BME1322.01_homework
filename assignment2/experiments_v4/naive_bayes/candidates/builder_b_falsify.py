"""Builder B: small-alpha asymptotics adjudication (incumbent vs tiny alpha).

Standalone (does not import the experiment module) so the numbers can be
re-derived quickly.  Uses only the 60,000 training images.
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
V4 = HERE.parent.parent
ROOT = V4.parent
for _p in (ROOT, V4):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from naive_bayes import NaiveBayesClassifier
from v4_common import (load_split, accuracy, compare_candidates, cross_seed,
                       top_confusions, mcnemar, macro_f1)

INCUMBENT_ALPHA = 0.05
CANDIDATE_ALPHA = 1e-4


def incumbent():
    return NaiveBayesClassifier(alpha=INCUMBENT_ALPHA)


def candidate():
    return NaiveBayesClassifier(alpha=CANDIDATE_ALPHA)


if __name__ == "__main__":
    Xtr, ytr, Xva, yva = load_split()
    print("== Main split (seed 1322) ==")
    res = compare_candidates(Xtr, ytr, Xva, yva, incumbent, candidate)
    print(f"incumbent  alpha={INCUMBENT_ALPHA}: {res['incumbent_accuracy']:.4f}")
    print(f"candidate  alpha={CANDIDATE_ALPHA}: {res['challenger_accuracy']:.4f}")
    print(f"delta={res['delta']:+.4f}  mcnemar b={res['mcnemar']['b']} c={res['mcnemar']['c']} "
          f"p={res['mcnemar']['p_value']:.3e}  bootstrap={res['bootstrap']}")
    print("real_gain:", res["real_gain"])

    pred_c = candidate().fit(Xtr, ytr).predict(Xva)
    _, cm = macro_f1(yva, pred_c)
    print("top confusions (candidate):", top_confusions(np.array(cm)))

    print("\n== Cross-seed falsification ==")
    cs_inc = cross_seed(incumbent)
    cs_cand = cross_seed(candidate)
    for seed in cs_inc:
        print(f"seed {seed}: incumbent={cs_inc[seed]:.4f} candidate={cs_cand[seed]:.4f} "
              f"delta={cs_cand[seed]-cs_inc[seed]:+.4f}")
