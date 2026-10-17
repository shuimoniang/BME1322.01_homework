"""Step 0: reproduce the incumbent 0.8614 at alpha=1.0 before anything else."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
V4 = HERE.parents[1]
ROOT = V4.parent
sys.path.insert(0, str(V4))
sys.path.insert(0, str(ROOT))

from v4_common import load_split, score_model, top_confusions, check_bn_bounds  # noqa: E402
from bayes_net import BayesNetClassifier, FEATURE_NAMES, FEATURE_CARDINALITIES, FEATURE_PARENTS  # noqa: E402


def main():
    Xtr, ytr, Xva, yva = load_split()
    print(f"protocol: train={Xtr.shape} val={Xva.shape}")
    for alpha in (0.5, 1.0, 2.0):
        t = time.perf_counter()
        model = BayesNetClassifier(alpha=alpha).fit(Xtr, ytr)
        res = score_model(model, Xva, yva)
        print(f"incumbent alpha={alpha:<5} val_acc={res['accuracy']:.4f} "
              f"macroF1={res['macro_f1']:.4f} ({time.perf_counter()-t:.1f}s)")
        if alpha == 1.0:
            print("top confusions (true, pred, n):", top_confusions(np.array(res["confusion_matrix"]), 8))
    check_bn_bounds(FEATURE_NAMES, FEATURE_CARDINALITIES, FEATURE_PARENTS)


if __name__ == "__main__":
    main()
