"""Round 0: reproduce the baseline (~83%) of the current bayes_net.py on the
standard validation split, using the shared cache infrastructure."""

from common import load_split, fit_eval
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
from bayes_net import BayesNetClassifier


def main():
    X_tr, y_tr, X_va, y_va = load_split()
    print(f"train={X_tr.shape[0]} val={X_va.shape[0]}")

    # exact protocol path
    for alpha in (0.1, 0.5, 1.0):
        model = BayesNetClassifier(alpha=alpha).fit(X_tr, y_tr)
        pred = model.predict(X_va)
        acc = float(np.mean(pred == y_va))
        print(f"[protocol] alpha={alpha}: val acc={acc:.4f}")

    import time
    t0 = time.perf_counter()
    model.predict(X_va)
    print(f"predict on {X_va.shape[0]} val images: {time.perf_counter()-t0:.2f}s")


if __name__ == "__main__":
    main()
