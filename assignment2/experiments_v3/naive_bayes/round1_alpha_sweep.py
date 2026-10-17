"""Round 3-v3, experiment 1: fine alpha sweep for the patch-L5 model.

Motivation: metrics.json's validation_study already shows alpha=0.1 gives
92.73% validation accuracy for the current 2x2 patch model, far above the
finalized alpha=0.5 (91.81%).  Round 4 of experiments_v2 only swept
{0.5, 1.0, 2.0} and missed the small-alpha region.  Here we sweep
alpha in {0.05, 0.1, 0.15, 0.2, 0.3, 0.5} on the fixed stratified split.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import Timer, accuracy, load_train, stratified_split

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from naive_bayes import NaiveBayesClassifier


def main() -> None:
    X, y = load_train()
    tr, va = stratified_split(y)
    X_tr, X_va = X[tr], X[va]
    y_tr, y_va = y[tr], y[va]
    print(f"split: train={tr.size}, val={va.size}", flush=True)

    for alpha in (0.05, 0.1, 0.15, 0.2, 0.3, 0.5):
        model = NaiveBayesClassifier(alpha=alpha)
        with Timer() as t_fit:
            model.fit(X_tr, y_tr)
        with Timer() as t_pred:
            preds = model.predict(X_va)
        acc = accuracy(y_va, preds)
        print(
            f"[patch L5 alpha={alpha}] val_acc={acc:.4f}  "
            f"fit={t_fit.seconds:.1f}s  predict={t_pred.seconds:.1f}s",
            flush=True,
        )


if __name__ == "__main__":
    main()
