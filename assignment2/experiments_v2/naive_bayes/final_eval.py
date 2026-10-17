"""Final evaluation: retrain the chosen model on all 60,000 training
images and evaluate ONCE on the full 10,000-image test set.

This is the single permitted use of the test set -- report only, no
further model changes afterwards.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import gzip
import struct

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

from naive_bayes import NaiveBayesClassifier  # final written-back module

DATA_DIR = ROOT / "assignment2-data" / "assignment2-data" / "mnist"


def _read_idx(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as stream:
        magic = stream.read(4)
        dtype_code, ndim = magic[2], magic[3]
        dtype_map = {0x08: np.uint8, 0x09: np.int8, 0x0B: ">i2", 0x0C: ">i4"}
        dims = [struct.unpack(">I", stream.read(4))[0] for _ in range(ndim)]
        raw = stream.read()
    return np.frombuffer(raw, dtype=dtype_map[dtype_code]).reshape(dims)


def confusion(y_true, y_pred):
    m = np.zeros((10, 10), dtype=np.int64)
    np.add.at(m, (y_true, y_pred), 1)
    return m


def main() -> None:
    X_train = _read_idx(DATA_DIR / "train-images-idx3-ubyte.gz").reshape(-1, 784)
    y_train = _read_idx(DATA_DIR / "train-labels-idx1-ubyte.gz").astype(np.int64)
    X_test = _read_idx(DATA_DIR / "t10k-images-idx3-ubyte.gz").reshape(-1, 784)
    y_test = _read_idx(DATA_DIR / "t10k-labels-idx1-ubyte.gz").astype(np.int64)
    print(f"train {X_train.shape}, test {X_test.shape}")

    model = NaiveBayesClassifier(alpha=0.5)
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    fit_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    preds = model.predict(X_test)
    pred_s = time.perf_counter() - t0
    acc = float(np.mean(preds == y_test))

    m = confusion(y_test, preds)
    tp = np.diag(m).astype(np.float64)
    prec = tp / m.sum(axis=0)
    rec = tp / m.sum(axis=1)
    f1 = 2 * prec * rec / (prec + rec)
    per_class_acc = tp / m.sum(axis=1)

    print(f"\nAccuracy: {acc:.4f}")
    print(f"Macro P: {prec.mean():.4f}  Macro R: {rec.mean():.4f}  Macro F1: {f1.mean():.4f}")
    print(f"fit={fit_s:.1f}s  predict={pred_s:.1f}s")
    print("per-class accuracy:")
    for c in range(10):
        print(f"  {c}: {per_class_acc[c]:.4f}")
    print("confusion matrix (rows=true, cols=pred):")
    print(m)

    mistakes = m.copy()
    np.fill_diagonal(mistakes, 0)
    flat = np.argsort(mistakes.ravel())[::-1][:5]
    top_pairs = [(int(i // 10), int(i % 10), int(mistakes.ravel()[i])) for i in flat]
    print("top confused pairs (true->pred, count):")
    for t, p, c in top_pairs:
        print(f"  {t}->{p}: {c}")

    result = {
        "accuracy": acc,
        "macro_precision": float(prec.mean()),
        "macro_recall": float(rec.mean()),
        "macro_f1": float(f1.mean()),
        "per_class_accuracy": per_class_acc.tolist(),
        "confusion_matrix": m.tolist(),
        "top_confused_pairs": top_pairs,
        "fit_seconds": fit_s,
        "predict_seconds": pred_s,
    }
    out = HERE / "final_test_result.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
