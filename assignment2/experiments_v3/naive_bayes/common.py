"""Shared utilities for naive bayes iteration experiments.

Only uses training data (60,000) with the fixed stratified split from
main.py (seed 1322).  The test set is never loaded here.
"""

from __future__ import annotations

import gzip
import struct
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "assignment2-data" / "assignment2-data" / "mnist"


def _read_idx(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as stream:
        magic = stream.read(4)
        dtype_code, ndim = magic[2], magic[3]
        dtype_map = {0x08: np.uint8, 0x09: np.int8, 0x0B: ">i2", 0x0C: ">i4"}
        dims = [struct.unpack(">I", stream.read(4))[0] for _ in range(ndim)]
        raw = stream.read()
    return np.frombuffer(raw, dtype=dtype_map[dtype_code]).reshape(dims)


def load_train():
    """Load only the 60,000 training images."""
    X = _read_idx(DATA_DIR / "train-images-idx3-ubyte.gz")
    y = _read_idx(DATA_DIR / "train-labels-idx1-ubyte.gz").astype(np.int64)
    return X.reshape(-1, 784), y


def stratified_split(y: np.ndarray, validation_fraction=1 / 6, seed: int = 1322):
    """Same deterministic split as main.py."""
    rng = np.random.default_rng(seed)
    train_parts, validation_parts = [], []
    for class_index in range(10):
        indices = rng.permutation(np.flatnonzero(y == class_index))
        validation_size = max(1, round(indices.size * validation_fraction))
        validation_parts.append(indices[:validation_size])
        train_parts.append(indices[validation_size:])
    return np.concatenate(train_parts), np.concatenate(validation_parts)


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true == y_pred))


class Timer:
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.seconds = time.perf_counter() - self.start


def bincount_2d(states: np.ndarray, n_states: int) -> np.ndarray:
    """Count feature/state occurrences per column.

    states: (n, d) int array.  Returns (d, n_states) float array.
    """
    n, d = states.shape
    offsets = states.astype(np.int64) + np.arange(d, dtype=np.int64) * n_states
    counts = np.bincount(offsets.ravel(), minlength=d * n_states)
    return counts.reshape(d, n_states).astype(np.float64)


def fit_categorical_nb(states: np.ndarray, y: np.ndarray, n_states: int,
                       alpha: float, n_classes: int = 10):
    """Fit P(y) and P(x_i=v|y) for an integer feature matrix.

    Returns (log_prior (C,), log_prob (C, d, S)).
    """
    d = states.shape[1]
    class_count = np.bincount(y, minlength=n_classes).astype(np.float64)
    log_prior = np.log((class_count + alpha) / (class_count.sum() + n_classes * alpha))
    log_prob = np.empty((n_classes, d, n_states), dtype=np.float64)
    for c in range(n_classes):
        counts = bincount_2d(states[y == c], n_states)
        log_prob[c] = np.log((counts + alpha) / (class_count[c] + n_states * alpha))
    return log_prior, log_prob


def predict_categorical_nb(states: np.ndarray, log_prior: np.ndarray,
                           log_prob: np.ndarray, batch_size: int = 2048) -> np.ndarray:
    """Vectorised MAP prediction for an integer feature matrix."""
    d, n_states = log_prob.shape[1], log_prob.shape[2]
    preds = np.empty(states.shape[0], dtype=np.int64)
    for start in range(0, states.shape[0], batch_size):
        stop = min(start + batch_size, states.shape[0])
        batch = states[start:stop].astype(np.int64)
        # (C, d, S) gather -> (batch, C, d)
        lp = log_prob[:, np.arange(d)[None, :], batch].transpose(1, 0, 2)
        scores = lp.sum(axis=2) + log_prior[None, :]
        preds[start:stop] = np.argmax(scores, axis=1)
    return preds


def evaluate_on_validation(model_fit_predict, X_tr, y_tr, X_val, y_val, label: str):
    """model_fit_predict(X, y) -> fitted predict function."""
    with Timer() as t_fit:
        predict_fn = model_fit_predict(X_tr, y_tr)
    with Timer() as t_pred:
        preds = predict_fn(X_val)
    acc = accuracy(y_val, preds)
    print(f"[{label}] val_acc={acc:.4f}  fit={t_fit.seconds:.1f}s  predict={t_pred.seconds:.1f}s")
    return acc


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    from naive_bayes import NaiveBayesClassifier

    X, y = load_train()
    print(f"train: X={X.shape}, y={y.shape}")
    tr, va = stratified_split(y)
    print(f"split: train={tr.size}, val={va.size}")

    # Baseline reproduction: 7 gray levels, alpha sweep on validation
    for alpha in (0.1, 0.5, 1.0):
        model = NaiveBayesClassifier(alpha=alpha).fit(X[tr], y[tr])
        acc = accuracy(y[va], model.predict(X[va]))
        print(f"[baseline 7-level alpha={alpha}] val_acc={acc:.4f}")
