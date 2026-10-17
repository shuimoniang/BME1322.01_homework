"""Shared, leak-proof harness for the v4 adversarial iteration.

Protocol (identical to main.py and all earlier experiment dirs):

  * Only the 60,000 *training* images are ever loaded here.  The 10,000
    test images are NEVER read by anything in experiments_v4.
  * Fixed stratified split, seed 1322, validation_fraction = 1/6
    -> 49,999 train / 10,001 validation.
  * Every candidate is scored on that single validation split.  A claim
    that does not also hold on the extra seed splits below is treated as
    split luck, not a real gain.

Use it like:

    from v4_common import load_split, score_model, check_nb_bounds

    Xtr, ytr, Xva, yva = load_split()
    model = MyNb(alpha=0.05).fit(Xtr, ytr)
    print(score_model(model, Xva, yva))

`score_model` returns (accuracy, {metrics...}) so candidates can be ranked
by accuracy and inspected by macro F1 / confusion pairs.
"""

from __future__ import annotations

import gzip
import struct
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA_DIR = ROOT / "assignment2-data" / "assignment2-data" / "mnist"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Extra partitions used to test that a gain is not an artifact of the seed
# 1322 split.  Same code path as main.py's stratified_split.
EXTRA_SEEDS = (7, 2024)


def _read_idx(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as stream:
        magic = stream.read(4)
        dtype_code, ndim = magic[2], magic[3]
        dtype_map = {0x08: np.uint8, 0x09: np.int8, 0x0B: ">i2", 0x0C: ">i4"}
        dims = [struct.unpack(">I", stream.read(4))[0] for _ in range(ndim)]
        raw = stream.read()
    return np.frombuffer(raw, dtype=dtype_map[dtype_code]).reshape(dims)


def stratified_split(y: np.ndarray, validation_fraction: float = 1 / 6, seed: int = 1322):
    """Byte-for-byte the split used by main.py."""
    rng = np.random.default_rng(seed)
    train_parts, validation_parts = [], []
    for class_index in range(10):
        indices = rng.permutation(np.flatnonzero(y == class_index))
        validation_size = max(1, round(indices.size * validation_fraction))
        validation_parts.append(indices[:validation_size])
        train_parts.append(indices[validation_size:])
    return np.concatenate(train_parts), np.concatenate(validation_parts)


def load_train() -> tuple[np.ndarray, np.ndarray]:
    """The 60,000 training images only (never the test set)."""
    X = _read_idx(DATA_DIR / "train-images-idx3-ubyte.gz").reshape(-1, 784)
    y = _read_idx(DATA_DIR / "train-labels-idx1-ubyte.gz").astype(np.int64)
    return X, y


def load_split(seed: int = 1322, validation_fraction: float = 1 / 6):
    """(X_train, y_train, X_val, y_val) under the fixed protocol."""
    X, y = load_train()
    train_indices, validation_indices = stratified_split(y, validation_fraction, seed)
    return X[train_indices], y[train_indices], X[validation_indices], y[validation_indices]


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    matrix = np.zeros((10, 10), dtype=np.int64)
    np.add.at(matrix, (y_true, y_pred), 1)
    tp = np.diag(matrix).astype(np.float64)
    pred = matrix.sum(axis=0)
    actual = matrix.sum(axis=1)
    precision = np.divide(tp, pred, out=np.zeros(10), where=pred != 0)
    recall = np.divide(tp, actual, out=np.zeros(10), where=actual != 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(10),
                   where=(precision + recall) != 0)
    return float(f1.mean()), matrix


def score_model(model, X_val: np.ndarray, y_val: np.ndarray, *, with_f1: bool = True) -> dict:
    """Fit is assumed already done.  Returns accuracy (+ macro F1, matrix)."""
    start = time.perf_counter()
    pred = np.asarray(model.predict(X_val))
    seconds = time.perf_counter() - start
    result = {"accuracy": accuracy(y_val, pred), "predict_seconds": seconds}
    if with_f1:
        f1, matrix = macro_f1(y_val, pred)
        result["macro_f1"] = f1
        result["confusion_matrix"] = matrix.tolist()
    return result


def top_confusions(matrix: np.ndarray, limit: int = 5) -> list[tuple[int, int, int]]:
    matrix = np.asarray(matrix).copy()
    np.fill_diagonal(matrix, 0)
    flat = np.argsort(matrix.ravel())[::-1][:limit]
    out = []
    for flat_index in flat:
        true_label, predicted_label = np.unravel_index(flat_index, matrix.shape)
        out.append((int(true_label), int(predicted_label), int(matrix[true_label, predicted_label])))
    return out


def cross_seed(model_factory, seeds=EXTRA_SEEDS) -> dict:
    """Re-fit `model_factory()` on other splits; report each validation accuracy.

    `model_factory` must be a zero-argument callable returning an unfitted
    model.  This is the falsification test: a real gain survives new seeds.
    """
    scores = {}
    for seed in seeds:
        Xtr, ytr, Xva, yva = load_split(seed=seed)
        model = model_factory().fit(Xtr, ytr)
        scores[seed] = accuracy(yva, model.predict(Xva))
    return scores


# ----------------------------------------------------------------------
# Adversarial statistics: is a gain real, or split luck?
# ----------------------------------------------------------------------
def mcnemar(y_true, pred_a, pred_b):
    """McNemar test on two paired predictions over the same samples.

    Returns dict with b = #(A wrong, B right), c = #(A right, B wrong),
    the exact two-sided binomial p-value on min(b,c), and the normal
    approximation chi-square.  A candidate only counts as a real gain over
    the incumbent when b > c and p is small.
    """
    from math import comb

    y_true = np.asarray(y_true)
    correct_a = np.asarray(pred_a) == y_true
    correct_b = np.asarray(pred_b) == y_true
    b = int(np.sum(~correct_a & correct_b))   # A wrong, B right
    c = int(np.sum(correct_a & ~correct_b))   # A right, B wrong
    n = b + c
    if n == 0:
        return {"b": 0, "c": 0, "n": 0, "p_value": 1.0, "chi2": 0.0}
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(k + 1)) / (2 ** n)
    p_value = float(min(1.0, 2 * tail))
    chi2 = float((abs(b - c) - 1) ** 2 / n) if n else 0.0
    return {"b": b, "c": c, "n": n, "p_value": p_value, "chi2": chi2}


def paired_bootstrap(y_true, pred_a, pred_b, n_boot: int = 4000, seed: int = 1322):
    """95% CI for accuracy(B) - accuracy(A) via paired bootstrap resampling."""
    y_true = np.asarray(y_true)
    correct_a = (np.asarray(pred_a) == y_true).astype(np.float64)
    correct_b = (np.asarray(pred_b) == y_true).astype(np.float64)
    diff = correct_b - correct_a
    rng = np.random.default_rng(seed)
    n = diff.size
    draws = rng.integers(0, n, size=(n_boot, n))
    boot = diff[draws].mean(axis=1)
    return {
        "delta": float(diff.mean()),
        "ci_low": float(np.percentile(boot, 2.5)),
        "ci_high": float(np.percentile(boot, 97.5)),
    }


def compare_candidates(X_tr, y_tr, X_va, y_va, factory_incumbent, factory_challenger):
    """Fit both on the same split and adjudicate the challenger's gain.

    Returns a report a critic can act on: accuracies, McNemar, bootstrap CI
    and the claim 'real_gain' (challenger better AND McNemar p < 0.05).
    """
    incumbent = factory_incumbent().fit(X_tr, y_tr)
    challenger = factory_challenger().fit(X_tr, y_tr)
    pred_i = np.asarray(incumbent.predict(X_va))
    pred_c = np.asarray(challenger.predict(X_va))
    acc_i = accuracy(y_va, pred_i)
    acc_c = accuracy(y_va, pred_c)
    mc = mcnemar(y_va, pred_i, pred_c)
    bs = paired_bootstrap(y_va, pred_i, pred_c)
    return {
        "incumbent_accuracy": acc_i,
        "challenger_accuracy": acc_c,
        "delta": acc_c - acc_i,
        "mcnemar": mc,
        "bootstrap": bs,
        "real_gain": bool(acc_c > acc_i and mc["p_value"] < 0.05),
    }


# ----------------------------------------------------------------------
# Mechanical constraint audits (see CONSTRAINTS.md)
# ----------------------------------------------------------------------
def check_nb_bounds() -> None:
    """Naive Bayes stays inside the PDF: raise if the module broke a rule."""
    import inspect
    import naive_bayes

    source = inspect.getsource(naive_bayes)
    forbidden = ("sklearn", "pgmpy", "bnlearn", "pomegranate", "scipy.stats")
    for token in forbidden:
        if token in source:
            raise AssertionError(f"forbidden dependency referenced in naive_bayes: {token}")
    if hasattr(naive_bayes, "NaiveBayesClassifier"):
        classifier = naive_bayes.NaiveBayesClassifier()
    else:
        classifier = None
    print("naive_bayes bounds: OK, alpha default =",
          getattr(classifier, "alpha", "n/a"))


def check_bn_bounds(names, cardinalities, parents) -> None:
    """Bayesian network structure rules from PDF section 4.2."""
    n = len(names)
    assert 6 <= n <= 12, f"feature count {n} outside 6..12"
    for name, card in zip(names, cardinalities):
        assert 2 <= card <= 5, f"{name}: cardinality {card} outside 2..5"
    feature_edges = [(p, c) for c, p in enumerate(parents) if p >= 0]
    assert len(feature_edges) >= 2, "need at least 2 feature-feature edges"
    for child, parent in enumerate(parents):
        assert parent < 0 or parent < n, f"feature {child}: bad parent index {parent}"
        assert parent != child, f"feature {child} is its own parent"
        parents_count = 1 + (1 if parent >= 0 else 0)  # Y is a parent of every feature
        assert parents_count <= 2, f"feature {child}: {parents_count} parents > 2"
    # acyclic
    indegree = np.zeros(n, dtype=int)
    children = {i: [] for i in range(n)}
    for parent, child in feature_edges:
        children[parent].append(child)
        indegree[child] += 1
    ready = [i for i in range(n) if indegree[i] == 0]
    visited = 0
    while ready:
        node = ready.pop()
        visited += 1
        for child in children[node]:
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(child)
    assert visited == n, "feature graph contains a cycle"
    print(f"bayes_net bounds: OK ({n} features, {len(feature_edges)} feature edges)")


if __name__ == "__main__":
    Xtr, ytr, Xva, yva = load_split()
    print(f"protocol ok: train={Xtr.shape} val={Xva.shape}")
