"""Discretisation + small-BN fitting over CACHED continuous columns.

This is the sweep engine.  It mirrors the incumbent's probabilistic model
exactly (same log-joint, same Dirichlet smoothing, same class-conditional
marginal back-off for nodes with a feature parent) but lets a design choose
(a) which cached columns to use, (b) how each is discretised, (c) the DAG.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
V4 = HERE.parents[1]
if str(V4) not in sys.path:
    sys.path.insert(0, str(V4))
from v4_common import mcnemar, paired_bootstrap, accuracy  # noqa: E402

N_CLASSES = 10


# ----------------------------------------------------------------------
# discretisers  (all return (train_states, val_states, thresholds))
# ----------------------------------------------------------------------
def disc_zero_quantile(col_tr, col_va, n_states):
    """State 0 = exactly zero; states 1..n-1 = train quantiles of positives."""
    positives = col_tr[col_tr > 0]
    if positives.size == 0:
        return (np.zeros(col_tr.shape, np.int64), np.zeros(col_va.shape, np.int64),
                np.array([0.0]))
    edges = np.quantile(positives, np.linspace(0, 1, n_states)[1:-1])
    return (_apply_zero(col_tr, edges), _apply_zero(col_va, edges), edges)


def _apply_zero(col, edges):
    return (np.searchsorted(edges, col, side="right") + (col > 0)).astype(np.int64)


def disc_quantile(col_tr, col_va, n_states):
    """Plain train quantiles (no zero separation)."""
    edges = np.quantile(col_tr, np.linspace(0, 1, n_states)[1:-1])
    return (np.searchsorted(edges, col_tr, side="right").astype(np.int64),
            np.searchsorted(edges, col_va, side="right").astype(np.int64), edges)


def disc_equal_width(col_tr, col_va, n_states):
    lo, hi = float(col_tr.min()), float(col_tr.max())
    edges = np.linspace(lo, hi, n_states + 1)[1:-1]
    return (np.searchsorted(edges, col_tr, side="right").astype(np.int64),
            np.searchsorted(edges, col_va, side="right").astype(np.int64), edges)


def disc_zero_plus_counts(col_tr, col_va, n_states):
    """For integer counts: state 0 = 0, then singletons, then a final 'many' bin.

    Uses only integer thresholds 1..n_states-2 (train-derived) so no count is
    split arbitrarily.  Falls back to quantiles when the count is non-integer.
    """
    if not np.allclose(col_tr, np.rint(col_tr)):
        return disc_zero_quantile(col_tr, col_va, n_states)
    edges = np.arange(1, n_states - 1, dtype=np.float64)
    return (_apply_zero(col_tr, edges), _apply_zero(col_va, edges), edges)


def disc_euler_incumbent(col_tr, col_va, n_states):
    """The incumbent's euler encoding: clip to [-1, 2] then +1 -> 4 states."""
    a = (np.clip(col_tr, -1, 2).astype(np.int64) + 1)
    b = (np.clip(col_va, -1, 2).astype(np.int64) + 1)
    return a, b, np.array([-1.0, 0.0, 2.0])


DISCRETISERS = {
    "zero_quantile": disc_zero_quantile,
    "quantile": disc_quantile,
    "equal_width": disc_equal_width,
    "zero_counts": disc_zero_plus_counts,
    "euler_incumbent": disc_euler_incumbent,
}


# ----------------------------------------------------------------------
# mutual information (train only) -- used to GUIDE, never to auto-select
# ----------------------------------------------------------------------
def mutual_information(states: np.ndarray, y: np.ndarray, n_states: int) -> float:
    joint = np.zeros((N_CLASSES, n_states), dtype=np.float64)
    np.add.at(joint, (y, states), 1)
    joint /= joint.sum()
    py = joint.sum(axis=1, keepdims=True)
    pf = joint.sum(axis=0, keepdims=True)
    mask = joint > 0
    return float(np.sum(joint[mask] * np.log(joint[mask] / (py * pf)[mask])))


def pairwise_mi(a: np.ndarray, b: np.ndarray, na: int, nb: int) -> float:
    joint = np.zeros((na, nb), dtype=np.float64)
    np.add.at(joint, (a, b), 1)
    joint /= joint.sum()
    pa = joint.sum(axis=1, keepdims=True)
    pb = joint.sum(axis=0, keepdims=True)
    mask = joint > 0
    return float(np.sum(joint[mask] * np.log(joint[mask] / (pa * pb)[mask])))


# ----------------------------------------------------------------------
# the BN itself
# ----------------------------------------------------------------------
class SweepBN:
    """Small Bayesian network over pre-discretised cached columns."""

    def __init__(self, names, cards, parents, alpha=1.0):
        self.names = tuple(names)
        self.cards = tuple(int(c) for c in cards)
        self.parents = tuple(int(p) for p in parents)
        self.alpha = float(alpha)

    def fit(self, F: np.ndarray, y: np.ndarray):
        alpha = self.alpha
        n = F.shape[0]
        class_count = np.bincount(y, minlength=N_CLASSES).astype(np.float64)
        self.class_log_prior_ = np.log(
            (class_count + alpha) / (n + N_CLASSES * alpha)
        )
        cpts = []
        for node, parent in enumerate(self.parents):
            card = self.cards[node]
            if parent < 0:
                counts = np.zeros((N_CLASSES, card), dtype=np.float64)
                np.add.at(counts, (y, F[:, node]), 1)
                cpts.append(np.log((counts + alpha) / (counts.sum(axis=-1, keepdims=True) + card * alpha)))
            else:
                pcard = self.cards[parent]
                counts = np.zeros((N_CLASSES, pcard, card), dtype=np.float64)
                np.add.at(counts, (y, F[:, parent], F[:, node]), 1)
                marginal = np.zeros((N_CLASSES, card), dtype=np.float64)
                np.add.at(marginal, (y, F[:, node]), 1)
                marginal = (marginal + alpha) / (class_count[:, None] + card * alpha)
                n_cp = counts.sum(axis=-1, keepdims=True)
                cpts.append(np.log((counts + alpha * marginal[:, None, :]) / (n_cp + alpha)))
        self.cpt_log_prob_ = tuple(cpts)
        return self

    def predict(self, F: np.ndarray) -> np.ndarray:
        scores = np.broadcast_to(self.class_log_prior_, (F.shape[0], N_CLASSES)).copy()
        for node, parent in enumerate(self.parents):
            log_cpt = self.cpt_log_prob_[node]
            if parent < 0:
                scores += log_cpt[:, F[:, node]].T
            else:
                scores += log_cpt[:, F[:, parent], F[:, node]].T
        return np.argmax(scores, axis=1).astype(np.int64)


# ----------------------------------------------------------------------
# a design = list of (cache_key, n_states, discretiser_name)
# ----------------------------------------------------------------------
def build_design(design, cols_tr, cols_va, y_tr):
    """Returns (F_tr, F_va, names, cards)."""
    names, cards = [], []
    tr_parts, va_parts = [], []
    for key, n_states, disc_name in design:
        disc = DISCRETISERS[disc_name]
        a, b, _ = disc(np.asarray(cols_tr[key], np.float64),
                       np.asarray(cols_va[key], np.float64), n_states)
        tr_parts.append(a)
        va_parts.append(b)
        names.append(key)
        cards.append(n_states)
    return (np.stack(tr_parts, axis=1), np.stack(va_parts, axis=1),
            tuple(names), tuple(cards))


def evaluate(design, parents, cols_tr, cols_va, y_tr, y_va, alpha=1.0):
    F_tr, F_va, names, cards = build_design(design, cols_tr, cols_va, y_tr)
    model = SweepBN(names, cards, parents, alpha=alpha).fit(F_tr, y_tr)
    pred = model.predict(F_va)
    return accuracy(y_va, pred), pred, model, F_tr, F_va


def validate_structure(names, cards, parents):
    n = len(names)
    assert 6 <= n <= 12, f"feature count {n} outside 6..12"
    for name, card in zip(names, cards):
        assert 2 <= card <= 5, f"{name}: cardinality {card} outside 2..5"
    edges = [(p, c) for c, p in enumerate(parents) if p >= 0]
    assert len(edges) >= 2, "need at least 2 feature-feature edges"
    for child, parent in enumerate(parents):
        assert parent < 0 or parent < n, f"feature {child}: bad parent {parent}"
        assert parent != child
        assert 1 + (1 if parent >= 0 else 0) <= 2
    indeg = np.zeros(n, int)
    kids = {i: [] for i in range(n)}
    for p, c in edges:
        kids[p].append(c)
        indeg[c] += 1
    ready = [i for i in range(n) if indeg[i] == 0]
    seen = 0
    while ready:
        u = ready.pop()
        seen += 1
        for v in kids[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                ready.append(v)
    assert seen == n, "cycle"
    return True
