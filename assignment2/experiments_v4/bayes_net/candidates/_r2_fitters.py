"""Round-2 CPT/prior estimator variants (all NumPy, all smoothed).

Two independent knobs are separated out so the mandate's "does M6 interact with
the class prior / does the class-conditional backoff still beat plain uniform
Lidstone" question can be answered cleanly:

  prior  : "empirical" (Dirichlet-smoothed class counts, the incumbent scheme)
           or "uniform"   (1/10 for every class)
  cpt    : "backoff"  (Dirichlet prior mean = class-conditional marginal P(F|y),
                       the incumbent scheme; only meaningful for nodes with a
                       feature parent)
           "uniform"  (plain Lidstone: alpha added flat to every cell)

For a node with no feature parent the two CPT schemes coincide, which is why
the backoff scheme is applied per-node only where a feature parent exists.
"""
from __future__ import annotations

import numpy as np

N_CLASSES = 10


class SweepBN2:
    """SweepBN with selectable class prior and CPT smoothing scheme."""

    def __init__(self, names, cards, parents, alpha=1.0,
                 prior="empirical", cpt="backoff"):
        self.names = tuple(names)
        self.cards = tuple(int(c) for c in cards)
        self.parents = tuple(int(p) for p in parents)
        self.alpha = float(alpha)
        assert prior in ("empirical", "uniform")
        assert cpt in ("backoff", "uniform")
        self.prior = prior
        self.cpt = cpt

    def fit(self, F: np.ndarray, y: np.ndarray):
        alpha = self.alpha
        n = F.shape[0]
        if self.prior == "empirical":
            class_count = np.bincount(y, minlength=N_CLASSES).astype(np.float64)
            self.class_log_prior_ = np.log(
                (class_count + alpha) / (n + N_CLASSES * alpha)
            )
        else:
            self.class_log_prior_ = np.full(N_CLASSES, -np.log(N_CLASSES))
            class_count = np.bincount(y, minlength=N_CLASSES).astype(np.float64)
        cpts = []
        for node, parent in enumerate(self.parents):
            card = self.cards[node]
            if parent < 0:
                counts = np.zeros((N_CLASSES, card), dtype=np.float64)
                np.add.at(counts, (y, F[:, node]), 1)
                cpts.append(np.log((counts + alpha)
                                   / (counts.sum(axis=-1, keepdims=True) + card * alpha)))
            else:
                pcard = self.cards[parent]
                counts = np.zeros((N_CLASSES, pcard, card), dtype=np.float64)
                np.add.at(counts, (y, F[:, parent], F[:, node]), 1)
                n_cp = counts.sum(axis=-1, keepdims=True)
                if self.cpt == "backoff":
                    marginal = np.zeros((N_CLASSES, card), dtype=np.float64)
                    np.add.at(marginal, (y, F[:, node]), 1)
                    marginal = (marginal + alpha) / (class_count[:, None] + card * alpha)
                    prior_mean = marginal[:, None, :]
                else:
                    prior_mean = 1.0 / card
                cpts.append(np.log((counts + alpha * prior_mean) / (n_cp + alpha)))
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
