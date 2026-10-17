"""CRITIC alpha -- recommended production Naive Bayes for Assignment 2.

This is the survivor of the adversarial falsification of Builder A's candidate
(see ``results/critic_alpha.md``).  It is Builder A's geometry -- 484 overlapping
*dilated 4x4 binary* patches (16 pixels on a spacing-2 lattice, 7x7 receptive
field, pixel>=72 = ink, 2^16 joint states) -- with one correction the critic's
alpha study found:

    Builder A recommended alpha = 1e-3 "to stay conservative".
    But on A's own geometry the validation curve does NOT turn over at tiny
    alpha: it rises monotonically from 1e-3 to about 1e-9 and then saturates.
    alpha = 1e-3 is therefore *under-tuned*, not a safe plateau.  The 8-seed
    means are 0.9737 (1e-3) -> 0.9766 (1e-6) -> 0.9773 (1e-9 sat.), and the
    improvement is statistically overwhelming (pooled McNemar p = 2.5e-36 for
    1e-3 -> 1e-6, p = 4.6e-44 for 1e-3 -> 1e-8).  Training-internal nested
    selection, which never looks at the outer validation split, independently
    picks the smallest grid point, confirming this is not val overfitting.

We therefore ship alpha = 1e-6 as the production default: it is deep in the
saturated, seed-stable part of the curve, comfortably above the (unobserved)
collapse boundary, and its exact value is irrelevant anywhere in 1e-6..1e-12
(all within 0.1pt).  alpha = 1e-9 gives the same result and is also fine.

The model is plain naive Bayes

    P(y | X)  proportional to  P(y) * prod_j P(z_j | y)

over the 484 patch variables z_j, produced with hand-written NumPy CPT
estimation and exact sparse storage of the observed (feature, state) counts
(unobserved pairs get the same smoothed value for every class, so they cancel
from the argmax -- this is exact, not an approximation).  No ready-made
classifier is used.

Protocol / drop-in compatibility: ``__init__(alpha=1e-6)``, ``fit(X, y) -> self``,
``predict(X)``, ``predict_proba(X)``; X is the raw (n_samples, 784) uint8/float
matrix with values in [0, 255], identical input validation to the incumbent.

Unlike Builder A's module this version materialises the exact counts and derives
log-probabilities per alpha at *predict* time, which is what made the alpha sweep
cheap; the fitted attributes mirror the incumbent's names where memory allows
(``class_log_prior_``) and expose ``observed_keys_`` / ``observed_counts_`` /
``default_log_prob_`` for the compressed CPT.

Reference (v4 protocol, training data only, 8 seeds):
    mean validation accuracy 0.9773, min 0.9756, vs incumbent-best 0.9379.
"""

from __future__ import annotations

import numpy as np

N_CLASSES = 10
N_FEATURES = 28 * 28
IMG_SIZE = 28

# ---- frozen feature geometry (Builder A's contribution, verified by critic) ----
PATCH_H = 4
PATCH_W = 4
PATCH_INNER = 2          # spacing-2 lattice -> 7x7 receptive field
PATCH_STRIDE = 1         # maximally overlapping -> 22x22 = 484 patches
BINARY_THRESHOLD = 72.0
N_STATES = 1 << (PATCH_H * PATCH_W)                      # 65,536
N_PATCH_FEATURES = (IMG_SIZE - ((PATCH_H - 1) * PATCH_INNER + 1)) // PATCH_STRIDE + 1
N_PATCH_FEATURES *= (IMG_SIZE - ((PATCH_W - 1) * PATCH_INNER + 1)) // PATCH_STRIDE + 1

# critic-corrected operating point; see module docstring / results/critic_alpha.md
BEST_ALPHA = 1e-6

# geometries whose (feature x state) grid exceeds this use sparse CPT storage
DENSE_SLOT_LIMIT = 20_000_000


def _validate_X(X: np.ndarray, *, allow_empty: bool) -> np.ndarray:
    array = np.asarray(X)
    if array.ndim != 2 or array.shape[1] != N_FEATURES:
        raise ValueError(f"X must have shape (n_samples, {N_FEATURES})")
    if not allow_empty and array.shape[0] == 0:
        raise ValueError("X must contain at least one sample")
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError("X must contain numeric pixel values")
    if not np.all(np.isfinite(array)):
        raise ValueError("X contains NaN or infinite values")
    if array.size and (np.min(array) < 0 or np.max(array) > 255):
        raise ValueError("pixel values must be in the range [0, 255]")
    return array


def _validate_y(y: np.ndarray, n_samples: int) -> np.ndarray:
    y_array = np.asarray(y)
    if y_array.ndim != 1 or y_array.shape[0] != n_samples:
        raise ValueError("y must have shape (n_samples,) and match X")
    if not np.issubdtype(y_array.dtype, np.number):
        raise TypeError("y must contain numeric labels")
    if not np.all(np.isfinite(y_array)):
        raise ValueError("y contains NaN or infinite values")
    y_int = y_array.astype(np.int64)
    if not np.array_equal(y_array, y_int) or np.any((y_int < 0) | (y_int >= N_CLASSES)):
        raise ValueError("labels must be integers in the range [0, 9]")
    return y_int


def _patch_states(X: np.ndarray) -> np.ndarray:
    """Encode every dilated 4x4 binary patch as one integer state in [0, 2^16).

    X: (n, 784) raw pixels -> (n, 484) int64 states.  The ink bit of the
    (i, j)-th pixel occupies bit (i*PATCH_W + j).
    """
    n_samples = X.shape[0]
    image = X.reshape(n_samples, IMG_SIZE, IMG_SIZE)
    ink = image >= BINARY_THRESHOLD
    grid = (IMG_SIZE - ((PATCH_H - 1) * PATCH_INNER + 1)) // PATCH_STRIDE + 1
    states = np.zeros((n_samples, grid, grid), dtype=np.int64)
    bit = 0
    for i in range(PATCH_H):
        for j in range(PATCH_W):
            rows = slice(i * PATCH_INNER, i * PATCH_INNER + (grid - 1) * PATCH_STRIDE + 1,
                         PATCH_STRIDE)
            cols = slice(j * PATCH_INNER, j * PATCH_INNER + (grid - 1) * PATCH_STRIDE + 1,
                         PATCH_STRIDE)
            states |= ink[:, rows, cols].astype(np.int64) << bit
            bit += 1
    return states.reshape(n_samples, grid * grid)


class NaiveBayesClassifier:
    """Naive Bayes over 484 dilated 4x4 binary patch patterns.

    Protocol: ``__init__(alpha=1e-6)``, ``fit(X, y) -> self``,
    ``predict(X) -> labels``, ``predict_proba(X)``.  X is the raw
    (n_samples, 784) pixel matrix with values 0-255.

    Prediction is ``argmax_y [ log P(y) + sum_j log P(z_j | y) ]`` with the 484
    patch states conditionally independent given the class.
    """

    def __init__(self, alpha: float = BEST_ALPHA, dense_slot_limit: int = DENSE_SLOT_LIMIT):
        if not np.isscalar(alpha) or not np.isfinite(alpha) or alpha <= 0:
            raise ValueError("alpha must be a finite positive number")
        self.alpha = float(alpha)
        self.dense_slot_limit = int(dense_slot_limit)

    # ------------------------------------------------------------------
    def fit(self, X, y):
        X_array = _validate_X(X, allow_empty=False)
        y_int = _validate_y(y, X_array.shape[0])
        states = _patch_states(X_array)
        n_samples = X_array.shape[0]

        class_count = np.bincount(y_int, minlength=N_CLASSES).astype(np.float64)
        self.class_count_ = class_count
        offsets = np.arange(N_PATCH_FEATURES, dtype=np.int64) * N_STATES
        flat = (states + offsets).ravel()
        keys = np.unique(flat)
        counts = np.empty((N_CLASSES, keys.size), dtype=np.uint32)
        for class_index in range(N_CLASSES):
            sub = (states[y_int == class_index] + offsets).ravel()
            position = np.searchsorted(keys, sub)
            counts[class_index] = np.bincount(position, minlength=keys.size).astype(np.uint32)
        self.observed_keys_ = keys
        self.observed_counts_ = counts
        self.n_observed_states_ = int(keys.size)
        self.n_train_ = n_samples
        self.is_fitted_ = True
        return self

    def _check_fitted(self):
        if not getattr(self, "is_fitted_", False):
            raise RuntimeError("the classifier must be fitted before prediction")

    # ------------------------------------------------------------------
    def _alpha_tables(self, alpha):
        """Per-class log CPT for the observed pairs + the shared unseen value."""
        denom = self.class_count_ + N_STATES * alpha
        default = np.log(alpha / denom)                          # (C,)
        observed = np.log((self.observed_counts_.astype(np.float64) + alpha) / denom[:, None])
        prior = np.log((self.class_count_ + alpha) / (self.n_train_ + N_CLASSES * alpha))
        return observed, default, prior

    def _joint_log_likelihood(self, X, *, alpha=None):
        self._check_fitted()
        X_array = _validate_X(X, allow_empty=True)
        if X_array.shape[0] == 0:
            return np.empty((0, N_CLASSES), dtype=np.float64)
        alpha = self.alpha if alpha is None else float(alpha)

        states = _patch_states(X_array)
        n_samples = states.shape[0]
        keys = self.observed_keys_
        offsets = (np.arange(N_PATCH_FEATURES, dtype=np.int64) * N_STATES)[None, :]
        flat = states + offsets                                  # (n, F)
        position = np.searchsorted(keys, flat)
        np.clip(position, 0, keys.size - 1, out=position)
        found = keys[position] == flat

        observed, default, prior = self._alpha_tables(alpha)
        scores = np.tile(N_PATCH_FEATURES * default + prior, (n_samples, 1))
        if found.any():
            sample_index, feature_index = np.nonzero(found)
            key_index = position[sample_index, feature_index]
            correction = observed[:, key_index] - default[:, None]
            for class_index in range(N_CLASSES):
                scores[:, class_index] += np.bincount(
                    sample_index, weights=correction[class_index], minlength=n_samples
                )
        return scores

    def predict(self, X):
        """Return the maximum a-posteriori class for each sample."""
        return np.argmax(self._joint_log_likelihood(X), axis=1).astype(np.int64)

    def predict_proba(self, X):
        """Return normalized posterior probabilities for all ten classes."""
        log_scores = self._joint_log_likelihood(X)
        if log_scores.shape[0] == 0:
            return np.empty((0, N_CLASSES), dtype=np.float64)
        shifted = log_scores - np.max(log_scores, axis=1, keepdims=True)
        probabilities = np.exp(shifted)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        return probabilities

    # ------------------------------------------------------------------
    def check_probability_mass(self, rtol: float = 1e-9) -> np.ndarray:
        """Verify every per-patch conditional distribution sums to 1.

        The dense CPT is never materialised (484 x 65,536 per class would not
        fit), so the sum is checked in the compressed representation: the
        observed pairs contribute exp(log P), and each of the unseen pairs
        contributes the shared default.  Returns the per-class total mass, which
        should equal N_PATCH_FEATURES.
        """
        self._check_fitted()
        observed, default, _ = self._alpha_tables(self.alpha)
        observed_mass = np.exp(observed).sum(axis=1)
        unseen = N_PATCH_FEATURES * N_STATES - self.n_observed_states_
        total = observed_mass + unseen * np.exp(default)
        np.testing.assert_allclose(total, float(N_PATCH_FEATURES), rtol=rtol)
        return total


# Backwards-compatible alias
CriticAlphaBest = NaiveBayesClassifier
