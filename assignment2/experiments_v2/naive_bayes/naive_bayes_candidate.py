"""Patch-pattern Naive Bayes classifier for MNIST.

The probability model is still a generative naive Bayes:

    P(y | X)  proportional to  P(y) * prod_j P(z_j | y),

where the features z_j are the 729 overlapping 2x2 patches of the
28x28 image.  Each patch's four pixels are independently quantized to
5 gray-level bands (0, 1-50, 51-101, 102-152, 153-255 boundaries via
integer thresholds) and the tuple of four band indices is encoded as a
single categorical state out of 5^4 = 625 possibilities.  Conditional
independence is assumed across patches (not raw pixels), which captures
local pixel correlations inside each patch while keeping the naive
Bayes factorization intact.  Everything is implemented with NumPy only;
no ready-made classifier is used.

Compared to the previous baseline (784 individual pixels quantized to 7
gray levels), this reaches substantially higher accuracy because joint
2x2 patterns encode stroke transitions, endpoints and local curvature
that single pixels cannot express.
"""

from __future__ import annotations

import numpy as np


N_CLASSES = 10
N_FEATURES = 28 * 28
PATCH_SIZE = 2
# 5 gray-level bands per pixel -> 625 patch states
PIXEL_EDGES = np.array([51, 102, 153, 204], dtype=np.int64)
N_LEVELS = 5
N_STATES = N_LEVELS ** 4
N_PATCH_FEATURES = 27 * 27  # overlapping 2x2 patches


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
    """Encode every overlapping 2x2 patch as one categorical state.

    X: (n, 784) raw pixels.  Returns (n, 729) int64 states in [0, 625).
    """
    n = X.shape[0]
    img = X.reshape(n, 28, 28)
    # (n, 5, 5-ish) quantized bands per pixel, computed once
    bands = np.searchsorted(PIXEL_EDGES, img, side="right").astype(np.int64)
    states = np.empty((n, N_PATCH_FEATURES), dtype=np.int64)
    col = 0
    for r in range(27):
        for c in range(27):
            states[:, col] = (
                ((bands[:, r, c] * N_LEVELS + bands[:, r, c + 1]) * N_LEVELS
                 + bands[:, r + 1, c]) * N_LEVELS
                + bands[:, r + 1, c + 1]
            )
            col += 1
    return states


class NaiveBayesClassifier:
    """Naive Bayes Protocol v1 over 2x2 patch gray-level patterns.

    Protocol: ``__init__(alpha=1.0)``, ``fit(X, y) -> self``,
    ``predict(X) -> labels``, ``predict_proba(X)``.  X is the raw
    (n_samples, 784) pixel matrix with values 0-255.
    """

    def __init__(self, alpha: float = 1.0):
        if not np.isscalar(alpha) or not np.isfinite(alpha) or alpha <= 0:
            raise ValueError("alpha must be a finite positive number")
        self.alpha = float(alpha)

    def fit(self, X, y):
        """Estimate priors and per-patch conditional distributions."""
        X_array = _validate_X(X, allow_empty=False)
        y_int = _validate_y(y, X_array.shape[0])

        states = _patch_states(X_array)
        class_count = np.bincount(y_int, minlength=N_CLASSES).astype(np.float64)
        alpha = self.alpha
        self.class_count_ = class_count
        self.class_log_prior_ = np.log(
            (class_count + alpha) / (X_array.shape[0] + N_CLASSES * alpha)
        )
        self.feature_log_prob_ = np.empty(
            (N_CLASSES, N_PATCH_FEATURES, N_STATES), dtype=np.float64
        )
        offsets_base = np.arange(N_PATCH_FEATURES, dtype=np.int64) * N_STATES
        for class_index in range(N_CLASSES):
            sub = states[y_int == class_index]
            counts = np.bincount(
                (sub + offsets_base).ravel(),
                minlength=N_PATCH_FEATURES * N_STATES,
            ).reshape(N_PATCH_FEATURES, N_STATES)
            self.feature_log_prob_[class_index] = np.log(
                (counts + alpha) / (class_count[class_index] + N_STATES * alpha)
            )
        self.is_fitted_ = True
        return self

    def _check_fitted(self) -> None:
        if not getattr(self, "is_fitted_", False):
            raise RuntimeError("the classifier must be fitted before prediction")

    def _joint_log_likelihood(self, X, *, batch_size: int = 2048) -> np.ndarray:
        self._check_fitted()
        X_array = _validate_X(X, allow_empty=True)
        if X_array.shape[0] == 0:
            return np.empty((0, N_CLASSES), dtype=np.float64)

        states = _patch_states(X_array)
        n = states.shape[0]
        scores = np.empty((n, N_CLASSES), dtype=np.float64)
        feature_index = np.arange(N_PATCH_FEATURES)[None, :]
        for start in range(0, n, batch_size):
            stop = min(start + batch_size, n)
            batch = states[start:stop]
            # (C, 729, 625) gather with (batch, 729) index -> (batch, C, 729)
            likelihood = self.feature_log_prob_[:, feature_index, batch].transpose(1, 0, 2)
            scores[start:stop] = likelihood.sum(axis=2) + self.class_log_prior_[None, :]
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
