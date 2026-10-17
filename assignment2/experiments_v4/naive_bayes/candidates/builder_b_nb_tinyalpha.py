"""Patch-pattern Naive Bayes with a data-driven small Lidstone constant.

Builder B candidate (BME1322.01 Assignment 2, experiments_v4).

This is *exactly* the incumbent ``naive_bayes.NaiveBayesClassifier`` model -- the
same 729 overlapping 2x2 patches, the same 5 gray-level bands per pixel, the same
625-state joint encoding, the same naive-Bayes product
``argmax_y [ log P(y) + sum_j log P(z_j | y) ]`` and the same Lidstone (add-alpha)
smoothing.  Nothing about the feature geometry or the independence structure is
changed.

The single change is the value of the additive smoothing constant ``alpha``:

    incumbent : alpha = 0.05   -> validation 0.9298
    candidate : alpha = 1e-4   -> validation 0.9395

Why: 85.9% of the 10 x 729 x 625 CPT cells have zero count, yet only ~10% of the
patch states actually occurring in held-out data are unseen for their class.  A
uniform add-0.05 prior therefore injects much more probability mass into the
625-1 *never-observed* states than into the handful of states that matter; it
biases the likelihood in favour of classes whose CPTs are denser (e.g. "1",
which fires far fewer patches).  Shrinking alpha removes that bias while keeping
the standard Lidstone form (alpha > 0), so every class still has full support and
no log(0) can occur.  The validation curve is flat between alpha = 1e-5 and
1e-4 and falls for alpha < 1e-6, i.e. 1e-4 sits safely on the plateau of the
asymptotic regime rather than at a knife-edge.

Protocol: ``__init__(alpha=1e-4)``, ``fit(X, y) -> self``, ``predict(X)``,
``predict_proba(X)``.  ``X`` is the raw (n_samples, 784) pixel matrix, values
0-255 (uint8 or any numeric dtype).  Drop-in replacement for the incumbent.
"""

from __future__ import annotations

import numpy as np

N_CLASSES = 10
N_FEATURES = 28 * 28
PIXEL_EDGES = np.array([51, 102, 153, 204], dtype=np.int64)
N_LEVELS = 5
N_STATES = N_LEVELS ** 4          # 625 joint states per patch
N_PATCH_FEATURES = 27 * 27        # 729 overlapping 2x2 patches

# Validation-selected Lidstone constant (see module docstring).
DEFAULT_ALPHA = 1e-4


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
    """Encode every overlapping 2x2 patch as one categorical state in [0, 625).

    Identical encoding and patch order to the incumbent.  ``X``: (n, 784) raw
    pixels -> (n, 729) int64 states.
    """
    n = X.shape[0]
    img = X.reshape(n, 28, 28)
    bands = np.searchsorted(PIXEL_EDGES, img, side="right").astype(np.int64)
    states = (
        bands[:, :-1, :-1] * (N_LEVELS ** 3)
        + bands[:, :-1, 1:] * (N_LEVELS ** 2)
        + bands[:, 1:, :-1] * N_LEVELS
        + bands[:, 1:, 1:]
    ).reshape(n, N_PATCH_FEATURES)
    return states


class NaiveBayesClassifier:
    """Patch-pattern Naive Bayes, drop-in compatible with the incumbent.

    Protocol: ``__init__(alpha=1e-4)``, ``fit(X, y) -> self``,
    ``predict(X) -> labels``, ``predict_proba(X)``.  ``X`` is the raw
    (n_samples, 784) pixel matrix with values 0-255.
    """

    def __init__(self, alpha: float = DEFAULT_ALPHA):
        if not np.isscalar(alpha) or not np.isfinite(alpha) or alpha <= 0:
            raise ValueError("alpha must be a finite positive number")
        self.alpha = float(alpha)

    def fit(self, X, y):
        """Estimate class priors and per-patch conditional distributions."""
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
