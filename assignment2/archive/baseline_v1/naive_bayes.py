"""Discrete pixel Naive Bayes classifier for MNIST.

The probability model is implemented with NumPy only.  No ready-made
classification implementation is used.
"""

from __future__ import annotations

import numpy as np


N_CLASSES = 10
N_FEATURES = 28 * 28
N_BINS = 7
_BIN_THRESHOLDS = np.array([1, 16, 32, 64, 128, 192], dtype=np.float64)


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


def _discretize(X: np.ndarray) -> np.ndarray:
    """Map raw pixels to the seven documented intensity states."""
    return np.searchsorted(_BIN_THRESHOLDS, X, side="right").astype(np.uint8)


class NaiveBayesClassifier:
    """Naive Bayes Protocol v1 implemented with seven gray-level states."""

    def __init__(self, alpha: float = 1.0):
        if not np.isscalar(alpha) or not np.isfinite(alpha) or alpha <= 0:
            raise ValueError("alpha must be a finite positive number")
        self.alpha = float(alpha)

    def fit(self, X, y):
        """Estimate class priors and per-pixel conditional distributions."""
        X_array = _validate_X(X, allow_empty=False)
        y_array = np.asarray(y)
        if y_array.ndim != 1 or y_array.shape[0] != X_array.shape[0]:
            raise ValueError("y must have shape (n_samples,) and match X")
        if not np.issubdtype(y_array.dtype, np.number):
            raise TypeError("y must contain numeric labels")
        if not np.all(np.isfinite(y_array)):
            raise ValueError("y contains NaN or infinite values")
        y_int = y_array.astype(np.int64)
        if not np.array_equal(y_array, y_int) or np.any((y_int < 0) | (y_int >= N_CLASSES)):
            raise ValueError("labels must be integers in the range [0, 9]")

        X_discrete = _discretize(X_array)
        class_count = np.bincount(y_int, minlength=N_CLASSES).astype(np.float64)
        feature_count = np.zeros((N_CLASSES, N_FEATURES, N_BINS), dtype=np.float64)

        for class_index in range(N_CLASSES):
            class_rows = X_discrete[y_int == class_index]
            for state in range(N_BINS):
                feature_count[class_index, :, state] = np.count_nonzero(
                    class_rows == state, axis=0
                )

        alpha = self.alpha
        self.class_count_ = class_count
        self.class_log_prior_ = np.log(
            (class_count + alpha) / (X_array.shape[0] + N_CLASSES * alpha)
        )
        self.feature_log_prob_ = np.log(
            (feature_count + alpha)
            / (class_count[:, None, None] + N_BINS * alpha)
        )
        self.is_fitted_ = True
        return self

    def _check_fitted(self) -> None:
        if not getattr(self, "is_fitted_", False):
            raise RuntimeError("the classifier must be fitted before prediction")

    def _joint_log_likelihood(self, X, *, batch_size: int = 512) -> np.ndarray:
        self._check_fitted()
        X_array = _validate_X(X, allow_empty=True)
        if X_array.shape[0] == 0:
            return np.empty((0, N_CLASSES), dtype=np.float64)

        X_discrete = _discretize(X_array)
        scores = np.empty((X_discrete.shape[0], N_CLASSES), dtype=np.float64)
        feature_indices = np.arange(N_FEATURES)[None, :]
        for start in range(0, X_discrete.shape[0], batch_size):
            stop = min(start + batch_size, X_discrete.shape[0])
            batch = X_discrete[start:stop]
            for class_index in range(N_CLASSES):
                likelihood = self.feature_log_prob_[
                    class_index, feature_indices, batch
                ]
                scores[start:stop, class_index] = (
                    self.class_log_prior_[class_index] + likelihood.sum(axis=1)
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

