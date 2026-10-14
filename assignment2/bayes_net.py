"""Hand-designed small Bayesian network classifier for MNIST."""

from __future__ import annotations

import numpy as np


N_CLASSES = 10
N_PIXELS = 28 * 28
BINARY_THRESHOLD = 64

FEATURE_NAMES = (
    "global_ink",
    "top_left",
    "top_center",
    "top_right",
    "middle_left",
    "middle_center",
    "middle_right",
    "bottom_left",
    "bottom_center",
    "bottom_right",
    "euler_topology",
    "vertical_runs",
)

# Each value is an optional feature parent.  The class Y is also a parent of
# every feature, so each node has at most two parents in total.
FEATURE_PARENTS = (-1, 4, 5, 6, 0, 0, 0, 4, 5, 6, -1, 10)
FEATURE_CARDINALITIES = (5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 4, 5)

# Fixed cut points were calibrated on the MNIST training split only.  Sparse
# cells reserve state 0 for exactly zero ink; 5e-324 is the smallest positive
# float and separates zero from every positive density.
_DENSITY_THRESHOLDS = (
    (0.09286615, 0.11662765, 0.13844038, 0.16607343),
    (5e-324, 0.15262745, 0.31827451, 0.47356863),
    (0.21035294, 0.36407843, 0.47325490, 0.59623529),
    (5e-324, 0.22007843, 0.39137255, 0.53819608),
    (5e-324, 0.16627451, 0.35874510, 0.52454902),
    (0.22917647, 0.41113725, 0.53113725, 0.65945098),
    (0.06305882, 0.24031373, 0.38302745, 0.52257255),
    (5e-324, 0.12580392, 0.32611765, 0.52047059),
    (0.23836863, 0.39011765, 0.50258824, 0.62698039),
    (5e-324, 0.13600000, 0.32486275, 0.48188235),
)
_VERTICAL_RUN_THRESHOLDS = (1.33333333, 1.5625, 1.75, 2.0)
_GRID_BOUNDS = (7, 12, 17, 22)


def _validate_X(X: np.ndarray, *, allow_empty: bool) -> np.ndarray:
    array = np.asarray(X)
    if array.ndim != 2 or array.shape[1] != N_PIXELS:
        raise ValueError(f"X must have shape (n_samples, {N_PIXELS})")
    if not allow_empty and array.shape[0] == 0:
        raise ValueError("X must contain at least one sample")
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError("X must contain numeric pixel values")
    if not np.all(np.isfinite(array)):
        raise ValueError("X contains NaN or infinite values")
    if array.size and (np.min(array) < 0 or np.max(array) > 255):
        raise ValueError("pixel values must be in the range [0, 255]")
    return array


def _euler_states(binary_images: np.ndarray) -> np.ndarray:
    """Estimate components minus holes with a 2x2 digital-topology formula."""
    padded = np.pad(binary_images, ((0, 0), (1, 1), (1, 1)))
    upper_left = padded[:, :-1, :-1]
    lower_left = padded[:, 1:, :-1]
    upper_right = padded[:, :-1, 1:]
    lower_right = padded[:, 1:, 1:]
    block_sum = (
        upper_left.astype(np.uint8)
        + lower_left
        + upper_right
        + lower_right
    )
    diagonal_pairs = (
        (upper_left & lower_right & ~lower_left & ~upper_right)
        | (lower_left & upper_right & ~upper_left & ~lower_right)
    )
    q1 = np.count_nonzero(block_sum == 1, axis=(1, 2))
    q3 = np.count_nonzero(block_sum == 3, axis=(1, 2))
    qd = np.count_nonzero(diagonal_pairs, axis=(1, 2))
    euler = np.rint((q1 - q3 + 2 * qd) / 4.0).astype(np.int64)
    return np.clip(euler, -1, 2) + 1


def extract_features(X):
    """Extract the 12 documented discrete features from flattened images."""
    X_array = _validate_X(X, allow_empty=True)
    if X_array.shape[0] == 0:
        return np.empty((0, len(FEATURE_NAMES)), dtype=np.int64)

    images = X_array.reshape(-1, 28, 28).astype(np.float64) / 255.0
    densities = [images.mean(axis=(1, 2))]
    intervals = list(zip(_GRID_BOUNDS[:-1], _GRID_BOUNDS[1:]))
    for row_start, row_stop in intervals:
        for col_start, col_stop in intervals:
            densities.append(
                images[:, row_start:row_stop, col_start:col_stop].mean(axis=(1, 2))
            )

    discrete = np.empty((images.shape[0], len(FEATURE_NAMES)), dtype=np.int64)
    for feature_index, (values, thresholds) in enumerate(
        zip(densities, _DENSITY_THRESHOLDS)
    ):
        discrete[:, feature_index] = np.searchsorted(
            thresholds, values, side="right"
        )

    binary = X_array.reshape(-1, 28, 28) > BINARY_THRESHOLD
    discrete[:, 10] = _euler_states(binary)

    run_starts = np.count_nonzero(binary[:, 0, :], axis=1)
    run_starts += np.count_nonzero(
        (~binary[:, :-1, :]) & binary[:, 1:, :], axis=(1, 2)
    )
    occupied_columns = np.count_nonzero(binary.any(axis=1), axis=1)
    mean_vertical_runs = run_starts / np.maximum(occupied_columns, 1)
    discrete[:, 11] = np.searchsorted(
        _VERTICAL_RUN_THRESHOLDS, mean_vertical_runs, side="right"
    )
    return discrete


class BayesNetClassifier:
    """Small Bayesian Network Protocol v1 with a fixed hand-designed DAG."""

    def __init__(self, alpha: float = 1.0):
        if not np.isscalar(alpha) or not np.isfinite(alpha) or alpha <= 0:
            raise ValueError("alpha must be a finite positive number")
        self.alpha = float(alpha)

    def fit(self, X, y):
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

        features = extract_features(X_array)
        alpha = self.alpha
        class_count = np.bincount(y_int, minlength=N_CLASSES).astype(np.float64)
        self.class_count_ = class_count
        self.class_log_prior_ = np.log(
            (class_count + alpha) / (features.shape[0] + N_CLASSES * alpha)
        )

        cpts = []
        for node, (parent, cardinality) in enumerate(
            zip(FEATURE_PARENTS, FEATURE_CARDINALITIES)
        ):
            if parent < 0:
                counts = np.zeros((N_CLASSES, cardinality), dtype=np.float64)
                np.add.at(counts, (y_int, features[:, node]), 1)
            else:
                parent_cardinality = FEATURE_CARDINALITIES[parent]
                counts = np.zeros(
                    (N_CLASSES, parent_cardinality, cardinality), dtype=np.float64
                )
                np.add.at(
                    counts,
                    (y_int, features[:, parent], features[:, node]),
                    1,
                )
            denominator = counts.sum(axis=-1, keepdims=True) + cardinality * alpha
            cpts.append(np.log((counts + alpha) / denominator))

        self.cpt_log_prob_ = tuple(cpts)
        self.is_fitted_ = True
        return self

    def _check_fitted(self) -> None:
        if not getattr(self, "is_fitted_", False):
            raise RuntimeError("the classifier must be fitted before prediction")

    def _joint_log_likelihood(self, X) -> np.ndarray:
        self._check_fitted()
        features = extract_features(X)
        scores = np.broadcast_to(
            self.class_log_prior_, (features.shape[0], N_CLASSES)
        ).copy()
        for node, parent in enumerate(FEATURE_PARENTS):
            log_cpt = self.cpt_log_prob_[node]
            if parent < 0:
                scores += log_cpt[:, features[:, node]].T
            else:
                scores += log_cpt[
                    :, features[:, parent], features[:, node]
                ].T
        return scores

    def predict(self, X):
        return np.argmax(self._joint_log_likelihood(X), axis=1).astype(np.int64)

    def predict_proba(self, X):
        log_scores = self._joint_log_likelihood(X)
        if log_scores.shape[0] == 0:
            return np.empty((0, N_CLASSES), dtype=np.float64)
        shifted = log_scores - np.max(log_scores, axis=1, keepdims=True)
        probabilities = np.exp(shifted)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        return probabilities

