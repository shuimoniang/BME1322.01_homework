"""Hand-designed small Bayesian network classifier for MNIST.

Final iterated design (v5).  The model uses twelve discrete semantic features,
including two content-relative densities measured in the image's own ink
bounding box.  The M6 hand-designed DAG and class-conditional-marginal CPT
backoff are retained.  The bbox frame removes writing-position and scale noise
from the middle-right and bottom-right contour cells while keeping all feature
cardinalities within the assignment's 2--5-state rule.

The structure is hand-authored: it was selected from explicitly named manual
families; no automatic structure-learning algorithm was used.  All probability
modelling, CPT estimation and inference are implemented with NumPy only.
"""

from __future__ import annotations

import numpy as np


N_CLASSES = 10
N_PIXELS = 28 * 28
BINARY_THRESHOLD = 64

FEATURE_NAMES = (
    "density_top_left",
    "density_top_center",
    "density_top_right",
    "density_middle_left",
    "density_middle_center",
    "bbox_density_middle_right",
    "density_bottom_left",
    "bbox_density_bottom_right",
    "euler_topology",
    "vertical_runs",
    "aspect_ratio",
    "vertical_dominance",
)

# Grid cell [row, col) bounds covering the central 15x15 region.
_GRID_BOUNDS = (7, 12, 17, 22)

# Optional feature parent of each feature (-1 = class-only parent).  The
# class Y is also a parent of every feature, so each node has at most two
# parents in total (M6 DAG, final v5 design).  The eleven feature-parent
# edges are: the top row chained left -> right (top_left -> top_center ->
# top_right); the middle band hung off the Euler node (euler ->
# middle_left -> middle_center, and euler -> bbox_middle_right); the
# bottom band closed right -> left (bbox_middle_right -> bbox_bottom_right
# -> bottom_left); the diagonal balance edge (bbox_bottom_right ->
# top_left); and the semantic chain (euler -> vertical_runs ->
# {aspect_ratio, vertical_dominance}).  There are no edges between the
# grid rows themselves.  bottom_center is omitted: it is the most
# redundant cell.
FEATURE_PARENTS = (7, 0, 1, 8, 3, 8, 7, 5, -1, 8, 9, 9)
FEATURE_CARDINALITIES = (
    5, 5, 5, 5, 5, 5, 5, 5,  # 8 grid densities
    4,   # euler_topology: -1, 0, 1, 2
    5,   # vertical_runs
    4,   # aspect_ratio
    4,   # vertical_dominance
)


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


def _discretize(values: np.ndarray, n_states: int, thresholds=None):
    """State 0 = exactly zero; positive values use supplied cut points."""
    values = np.asarray(values)
    if thresholds is None:
        positives = values[values > 0]
        if positives.size == 0:
            thresholds = np.linspace(0.0, 1.0, n_states + 1)[1:-1]
        else:
            thresholds = np.quantile(
                positives, np.linspace(0, 1, n_states)[1:-1]
            )
    discrete = np.searchsorted(thresholds, values, side="right") + (values > 0)
    return discrete.astype(np.int64), np.asarray(thresholds, dtype=np.float64)


class BayesNetClassifier:
    """Small Bayesian Network Protocol v1 with a fixed hand-designed DAG."""

    def __init__(self, alpha: float = 1.0):
        if not np.isscalar(alpha) or not np.isfinite(alpha) or alpha <= 0:
            raise ValueError("alpha must be a finite positive number")
        self.alpha = float(alpha)

    # ------------------------------------------------------------------
    # feature extraction
    # ------------------------------------------------------------------
    @staticmethod
    def _bbox_bounds(binary: np.ndarray):
        """Return each image's ink bounding-box coordinates and extent."""
        indices = np.arange(28)[None, :]
        rows_any = binary.any(axis=2)
        cols_any = binary.any(axis=1)
        top = np.where(rows_any, indices, 28).min(axis=1)
        bottom = np.where(rows_any, indices, -1).max(axis=1)
        left = np.where(cols_any, indices, 28).min(axis=1)
        right = np.where(cols_any, indices, -1).max(axis=1)
        height = np.clip(bottom - top + 1, 1, 28)
        width = np.clip(right - left + 1, 1, 28)
        return top, bottom, left, right, height, width

    @classmethod
    def _bbox_band_density(cls, binary: np.ndarray, row_band: int, col_band: int):
        """Density in one cell of each image's normalised 3x3 ink box."""
        top, bottom, left, right, height, width = cls._bbox_bounds(binary)
        rows = np.arange(28)[None, :, None]
        cols = np.arange(28)[None, None, :]
        row_pos = np.clip(rows - top[:, None, None], 0, 27)
        col_pos = np.clip(cols - left[:, None, None], 0, 27)
        row_state = np.clip((row_pos * 3) // height[:, None, None], 0, 2)
        col_state = np.clip((col_pos * 3) // width[:, None, None], 0, 2)
        selected = (row_state == row_band) & (col_state == col_band)
        inside = (rows >= top[:, None, None]) & (rows <= bottom[:, None, None])
        inside = inside & (cols >= left[:, None, None]) & (cols <= right[:, None, None])
        ink_count = np.count_nonzero(selected & binary, axis=(1, 2)).astype(np.float64)
        area = np.count_nonzero(selected & inside, axis=(1, 2)).astype(np.float64)
        return ink_count / np.maximum(area, 1.0)

    def _continuous_features(self, X_array: np.ndarray) -> dict:
        images = X_array.reshape(-1, 28, 28).astype(np.float64) / 255.0
        binary = X_array.reshape(-1, 28, 28) > BINARY_THRESHOLD

        cells = (
            (7, 12, 7, 12), (7, 12, 12, 17), (7, 12, 17, 22),
            (12, 17, 7, 12), (12, 17, 12, 17), (12, 17, 17, 22),
            (17, 22, 7, 12), (17, 22, 17, 22),
        )
        densities = np.empty((images.shape[0], len(cells)), dtype=np.float64)
        for index, (r0, r1, c0, c1) in enumerate(cells):
            if index == 5:
                densities[:, index] = self._bbox_band_density(binary, 1, 2)
            elif index == 7:
                densities[:, index] = self._bbox_band_density(binary, 2, 2)
            else:
                densities[:, index] = images[:, r0:r1, c0:c1].mean(axis=(1, 2))

        rows_any = np.any(binary, axis=2)
        cols_any = np.any(binary, axis=1)
        row_idx = np.arange(28)[None, :]
        col_idx = np.arange(28)[None, :]
        first_row = np.where(rows_any, row_idx, 28).min(axis=1)
        last_row = np.where(rows_any, row_idx, -1).max(axis=1)
        first_col = np.where(cols_any, col_idx, 28).min(axis=1)
        last_col = np.where(cols_any, col_idx, -1).max(axis=1)
        height = np.clip(last_row - first_row + 1, 1, 28)
        width = np.clip(last_col - first_col + 1, 1, 28)
        aspect = width / height

        gx = np.zeros_like(images)
        gy = np.zeros_like(images)
        gx[:, 1:-1, :] = images[:, 2:, :] - images[:, :-2, :]
        gy[:, :, 1:-1] = images[:, :, 2:] - images[:, :, :-2]
        gx_abs = np.abs(gx).sum(axis=(1, 2))
        gy_abs = np.abs(gy).sum(axis=(1, 2))
        vertical_dominance = gy_abs / np.maximum(gx_abs + gy_abs, 1e-9)

        return {
            "densities": densities,
            "euler": _euler_states(binary),
            "vertical_runs": self._vertical_runs(binary),
            "aspect": aspect,
            "vertical_dominance": vertical_dominance,
        }

    @staticmethod
    def _vertical_runs(binary: np.ndarray) -> np.ndarray:
        run_starts = np.count_nonzero(binary[:, 0, :], axis=1)
        run_starts += np.count_nonzero(
            (~binary[:, :-1, :]) & binary[:, 1:, :], axis=(1, 2)
        )
        occupied_columns = np.count_nonzero(binary.any(axis=1), axis=1)
        return run_starts / np.maximum(occupied_columns, 1)

    # ------------------------------------------------------------------
    # protocol
    # ------------------------------------------------------------------
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

        continuous = self._continuous_features(X_array)
        self._fit_thresholds(continuous)
        features = self._discretize_features(continuous)
        self._fit_cpts(features, y_int)
        self.is_fitted_ = True
        return self

    def _fit_thresholds(self, feats: dict) -> None:
        self.density_thresholds_ = []
        for j in range(feats["densities"].shape[1]):
            _, thresholds = _discretize(feats["densities"][:, j], 5)
            self.density_thresholds_.append(thresholds)
        self.semantic_thresholds_ = []
        for name, card in (
            ("vertical_runs", 5),
            ("aspect", 4),
            ("vertical_dominance", 4),
        ):
            _, thresholds = _discretize(feats[name], card)
            self.semantic_thresholds_.append(thresholds)

    def _discretize_features(self, feats: dict) -> np.ndarray:
        n = feats["densities"].shape[0]
        features = np.empty((n, len(FEATURE_NAMES)), dtype=np.int64)
        for j in range(feats["densities"].shape[1]):
            features[:, j] = _discretize(
                feats["densities"][:, j], 5, self.density_thresholds_[j]
            )[0]
        features[:, 8] = feats["euler"]
        for col, name, card in (
            (9, "vertical_runs", 5),
            (10, "aspect", 4),
            (11, "vertical_dominance", 4),
        ):
            features[:, col] = _discretize(
                feats[name], card, self.semantic_thresholds_[col - 9]
            )[0]
        return features

    def _fit_cpts(self, features: np.ndarray, y_int: np.ndarray) -> None:
        alpha = self.alpha
        class_count = np.bincount(y_int, minlength=N_CLASSES).astype(np.float64)
        self.class_count_ = class_count
        self.class_log_prior_ = np.log(
            (class_count + alpha) / (features.shape[0] + N_CLASSES * alpha)
        )
        cpts = []
        for node, parent in enumerate(FEATURE_PARENTS):
            cardinality = FEATURE_CARDINALITIES[node]
            if parent < 0:
                counts = np.zeros((N_CLASSES, cardinality), dtype=np.float64)
                np.add.at(counts, (y_int, features[:, node]), 1)
                denominator = counts.sum(axis=-1, keepdims=True) + cardinality * alpha
                cpts.append(np.log((counts + alpha) / denominator))
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
                # backoff: class-conditional marginal as Dirichlet prior mean
                marginal = np.zeros((N_CLASSES, cardinality), dtype=np.float64)
                np.add.at(marginal, (y_int, features[:, node]), 1)
                marginal = (marginal + alpha) / (
                    class_count[:, None] + cardinality * alpha
                )
                n_cp = counts.sum(axis=-1, keepdims=True)
                cpts.append(
                    np.log((counts + alpha * marginal[:, None, :]) / (n_cp + alpha))
                )
        self.cpt_log_prob_ = tuple(cpts)

    def _check_fitted(self) -> None:
        if not getattr(self, "is_fitted_", False):
            raise RuntimeError("the classifier must be fitted before prediction")

    def _joint_log_likelihood(self, X) -> np.ndarray:
        self._check_fitted()
        X_array = _validate_X(X, allow_empty=True)
        if X_array.shape[0] == 0:
            return np.empty((0, N_CLASSES), dtype=np.float64)
        features = self._discretize_features(self._continuous_features(X_array))
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


def extract_features(X):
    """Extract a deterministic, reusable 12-feature representation.

    The classifier learns training quantiles in ``fit``.  This public helper
    intentionally uses fixed canonical cut points so the same image always
    receives the same output regardless of what other samples share the call.
    """
    X_array = _validate_X(X, allow_empty=True)
    if X_array.shape[0] == 0:
        return np.empty((0, len(FEATURE_NAMES)), dtype=np.int64)
    model = BayesNetClassifier()
    feats = model._continuous_features(X_array)
    n = feats["densities"].shape[0]
    features = np.empty((n, len(FEATURE_NAMES)), dtype=np.int64)
    density_cutpoints = np.array([0.15, 0.30, 0.50], dtype=np.float64)
    for j in range(feats["densities"].shape[1]):
        features[:, j] = _discretize(
            feats["densities"][:, j], 5, density_cutpoints
        )[0]
    features[:, 8] = feats["euler"]
    fixed_cutpoints = {
        "vertical_runs": np.array([1.0, 1.5, 2.5], dtype=np.float64),
        "aspect": np.array([0.75, 1.25], dtype=np.float64),
        "vertical_dominance": np.array([0.40, 0.60], dtype=np.float64),
    }
    for col, name, card in (
        (9, "vertical_runs", 5),
        (10, "aspect", 4),
        (11, "vertical_dominance", 4),
    ):
        features[:, col] = _discretize(
            feats[name], card, fixed_cutpoints[name]
        )[0]
    return features
