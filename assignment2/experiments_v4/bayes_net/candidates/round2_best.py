"""Round-2 NEW CHAMPION: content-relative (bounding-box-normalised) grid densities.

WHAT THIS CHANGES vs M6
=======================
The DAG, the feature count, the cardinalities, the discretisation rule and the
smoothing scheme are ALL unchanged from M6.  Exactly TWO of the twelve feature
definitions change, and only in their *coordinate frame*:

  slot 5  density_middle_right : absolute central window -> bbox-normalised
  slot 7  density_bottom_right : absolute central window -> bbox-normalised

The incumbent (and M6) measure the eight cell densities in FIXED image
coordinates (rows/cols 7..22 of the 28x28 frame).  A digit written small and
centred, or large and shifted, therefore lands its ink in different fixed
windows: the absolute cell mean mixes "shape" with "where and how big the digit
was written".  The two replaced features measure the SAME quantity -- ink
density of the middle-right / bottom-right third of the glyph -- in the image's
OWN ink bounding box, so position and scale are normalised away and only shape
survives.  This is a genuine re-definition of the variable (a single discrete
feature with a clear semantic and 5 states), not a hyper-parameter tweak.

Measured (8 seeds; pooled McNemar on the 7 held-out seeds as in
results/round2_frontier.md):
  M6                             0.8630 mean
  this module (bbMR + bbBR)      0.8778 mean   (+1.48 pt, positive on 8/8 seeds)

Public API is identical to assignment2/bayes_net.py:
  BayesNetClassifier, FEATURE_NAMES, FEATURE_CARDINALITIES, FEATURE_PARENTS,
  extract_features.  All probability modelling is NumPy only.
"""

from __future__ import annotations

import numpy as np


N_CLASSES = 10
N_PIXELS = 28 * 28
BINARY_THRESHOLD = 64

# The two features measured in the content-relative frame (0-based slot indices).
BBOX_RELATIVE_SLOTS = (5, 7)

FEATURE_NAMES = (
    "density_top_left",
    "density_top_center",
    "density_top_right",
    "density_middle_left",
    "density_middle_center",
    "bbox_density_middle_right",   # CHANGED: bbox-normalised
    "density_bottom_left",
    "bbox_density_bottom_right",   # CHANGED: bbox-normalised
    "euler_topology",
    "vertical_runs",
    "aspect_ratio",
    "vertical_dominance",
)

# Absolute grid cell [row0, row1) x [col0, col1) in fixed image coordinates.
_ABS_CELLS = (
    (7, 12, 7, 12),    # top_left
    (7, 12, 12, 17),   # top_center
    (7, 12, 17, 22),   # top_right
    (12, 17, 7, 12),   # middle_left
    (12, 17, 12, 17),  # middle_center
    (12, 17, 17, 22),  # middle_right  (overridden for slot 5)
    (17, 22, 7, 12),   # bottom_left
    (17, 22, 17, 22),  # bottom_right  (overridden for slot 7)
)
# Slot -> (row band, col band) inside the normalised 3x3 bbox grid.
_BBOX_CELLS = {
    5: (1, 2),   # middle band, right column
    7: (2, 2),   # bottom band, right column
}

# M6 DAG, unchanged.  Y is a parent of every feature; each feature has at most
# one feature parent.  br -> tl (diagonal ink balance), euler -> ml and
# euler -> mr (middle band conditioned on global structure), plus the reading
# chains and the semantic orientation chain.
FEATURE_PARENTS = (7, 0, 1, 8, 3, 8, 7, 5, -1, 8, 9, 9)

FEATURE_CARDINALITIES = (
    5, 5, 5, 5, 5, 5, 5, 5,  # 8 grid densities (2 of them bbox-normalised)
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


def _bbox_bounds(binary: np.ndarray):
    """(top, bottom, left, right, height, width) of each image's ink bbox."""
    idx = np.arange(28)[None, :]
    rows_any = binary.any(axis=2)
    cols_any = binary.any(axis=1)
    top = np.where(rows_any, idx, 28).min(axis=1)
    bottom = np.where(rows_any, idx, -1).max(axis=1)
    left = np.where(cols_any, idx, 28).min(axis=1)
    right = np.where(cols_any, idx, -1).max(axis=1)
    height = np.clip(bottom - top + 1, 1, 28)
    width = np.clip(right - left + 1, 1, 28)
    return top, bottom, left, right, height, width


def _bbox_band_densities(binary: np.ndarray, row_band: int, col_band: int) -> np.ndarray:
    """Ink density of the (row_band, col_band) cell of the normalised 3x3 box.

    The box is the image's own ink bounding box, split into three equal row
    bands and three equal column bands.  Band membership uses the integer
    formula ``floor(offset * 3 / extent)`` so the cell matches the cached
    `bb*` columns used in results/round2_frontier.md exactly.
    """
    n = binary.shape[0]
    top, bottom, left, right, height, width = _bbox_bounds(binary)
    height = height.astype(np.float64)
    width = width.astype(np.float64)

    r = np.arange(28)[None, :, None]
    c = np.arange(28)[None, None, :]
    rpos = np.clip(r - top[:, None, None], 0, 27)
    cpos = np.clip(c - left[:, None, None], 0, 27)
    rb = np.clip((rpos * 3) // height[:, None, None], 0, 2).astype(np.int64)
    cb = np.clip((cpos * 3) // width[:, None, None], 0, 2).astype(np.int64)

    in_rows = (r >= top[:, None, None]) & (r <= bottom[:, None, None])
    in_cols = (c >= left[:, None, None]) & (c <= right[:, None, None])
    row_sel = (rb == row_band)
    col_sel = (cb == col_band)

    counts = np.count_nonzero(row_sel & col_sel & binary, axis=(1, 2)).astype(np.float64)
    row_h = np.count_nonzero(row_sel & in_rows, axis=(1, 2)).astype(np.float64)
    col_w = np.count_nonzero(col_sel & in_cols, axis=(1, 2)).astype(np.float64)
    return counts / np.maximum(row_h * col_w, 1.0)


def _discretize(values: np.ndarray, n_states: int, thresholds=None):
    """State 0 = exactly zero; states 1..n-1 = training quantiles of positives."""
    if thresholds is None:
        positives = values[values > 0]
        thresholds = np.quantile(positives, np.linspace(0, 1, n_states)[1:-1])
    discrete = np.searchsorted(thresholds, values, side="right") + (values > 0)
    return discrete.astype(np.int64), thresholds


class BayesNetClassifier:
    """Small Bayesian Network Protocol v1: M6 DAG + 2 bbox-relative densities."""

    def __init__(self, alpha: float = 1.0):
        if not np.isscalar(alpha) or not np.isfinite(alpha) or alpha <= 0:
            raise ValueError("alpha must be a finite positive number")
        self.alpha = float(alpha)

    # ------------------------------------------------------------------
    # feature extraction
    # ------------------------------------------------------------------
    def _continuous_features(self, X_array: np.ndarray) -> dict:
        images = X_array.reshape(-1, 28, 28).astype(np.float64) / 255.0
        binary = X_array.reshape(-1, 28, 28) > BINARY_THRESHOLD

        densities = np.empty((images.shape[0], len(_ABS_CELLS)), dtype=np.float64)
        for j, (r0, r1, c0, c1) in enumerate(_ABS_CELLS):
            if j in _BBOX_CELLS:
                row_band, col_band = _BBOX_CELLS[j]
                densities[:, j] = _bbox_band_densities(binary, row_band, col_band)
            else:
                densities[:, j] = images[:, r0:r1, c0:c1].mean(axis=(1, 2))

        top, bottom, left, right, height, width = _bbox_bounds(binary)
        # The incumbent's aspect formula, kept byte-for-byte identical so this
        # module changes ONLY the two density slots.
        rows_any = binary.any(axis=2)
        cols_any = binary.any(axis=1)
        idx = np.arange(28)[None, :]
        first_row = np.where(rows_any, idx * rows_any, 0).max(axis=1)
        last_row = np.where(rows_any, (27 - idx) * rows_any, 0).max(axis=1)
        first_col = np.where(cols_any, idx * cols_any, 0).max(axis=1)
        last_col = np.where(cols_any, (27 - idx) * cols_any, 0).max(axis=1)
        inc_height = np.clip(last_row - first_row + 1, 1, 28)
        inc_width = np.clip(last_col - first_col + 1, 1, 28)
        aspect = inc_width / inc_height

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

        self._fit_thresholds(self._continuous_features(X_array))
        features = self._discretize_features(self._continuous_features(X_array))
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
    """Extract the 12 documented discrete features from flattened images."""
    model = BayesNetClassifier()
    X_array = _validate_X(X, allow_empty=True)
    if X_array.shape[0] == 0:
        return np.empty((0, len(FEATURE_NAMES)), dtype=np.int64)
    feats = model._continuous_features(X_array)
    n = feats["densities"].shape[0]
    features = np.empty((n, len(FEATURE_NAMES)), dtype=np.int64)
    for j in range(feats["densities"].shape[1]):
        features[:, j] = _discretize(feats["densities"][:, j], 5)[0]
    features[:, 8] = feats["euler"]
    for col, name, card in (
        (9, "vertical_runs", 5),
        (10, "aspect", 4),
        (11, "vertical_dominance", 4),
    ):
        features[:, col] = _discretize(feats[name], card)[0]
    return features
