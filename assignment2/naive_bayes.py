"""Dilated-patch binary Naive Bayes classifier for MNIST.

The probability model is a strict generative naive Bayes built on the same
factorisation the assignment states for the raw pixels,

    P(y | X)  proportional to  P(y) * prod_j P(z_j | y),
    y_hat = argmax_y [ log P(y) + sum_j log P(z_j | y) ],

except that the 784 pixel variables are redefined as 484 *patch* variables z_j
("feature redefinition", exactly as the previous 2x2-patch version already did).
Each patch is a 4x4 group of pixels that are *dilated* (sampled every second
pixel), so a patch covers a 7x7 receptive field while still costing only 16
pixels of state:

    z_j  =  bit pattern of the 16 binary pixels  in  {0, ..., 2^16 - 1}.

Pixels are binarised with a fixed threshold (pixel >= 72 counts as ink); the
16 bits of a patch are combined into one categorical state out of 65536.  The
patch variables stay conditionally independent given the class, so this is
still naive Bayes -- only the feature definition changed.

Why this beats single pixels and small patches (see experiments_v4/):

  * A 2x2 patch sees 4 pixels; it can encode a local edge but not the *shape* of
    a stroke (a corner, a junction, a thin diagonal).  A 7x7 receptive field
    sampled at 16 points can tell "diagonal through the field" from "corner"
    from "isolated blob" -- the handwriting primitives that separate 4 from 9
    or 3 from 8.
  * Widening the receptive field is a better use of the state budget than
    spending it on gray levels: 16 binary pixels (65536 states) clearly beat
    5 gray levels on 4 pixels (625 states), and also beat 3 gray levels at width
    4.  At this scale the useful MNIST signal is stroke *presence*, not
    intensity.

Everything is implemented with NumPy only; no ready-made classifier is used.

Smoothing.  The CPT for a 65536-state variable is extremely sparse (only ~4% of
(feature, state) slots are ever observed in 50k images), so the additive
smoothing coefficient is parametrised as the *total* Dirichlet mass ``alpha``
per patch variable, with a per-state pseudo-count of ``alpha / 65536``.  Scaling
the pseudo-count to the state space keeps the effective regularisation of a
65k-state variable comparable to that of a small one; the protocol default
``alpha = 1.0`` then lands on a broad plateau (validation 97.4%-97.7% for
alpha in [0.01, 10]) instead of the knife-edge a per-state coefficient would
create.  This follows the usual interpretation of ``alpha`` as the strength of
the Dirichlet prior rather than a raw per-cell count.
"""

from __future__ import annotations

import numpy as np


N_CLASSES = 10
N_PIXELS = 28 * 28
IMAGE_SIZE = 28

# ---- feature geometry: dilated 4x4 binary patches ----
PATCH_ROWS = 4
PATCH_COLS = 4
DILATION = 2                     # pixel spacing inside a patch
PATCH_STRIDE = 1                 # maximally overlapping patch grid
BINARY_THRESHOLD = 72.0          # pixel >= threshold counts as ink
N_STATES = 1 << (PATCH_ROWS * PATCH_COLS)   # 65536
# grid size along one axis: 28 - (3*2 + 1) + 1 = 22
GRID = IMAGE_SIZE - ((PATCH_ROWS - 1) * DILATION + 1) + 1
N_PATCH_FEATURES = GRID * GRID   # 484 overlapping patches


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
    """Encode every dilated 4x4 binary patch as one state in [0, 2^16).

    X: (n, 784) raw pixels -> (n, 484) int64 states.  Bit (i * PATCH_COLS + j)
    holds the ink flag of the (i, j)-th dilated pixel of the patch.
    """
    n_samples = X.shape[0]
    ink = X.reshape(n_samples, IMAGE_SIZE, IMAGE_SIZE) >= BINARY_THRESHOLD
    states = np.zeros((n_samples, GRID, GRID), dtype=np.int64)
    bit = 0
    for i in range(PATCH_ROWS):
        for j in range(PATCH_COLS):
            rows = slice(i * DILATION, i * DILATION + GRID)
            cols = slice(j * DILATION, j * DILATION + GRID)
            states |= ink[:, rows, cols].astype(np.int64) << bit
            bit += 1
    return states.reshape(n_samples, GRID * GRID)


class NaiveBayesClassifier:
    """Naive Bayes Protocol v1 over dilated 4x4 binary patch patterns.

    Protocol: ``__init__(alpha=1.0)``, ``fit(X, y) -> self``,
    ``predict(X) -> labels``, ``predict_proba(X)``.  X is the raw
    (n_samples, 784) pixel matrix with values 0-255.
    """

    def __init__(self, alpha: float = 1.0):
        if not np.isscalar(alpha) or not np.isfinite(alpha) or alpha <= 0:
            raise ValueError("alpha must be a finite positive number")
        self.alpha = float(alpha)

    # ------------------------------------------------------------------
    def fit(self, X, y):
        """Estimate the class priors and the per-patch conditional tables."""
        X_array = _validate_X(X, allow_empty=False)
        y_int = _validate_y(y, X_array.shape[0])

        states = _patch_states(X_array)
        n_samples = X_array.shape[0]
        class_count = np.bincount(y_int, minlength=N_CLASSES).astype(np.float64)

        # total Dirichlet mass alpha split evenly over the state space
        per_state = self.alpha / N_STATES
        self.class_count_ = class_count
        self.class_log_prior_ = np.log(
            (class_count + per_state) / (n_samples + N_CLASSES * per_state)
        )
        self.default_log_prob_ = np.log(
            per_state / (class_count + self.alpha)
        )

        # exact sparse CPT: only observed (feature, state) pairs are stored;
        # every unobserved pair receives the same smoothed default value for all
        # classes, so it can never change the argmax.
        offsets = np.arange(N_PATCH_FEATURES, dtype=np.int64) * N_STATES
        keys = np.unique((states + offsets).ravel())
        log_prob = np.empty((N_CLASSES, keys.size), dtype=np.float64)
        for class_index in range(N_CLASSES):
            sub = (states[y_int == class_index] + offsets).ravel()
            position = np.searchsorted(keys, sub)
            counts = np.bincount(position, minlength=keys.size).astype(np.float64)
            log_prob[class_index] = np.log(
                (counts + per_state) / (class_count[class_index] + self.alpha)
            )
        self.observed_keys_ = keys
        self.observed_log_prob_ = log_prob
        self.n_observed_states_ = int(keys.size)
        self.is_fitted_ = True
        return self

    def _check_fitted(self) -> None:
        if not getattr(self, "is_fitted_", False):
            raise RuntimeError("the classifier must be fitted before prediction")

    def _joint_log_likelihood(self, X, *, batch_size: int = 1024) -> np.ndarray:
        self._check_fitted()
        X_array = _validate_X(X, allow_empty=True)
        if X_array.shape[0] == 0:
            return np.empty((0, N_CLASSES), dtype=np.float64)

        states = _patch_states(X_array)
        n_samples = states.shape[0]
        offsets = (np.arange(N_PATCH_FEATURES, dtype=np.int64) * N_STATES)[None, :]
        keys = self.observed_keys_
        default_log = self.default_log_prob_
        log_prob = self.observed_log_prob_

        # every patch factor contributes exactly once: start from the smoothed
        # default for all of them, then correct the observed pairs.
        scores = np.tile(self.class_log_prior_, (n_samples, 1))
        scores += float(N_PATCH_FEATURES) * default_log[None, :]
        for start in range(0, n_samples, batch_size):
            stop = min(start + batch_size, n_samples)
            flat = states[start:stop] + offsets
            position = np.searchsorted(keys, flat)
            np.clip(position, 0, keys.size - 1, out=position)
            found = keys[position] == flat
            if not found.any():
                continue
            sample_index, _ = np.nonzero(found)
            key_index = position[sample_index, np.nonzero(found)[1]]
            correction = log_prob[:, key_index] - default_log[:, None]
            for class_index in range(N_CLASSES):
                scores[start:stop, class_index] += np.bincount(
                    sample_index,
                    weights=correction[class_index],
                    minlength=stop - start,
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
        """Verify that every per-patch conditional distribution sums to 1.

        The dense CPT is never materialised (10 x 484 x 65536 doubles would be
        ~2.5 GB), so normalisation is verified in the compressed form: the
        observed pairs contribute their own smoothed probability and every one
        of the ``N_PATCH_FEATURES * N_STATES - n_observed`` unseen pairs
        contributes the shared default.  The total must equal the number of
        patch features for each class.
        """
        self._check_fitted()
        observed_mass = np.exp(self.observed_log_prob_).sum(axis=1)
        unseen = N_PATCH_FEATURES * N_STATES - self.n_observed_states_
        total = observed_mass + unseen * np.exp(self.default_log_prob_)
        np.testing.assert_allclose(total, float(N_PATCH_FEATURES), rtol=rtol)
        return total
