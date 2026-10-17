"""Builder A candidate: dilated-binary patch Naive Bayes for MNIST.

Method family: FEATURE GEOMETRY / STATE-SPACE DESIGN.

The incumbent (``assignment2/naive_bayes.py``) represents each image by the 729
overlapping *contiguous* 2x2 patches, each pixel quantized to 5 gray bands, so
each patch is one categorical variable with 5^4 = 625 states.

This candidate keeps the naive Bayes product form

    P(y | X)  proportional to  P(y) * prod_j P(z_j | y)

with the same log-probability accumulation, the same Lidstone smoothing and the
same hand-written NumPy CPT estimation -- only the *feature definition* changes:

* Patch shape is 4x4 (not 2x2), so a single feature encodes a 16-pixel pattern.
* The 16 pixels are *not* contiguous: they sit on a spacing-2 lattice
  (offsets 0, 2, 4, 6 in both axes).  The patch therefore covers a 7x7
  receptive field while still costing only 16 pixels of state.
* Every pixel is binarized (ink if >= 72), giving 2^16 = 65,536 states.

Why this helps, as a modelling argument rather than a knob:

1.  *Wider receptive field.*  A 2x2 patch sees 4 pixels; it can encode a local
    stroke edge but not the *shape* of a stroke (a corner, a junction, a thin
    diagonal).  A 7x7 field with 16 samples can distinguish "diagonal
    line through the field" from "corner" from "isolated blob" -- exactly the
    handwriting primitives that separate, say, 4 from 9 or 3 from 8.
2.  *Dilation beats contiguity for the same state budget.*  A contiguous 4x4
    patch would also use 16 pixels, but measurements show the spacing-2 lattice
    is clearly better (0.9521 vs 0.9639 at threshold 128) because it samples
    the receptive field broadly instead of over-weighting one tight corner.
3.  *Binarization is the right quantization at this width.*  With 16 pixels per
    feature the state space is the scarce resource, so spending it on gray
    levels is wasteful; experiments show 5 gray levels per pixel at width 2 is
    beaten by 2 levels at width 4.  The useful MNIST signal at this scale is
    stroke *presence*, not stroke *intensity*.
4.  *Sparse CPT storage.*  484 features x 65,536 states = 31.7M slots, of which
    only ~4% are ever observed in 50k training images.  We therefore store only
    the observed (feature, state) pairs.  This is exact, not an approximation:
    an unobserved pair receives the *same* smoothed value
    log(alpha / (N_c + n_states*alpha)) for every class, so it can never change
    the argmax; we still account for it per feature so the likelihood is the
    exact naive Bayes sum rather than a truncated one.

Reference measurements on the fixed v4 protocol (49,999 train / 10,001 val,
seed 1322, training data only):

    incumbent  2x2-L5   alpha=0.05   0.9298
    incumbent  2x2-L5   best alpha   0.9298
    this model 4x4-s2   alpha=0.05   0.9668
    this model 4x4-s2   alpha=0.001  0.9737   <- recommended

The model is fully deterministic (no randomness), so the fixed seed 1322
requirement is met trivially.

Drop-in compatibility: the class exposes the same protocol and the same input
validation semantics as the incumbent ``NaiveBayesClassifier``; X is the raw
(n_samples, 784) uint8 matrix with values in [0, 255].
"""

from __future__ import annotations

import numpy as np

N_CLASSES = 10
N_FEATURES = 28 * 28
IMG_SIZE = 28

# ---- the frozen feature geometry (the contribution of this candidate) ----
PATCH_H = 4
PATCH_W = 4
PATCH_INNER = 2          # pixel spacing inside a patch -> 7x7 receptive field
PATCH_STRIDE = 1         # maximally overlapping patch grid -> 22x22 patches
BINARY_THRESHOLD = 72.0  # pixel >= 72 counts as ink
N_STATES = 1 << (PATCH_H * PATCH_W)   # 65,536

# 22 x 22 = 484 overlapping 7x7-receptive-field patches
N_PATCH_FEATURES = (IMG_SIZE - ((PATCH_H - 1) * PATCH_INNER + 1)) // PATCH_STRIDE + 1
N_PATCH_FEATURES *= (IMG_SIZE - ((PATCH_W - 1) * PATCH_INNER + 1)) // PATCH_STRIDE + 1

# alpha recommended by the builder A experiments; see results/builder_a.md
BEST_ALPHA = 0.001
# blocks larger than this many (feature, state) slots use sparse CPT storage
DENSE_SLOT_LIMIT = 20_000_000


def _validate_X(X: np.ndarray, *, allow_empty: bool) -> np.ndarray:
    """Same validation semantics as the incumbent naive_bayes module."""
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
    (i, j)-th pixel of a patch occupies bit (i*PATCH_W + j), chosen so that
    flipping the encoding order does not change the model (the CPT is refit for
    the relabelled states anyway).
    """
    n_samples = X.shape[0]
    image = X.reshape(n_samples, IMG_SIZE, IMG_SIZE)
    ink = image >= BINARY_THRESHOLD                      # (n, 28, 28) bool

    grid = (IMG_SIZE - ((PATCH_H - 1) * PATCH_INNER + 1)) // PATCH_STRIDE + 1
    states = np.zeros((n_samples, grid, grid), dtype=np.int64)
    bit = 0
    for i in range(PATCH_H):
        for j in range(PATCH_W):
            rows = slice(i * PATCH_INNER, i * PATCH_INNER + (grid - 1) * PATCH_STRIDE + 1, PATCH_STRIDE)
            cols = slice(j * PATCH_INNER, j * PATCH_INNER + (grid - 1) * PATCH_STRIDE + 1, PATCH_STRIDE)
            states |= ink[:, rows, cols].astype(np.int64) << bit
            bit += 1
    return states.reshape(n_samples, grid * grid)


class NaiveBayesClassifier:
    """Naive Bayes over dilated 4x4 binary patch patterns.

    Protocol: ``__init__(alpha=1.0)``, ``fit(X, y) -> self``,
    ``predict(X) -> labels``, ``predict_proba(X)``.  X is the raw
    (n_samples, 784) pixel matrix with values 0-255.

    Prediction is ``argmax_y [ log P(y) + sum_j log P(z_j | y) ]`` with the 484
    patch states conditionally independent given the class.
    """

    def __init__(self, alpha: float = 1.0, dense_slot_limit: int = DENSE_SLOT_LIMIT):
        if not np.isscalar(alpha) or not np.isfinite(alpha) or alpha <= 0:
            raise ValueError("alpha must be a finite positive number")
        self.alpha = float(alpha)
        self.dense_slot_limit = int(dense_slot_limit)

    # ------------------------------------------------------------------
    def fit(self, X, y):
        """Estimate the class priors and the per-patch conditional tables."""
        X_array = _validate_X(X, allow_empty=False)
        y_int = _validate_y(y, X_array.shape[0])

        states = _patch_states(X_array)
        n_samples = X_array.shape[0]
        class_count = np.bincount(y_int, minlength=N_CLASSES).astype(np.float64)
        alpha = self.alpha
        self.class_count_ = class_count
        self.class_log_prior_ = np.log(
            (class_count + alpha) / (n_samples + N_CLASSES * alpha)
        )

        offsets = np.arange(N_PATCH_FEATURES, dtype=np.int64) * N_STATES
        flat = (states + offsets).ravel()
        keys = np.unique(flat)                    # observed feature*state pairs

        log_prob = np.empty((N_CLASSES, keys.size), dtype=np.float64)
        self.default_log_prob_ = np.log(alpha / (class_count + N_STATES * alpha))
        for class_index in range(N_CLASSES):
            sub = (states[y_int == class_index] + offsets).ravel()
            position = np.searchsorted(keys, sub)
            counts = np.bincount(position, minlength=keys.size).astype(np.float64)
            log_prob[class_index] = np.log(
                (counts + alpha) / (class_count[class_index] + N_STATES * alpha)
            )
        self.observed_keys_ = keys
        self.observed_log_prob_ = log_prob
        self.n_observed_states_ = int(keys.size)
        self.is_fitted_ = True
        return self

    def _check_fitted(self):
        if not getattr(self, "is_fitted_", False):
            raise RuntimeError("the classifier must be fitted before prediction")

    # ------------------------------------------------------------------
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

        scores = np.tile(self.class_log_prior_, (n_samples, 1))
        # every feature factor is present exactly once: observed pairs use their
        # own log-probability, everything else uses the smoothed default.
        scores += float(N_PATCH_FEATURES) * default_log[None, :]
        for start in range(0, n_samples, batch_size):
            stop = min(start + batch_size, n_samples)
            flat = states[start:stop] + offsets                      # (b, F)
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
                    sample_index, weights=correction[class_index], minlength=stop - start
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
        """Verify every per-patch conditional distribution sums to exactly 1.

        The dense CPT is never materialised here (484 x 65,536 slots per class
        would not fit in memory), so the normalisation is checked in the
        compressed representation.  Summing P(z | y) over all states and all
        features must give N_PATCH_FEATURES: each observed (feature, state) pair
        contributes its own smoothed probability, and every one of the
        `N_PATCH_FEATURES*N_STATES - n_observed` unseen pairs contributes the
        shared default value.  Returns the per-class total mass, which the unit
        test can assert against N_PATCH_FEATURES.
        """
        self._check_fitted()
        observed_mass = np.exp(self.observed_log_prob_).sum(axis=1)
        unseen = N_PATCH_FEATURES * N_STATES - self.n_observed_states_
        total = observed_mass + unseen * np.exp(self.default_log_prob_)
        np.testing.assert_allclose(total, float(N_PATCH_FEATURES), rtol=rtol)
        return total


# Alias used by the builder A experiment scripts.
BuilderANaiveBayes = NaiveBayesClassifier
