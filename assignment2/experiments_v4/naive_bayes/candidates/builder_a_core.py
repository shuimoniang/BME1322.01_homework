"""Builder A core: flexible patch-geometry / state-space Naive Bayes for MNIST.

The family explored here is **feature geometry and state-space design**: which
local patterns the categorical features encode, how many gray levels each pixel
gets, where the level boundaries sit, how the joint patch state space is built
and whether rare states are merged.  The probability model is *always* plain
naive Bayes:

    P(y | X)  proportional to  P(y) * prod_j P(z_j | y)

with every z_j conditionally independent of the others given y, log-probability
accumulation, Lidstone smoothing and hand-written NumPy CPT estimation.  No
ready-made classifier is used.

A "view" is one family of identically-shaped patches; each view contributes one
or more categorical feature blocks (a block = a group of features that share a
state cardinality).  Views are concatenated, which keeps the NB product form.

View keys
---------
ph, pw        : patch height / width in pixels (default 2 x 2)
inner         : pixel spacing *inside* a patch (1 = contiguous, 2 = dilated)
stride        : patch spacing across the image (1 = maximal overlap)
levels        : gray-level bands per pixel inside the patch
edges         : 'uniform'  -> incumbent edges linspace(0,255,levels+1)[1:-1]
                'ink'      -> band 0 is exactly {0}; ink split into levels-1 bands
                'posq'     -> per-pixel-position quantile thresholds (train only)
                np.ndarray -> explicit thresholds
factorize     : if True, emit TWO blocks instead of one joint block:
                  * binary pattern  (2 ** (ph*pw) states)
                  * patch mean intensity quantized to `mean_levels` bands
                This is still naive Bayes (each block is its own independent
                feature set) but it *adds* an independence assumption
                (pattern independent of ink level given y).
mean_levels   : band count for the mean-intensity block of a factorized view
binary_edges  : threshold(s) used for the binary pattern block (default [0.5])

Everything here is deterministic (no randomness at all), so seed 1322 is
satisfied trivially.
"""

from __future__ import annotations

import numpy as np

N_CLASSES = 10
N_FEATURES = 28 * 28
IMG_SIZE = 28

__all__ = ["PatchNaiveBayes", "N_CLASSES", "N_FEATURES"]


# ----------------------------------------------------------------------
# input validation -- identical semantics to naive_bayes.NaiveBayesClassifier
# ----------------------------------------------------------------------
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


# ----------------------------------------------------------------------
# quantization
# ----------------------------------------------------------------------
def uniform_edges(levels: int) -> np.ndarray:
    """The incumbent's band boundaries: exactly [51,102,153,204] for levels=5."""
    if levels < 2:
        raise ValueError("levels must be >= 2")
    return np.linspace(0.0, 255.0, levels + 1)[1:-1]


def ink_edges(levels: int) -> np.ndarray:
    """band 0 == {0} (background), remaining levels-1 bands split the ink 1..255."""
    if levels < 2:
        raise ValueError("levels must be >= 2")
    if levels == 2:
        return np.array([0.5])
    inner = np.linspace(1.0, 255.0, levels)[1:-1]
    return np.concatenate([[0.5], inner])


def position_quantile_edges(X_train: np.ndarray, levels: int) -> np.ndarray:
    """Per-pixel-position quantile thresholds, fit on TRAINING pixels only.

    Returns (levels-1, 28, 28) so the result broadcasts against (n, 28, 28).
    Because MNIST background dominates most border positions, the low quantiles
    collapse to 0 there; that is handled naturally by the >= comparison (a band
    simply becomes unreachable), keeping the encoding well defined.
    """
    if levels < 2:
        raise ValueError("levels must be >= 2")
    percentiles = np.linspace(0.0, 100.0, levels + 1)[1:-1]
    edges = np.percentile(X_train, percentiles, axis=0)
    return edges.reshape(levels - 1, IMG_SIZE, IMG_SIZE).astype(np.float64)


def ink_position_quantile_edges(X_train: np.ndarray, levels: int) -> np.ndarray:
    """Per-position thresholds placed on the *ink* distribution of that position.

    For levels=2 this is the per-position median of the non-zero (ink) pixels,
    i.e. an adaptive binarization threshold: positions that are usually
    background get a threshold of 0 and behave like a plain 'ink present' test,
    while stroke-dense positions get a threshold matched to their own
    intensity distribution.  Fit on TRAINING pixels only.
    Returns (levels-1, 28, 28).
    """
    if levels < 2:
        raise ValueError("levels must be >= 2")
    percentiles = np.linspace(0.0, 100.0, levels + 1)[1:-1]
    ink = np.where(X_train > 0, X_train, np.nan)
    with np.errstate(all="ignore"):
        edges = np.nanpercentile(ink, percentiles, axis=0)
    # positions that are *always* background get an unreachable edge so that
    # zero pixels stay in band 0 rather than being pushed into a high band
    edges = np.where(np.isnan(edges), 256.0, edges)
    return edges.reshape(levels - 1, IMG_SIZE, IMG_SIZE).astype(np.float64)


def _bands_from_edges(image_2d: np.ndarray, edges_2d, levels: int) -> np.ndarray:
    """band(x) = #{edge <= x}; edges_2d broadcastable to image_2d."""
    bands = np.zeros(image_2d.shape, dtype=np.int64)
    edge_array = np.asarray(edges_2d, dtype=np.float64)
    if edge_array.ndim == 1:
        for edge in edge_array:
            bands += image_2d >= edge
    else:
        for k in range(edge_array.shape[0]):
            bands += image_2d >= edge_array[k]
    np.clip(bands, 0, levels - 1, out=bands)
    return bands


# ----------------------------------------------------------------------
# state encoding
# ----------------------------------------------------------------------
def _grid_shape(ph, pw, inner, stride):
    inner_h, inner_w = _spacings(inner)
    extent_h = (ph - 1) * inner_h + 1
    extent_w = (pw - 1) * inner_w + 1
    height = (IMG_SIZE - extent_h) // stride + 1
    width = (IMG_SIZE - extent_w) // stride + 1
    return height, width


def _spacings(inner):
    """`inner` may be an int or an (inner_h, inner_w) pair."""
    if isinstance(inner, (tuple, list)):
        return int(inner[0]), int(inner[1])
    return int(inner), int(inner)


def _patch_slice(bands, i, j, ph, pw, inner, stride, height, width):
    inner_h, inner_w = _spacings(inner)
    rows = slice(i * inner_h, i * inner_h + (height - 1) * stride + 1, stride)
    cols = slice(j * inner_w, j * inner_w + (width - 1) * stride + 1, stride)
    return bands[:, rows, cols]


def encode_patch_states(bands: np.ndarray, ph, pw, inner, stride, base: int) -> np.ndarray:
    """Joint-encode every patch as one categorical state (base**n_pixels states).

    Row-major, first pixel most significant -- identical to the incumbent's
    ``(b00*L + b01)*L + b10)*L + b11`` encoding for a 2x2 patch, so the default
    view is a byte-for-byte equivalent model.
    """
    height, width = _grid_shape(ph, pw, inner, stride)
    n_pixels = ph * pw
    out = np.zeros((bands.shape[0], height, width), dtype=np.int64)
    order = 0
    for i in range(ph):
        for j in range(pw):
            out += (base ** (n_pixels - 1 - order)) * _patch_slice(
                bands, i, j, ph, pw, inner, stride, height, width
            )
            order += 1
    return out.reshape(bands.shape[0], height * width)


def encode_binary_states(bands: np.ndarray, ph, pw, inner, stride) -> np.ndarray:
    """Bitmask of 'ink present' inside each patch -> 2**(ph*pw) states."""
    height, width = _grid_shape(ph, pw, inner, stride)
    out = np.zeros((bands.shape[0], height, width), dtype=np.int64)
    order = 0
    for i in range(ph):
        for j in range(pw):
            sub = _patch_slice(bands, i, j, ph, pw, inner, stride, height, width)
            out += (1 << order) * sub
            order += 1
    return out.reshape(bands.shape[0], height * width)


def encode_mean_states(bands: np.ndarray, ph, pw, inner, stride, mean_levels: int) -> np.ndarray:
    """Mean gray level inside each patch, quantized to `mean_levels` uniform bands."""
    height, width = _grid_shape(ph, pw, inner, stride)
    total = np.zeros((bands.shape[0], height, width), dtype=np.float64)
    for i in range(ph):
        for j in range(pw):
            total += _patch_slice(bands, i, j, ph, pw, inner, stride, height, width)
    mean = total / float(ph * pw)
    edges = uniform_edges(mean_levels)
    out = np.zeros(mean.shape, dtype=np.int64)
    for edge in edges:
        out += mean >= edge
    return out.reshape(bands.shape[0], height * width)


# ----------------------------------------------------------------------
# the classifier
# ----------------------------------------------------------------------
DEFAULT_VIEW = {"ph": 2, "pw": 2, "levels": 5, "edges": "uniform", "inner": 1, "stride": 1}


class PatchNaiveBayes:
    """Naive Bayes over hand-designed local patch features.

    Protocol identical to ``naive_bayes.NaiveBayesClassifier``:
    ``__init__(alpha=1.0)``, ``fit(X, y) -> self``, ``predict(X)``,
    ``predict_proba(X)``; X is the raw (n_samples, 784) pixel matrix, 0-255.
    """

    def __init__(self, alpha: float = 1.0, views=None, merge_min_count: int = 0,
                 dense_slot_limit: int = 20_000_000):
        if not np.isscalar(alpha) or not np.isfinite(alpha) or alpha <= 0:
            raise ValueError("alpha must be a finite positive number")
        self.alpha = float(alpha)
        self.views = [dict(DEFAULT_VIEW, **v) for v in (views or [DEFAULT_VIEW])]
        self.merge_min_count = int(merge_min_count)
        self.dense_slot_limit = int(dense_slot_limit)

    # ---------------- feature construction ----------------
    def _view_blocks(self, X: np.ndarray, view: dict, *, fitting: bool):
        """Return [(states (n, F), nstates, remap_table_or_None), ...] for one view.

        When `fitting` is False the quantization thresholds and merge tables
        stored by ``fit`` are reused, so train and predict are guaranteed to use
        exactly the same feature definition.
        """
        images = X.reshape(-1, IMG_SIZE, IMG_SIZE).astype(np.float64)
        key = view["key"]
        blocks = []
        if view.get("factorize", False):
            if fitting:
                binary_edges = np.asarray(view.get("binary_edges", [0.5]), dtype=np.float64)
                self._store(f"bin_edges_{key}", binary_edges)
            else:
                binary_edges = self._load(f"bin_edges_{key}")
            binary_bands = np.zeros(images.shape, dtype=np.int64)
            for edge in binary_edges:
                binary_bands += images >= edge
            pattern = encode_binary_states(binary_bands, view["ph"], view["pw"], view["inner"], view["stride"])
            blocks.append((pattern, 1 << (view["ph"] * view["pw"])))
            mean_states = encode_mean_states(
                images, view["ph"], view["pw"], view["inner"], view["stride"],
                view.get("mean_levels", 8),
            )
            blocks.append((mean_states, view.get("mean_levels", 8)))
            return blocks

        levels = view["levels"]
        edges_spec = view["edges"]
        if edges_spec == "uniform":
            edges = uniform_edges(levels)
        elif edges_spec == "ink":
            edges = ink_edges(levels)
        elif edges_spec == "posq":
            if fitting:
                edges = position_quantile_edges(X, levels)
                self._store(f"edges_{key}", edges)
            else:
                edges = self._load(f"edges_{key}")
        elif edges_spec == "inkq":
            if fitting:
                edges = ink_position_quantile_edges(X, levels)
                self._store(f"edges_{key}", edges)
            else:
                edges = self._load(f"edges_{key}")
        else:
            edges = np.asarray(edges_spec, dtype=np.float64)

        bands = _bands_from_edges(images, edges, levels)
        states = encode_patch_states(bands, view["ph"], view["pw"], view["inner"], view["stride"], base=levels)
        blocks.append((states, levels ** (view["ph"] * view["pw"])))
        return blocks

    def _store(self, name, value):
        if not hasattr(self, "_fit_cache"):
            self._fit_cache = {}
        self._fit_cache[name] = value

    def _load(self, name):
        return self._fit_cache[name]

    # ---------------- merging of rare states ----------------
    @staticmethod
    def _build_merge_table(states, nstates, min_count):
        """Merge every (feature, state) whose training count < min_count into a new state."""
        n_features = states.shape[1]
        offsets = np.arange(n_features, dtype=np.int64)[None, :] * nstates
        totals = np.bincount((states + offsets).ravel(), minlength=n_features * nstates)
        totals = totals.reshape(n_features, nstates)
        table = np.tile(np.arange(nstates, dtype=np.int64), (n_features, 1))
        rare = totals < min_count
        if rare.any():
            table[rare] = nstates
        return table, (nstates + 1 if rare.any() else nstates)

    @staticmethod
    def _apply_merge(states, table):
        if table is None:
            return states
        out = np.empty_like(states)
        for column in range(states.shape[1]):
            out[:, column] = table[column][states[:, column]]
        return out

    # ---------------- fitting ----------------
    def fit(self, X, y):
        X_array = _validate_X(X, allow_empty=False)
        y_int = _validate_y(y, X_array.shape[0])
        self._fit_cache = {}

        class_count = np.bincount(y_int, minlength=N_CLASSES).astype(np.float64)
        self.class_count_ = class_count
        self.class_log_prior_ = np.log(
            (class_count + self.alpha) / (X_array.shape[0] + N_CLASSES * self.alpha)
        )

        self.blocks_ = []
        self.feature_log_prob_ = []
        self.merge_tables_ = []
        for index, view in enumerate(self.views):
            view = dict(view)
            view.setdefault("key", f"v{index}")
            for states, nstates in self._view_blocks(X_array, view, fitting=True):
                table, nstates_eff = None, nstates
                if self.merge_min_count > 0:
                    table, nstates_eff = self._build_merge_table(states, nstates, self.merge_min_count)
                merged = self._apply_merge(states, table)
                self.blocks_.append((merged.shape[1], nstates_eff))
                self.merge_tables_.append(table)
                dense_slots = merged.shape[1] * nstates_eff
                if dense_slots <= self.dense_slot_limit:
                    self.feature_log_prob_.append(
                        self._fit_block(merged, y_int, class_count, nstates_eff)
                    )
                else:
                    self.feature_log_prob_.append(
                        self._fit_block_sparse(merged, y_int, class_count, nstates_eff)
                    )
        self._active_views_ = self.views
        self.is_fitted_ = True
        return self

    def _fit_block(self, states, y_int, class_count, nstates):
        n_features = states.shape[1]
        offsets = np.arange(n_features, dtype=np.int64) * nstates
        log_prob = np.empty((N_CLASSES, n_features, nstates), dtype=np.float64)
        for class_index in range(N_CLASSES):
            sub = states[y_int == class_index]
            counts = np.bincount(
                (sub + offsets).ravel(), minlength=n_features * nstates
            ).reshape(n_features, nstates)
            log_prob[class_index] = np.log(
                (counts + self.alpha) / (class_count[class_index] + nstates * self.alpha)
            )
        return log_prob

    def _fit_block_sparse(self, states, y_int, class_count, nstates):
        """Same CPT math as `_fit_block`, storing only observed (feature, state) pairs.

        For an unobserved pair the smoothed probability is
        alpha / (N_c + nstates*alpha) for EVERY class, i.e. a constant that
        shifts all class scores equally and can never change the argmax.  So
        omitting unobserved pairs from the likelihood sum is exact, and state
        spaces too large to materialise densely (e.g. 2**25) become tractable.
        """
        n_features = states.shape[1]
        offsets = np.arange(n_features, dtype=np.int64) * nstates
        flat = (states + offsets).ravel()
        keys = np.unique(flat)                      # observed feature*state pairs

        log_prob = np.empty((N_CLASSES, keys.size), dtype=np.float64)
        for class_index in range(N_CLASSES):
            sub = (states[y_int == class_index] + offsets).ravel()
            position = np.searchsorted(keys, sub)
            counts = np.bincount(position, minlength=keys.size).astype(np.float64)
            log_prob[class_index] = np.log(
                (counts + self.alpha) / (class_count[class_index] + nstates * self.alpha)
            )
        return {
            "sparse": True,
            "keys": keys,
            "log_prob": log_prob,
            "default_log": np.log(self.alpha / (class_count + nstates * self.alpha)),
            "n_features": n_features,
            "nstates": nstates,
            "n_observed": int(keys.size),
        }

    @staticmethod
    def _block_predict_sparse(block, merged):
        """Exact log-likelihood sum for a sparse block.

        Every one of the `n_features` factors contributes either its observed
        value log((count+alpha)/(N_c+nstates*alpha)) or, if that (feature,state)
        pair was never seen in training, the constant
        log(alpha/(N_c+nstates*alpha)).  So we start from
        n_features * default_log and then correct the observed pairs by
        (observed_log - default_log).  The per-feature constant matters: adding
        it once instead of n_features times would silently change the effective
        smoothing and break the equivalence with the dense CPT.
        """
        n_features = block["n_features"]
        nstates = block["nstates"]
        keys = block["keys"]
        default_log = block["default_log"]
        offsets = (np.arange(n_features, dtype=np.int64) * nstates)[None, :]
        flat = merged + offsets                                       # (n, F)
        position = np.searchsorted(keys, flat)
        np.clip(position, 0, keys.size - 1, out=position)
        found = keys[position] == flat
        out = np.tile(n_features * default_log, (merged.shape[0], 1))
        if found.any():
            sample_index, feature_index = np.nonzero(found)
            key_index = position[sample_index, feature_index]
            correction = block["log_prob"][:, key_index] - default_log[:, None]
            for class_index in range(N_CLASSES):
                out[:, class_index] += np.bincount(
                    sample_index, weights=correction[class_index],
                    minlength=merged.shape[0],
                )
        return out

    def _check_fitted(self):
        if not getattr(self, "is_fitted_", False):
            raise RuntimeError("the classifier must be fitted before prediction")

    # ---------------- prediction ----------------
    def _joint_log_likelihood(self, X, *, batch_size: int = 512) -> np.ndarray:
        self._check_fitted()
        X_array = _validate_X(X, allow_empty=True)
        if X_array.shape[0] == 0:
            return np.empty((0, N_CLASSES), dtype=np.float64)

        scores = np.tile(self.class_log_prior_, (X_array.shape[0], 1))
        order = 0
        for index, view in enumerate(self._active_views_):
            view = dict(view)
            view.setdefault("key", f"v{index}")
            for states, _ in self._view_blocks(X_array, view, fitting=False):
                table = self.merge_tables_[order]
                merged = self._apply_merge(states, table)
                n_features, nstates = self.blocks_[order]
                stored = self.feature_log_prob_[order]
                if isinstance(stored, dict):
                    scores += self._block_predict_sparse(stored, merged)
                    order += 1
                    continue
                log_prob = stored.reshape(N_CLASSES, n_features * nstates)
                offsets = (np.arange(n_features, dtype=np.int64) * nstates)[None, :]
                n_samples = merged.shape[0]
                for start in range(0, n_samples, batch_size):
                    stop = min(start + batch_size, n_samples)
                    index_matrix = merged[start:stop] + offsets
                    gathered = log_prob[:, index_matrix]          # (C, b, F)
                    scores[start:stop] += gathered.sum(axis=2).T
                order += 1
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


def make_factory(views, merge_min_count=0):
    """Convenience: a zero-argument factory so cross_seed() can refit the config."""
    def factory(alpha=1.0):
        return PatchNaiveBayes(alpha=alpha, views=views, merge_min_count=merge_min_count)
    return factory


def config_key(views, merge_min_count=0):
    parts = []
    for view in views:
        bits = [f"{view.get('ph',2)}x{view.get('pw',2)}", f"L{view.get('levels','bin')}",
                str(view.get("edges", "uniform")), f"in{view.get('inner',1)}",
                f"st{view.get('stride',1)}"]
        if view.get("factorize"):
            bits.append(f"fac{view.get('mean_levels',8)}")
        parts.append("-".join(bits))
    if merge_min_count:
        parts.append(f"merge{merge_min_count}")
    return "|".join(parts)
