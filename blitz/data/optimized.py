import sys

import numpy as np
import logging

try:
    from numba import jit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    # Dummy decorators if numba is missing, though this module shouldn't be used then
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    def prange(x):
        return range(x)

# Disable Numba disk cache in PyInstaller/frozen builds (__pycache__ may be read-only)
_JIT_CACHE = not getattr(sys, "frozen", False)

@jit(nopython=True, parallel=True, fastmath=True, cache=_JIT_CACHE)
def apply_pipeline_fused(
    image,
    do_sub, sub_ref, sub_amt,
    do_div, div_ref, div_amt
):
    """
    Apply subtraction and division in a single pass.

    image: (T, H, W, C) float32 array (modified in-place)
    sub_ref: (T_sub, H, W, C) float32 array or dummy
    div_ref: (T_div, H, W, C) float32 array or dummy
    sub_amt: float
    div_amt: float
    """
    # Parallel loop over all pixels
    # We can parallelize over T and H

    T, H, W, C = image.shape

    # Broadcasting check handled by indexing with modulo or 0
    # But Numba requires explicit shapes usually.
    # We assume inputs are correctly shaped (T or 1).

    # Pre-calculate strides/checks to avoid branching inside inner loop if possible?
    # Numba handles basic branching well.

    sub_T = sub_ref.shape[0]
    div_T = div_ref.shape[0]

    for t in range(T):
        # Determine indices for refs
        if sub_T > 1:
            ts = int(t)
        else:
            ts = 0

        if div_T > 1:
            td = int(t)
        else:
            td = 0

        for h in prange(H):
            for w in range(W):
                for c in range(C):
                    val = image[t, h, w, c]

                    if do_sub:
                        s = sub_ref[ts, h, w, c]
                        val = val - (sub_amt * s)

                    if do_div:
                        d = div_ref[td, h, w, c]
                        # denom = amount * ref + (1 - amount)
                        denom = (div_amt * d) + (1.0 - div_amt)

                        if denom == 0.0:
                            val = 0.0
                        else:
                            val = val / denom
                            # Check for NaN/Inf
                            if np.isnan(val) or np.isinf(val):
                                val = 0.0

                    image[t, h, w, c] = val

    return image


@jit(nopython=True, parallel=True, fastmath=True, cache=_JIT_CACHE)
def sliding_mean_numba(images, window, lag):
    """
    Compute sliding mean over time dimension T.

    images: (T, H, W, C) float32
    window: int
    lag: int

    Returns: (N, H, W, C) where N = T - (lag + window)
    """
    T, H, W, C = images.shape
    N = T - (lag + window)

    # Allocate result
    result = np.empty((N, H, W, C), dtype=np.float32)

    # Parallelize over spatial dimensions
    # Using prange on H is usually sufficient parallelism
    for h in prange(H):
        for w in range(W):
            for c in range(C):
                # Sliding window sum
                # Initialize first window sum
                # Window range for output t=0 corresponds to input indices [lag : lag+window]
                # Actually, standard definition:
                # mean[t] = mean(input[t+lag : t+lag+window])

                # Compute sum for first window
                # Corresponds to input indices [lag+1 : lag+window] (inclusive-exclusive?)
                # Legacy implementation: cs[lag+window] - cs[lag] -> sums elements lag+1 to lag+window (1-based count?)
                # Actually indices: lag+1, lag+2, ..., lag+window.
                current_sum = 0.0
                start_idx = lag + 1
                for k in range(window):
                    current_sum += images[start_idx + k, h, w, c]

                result[0, h, w, c] = current_sum / window

                # Sliding update
                for n in range(1, N):
                    # Remove element leaving the window
                    leaving_idx = lag + n
                    # Add element entering the window
                    entering_idx = lag + window + n

                    val_leaving = images[leaving_idx, h, w, c]
                    val_entering = images[entering_idx, h, w, c]

                    current_sum -= val_leaving
                    current_sum += val_entering

                    result[n, h, w, c] = current_sum / window

    return result


def _reduce_axis0_numba(x: np.ndarray, op: str) -> np.ndarray | None:
    """Numba-accelerated reduce over axis 0. Returns (1, H, W, C). Expects 4D input."""
    if not HAS_NUMBA:
        return None
    if x.ndim != 4:
        return None
    T, H, W, C = x.shape
    result = np.empty((1, H, W, C), dtype=np.float32)
    x_f = x.astype(np.float32) if x.dtype != np.float32 else x
    if op == "mean":
        _reduce_mean_impl(result, x_f, T, H, W, C)
    elif op == "max":
        _reduce_max_impl(result, x_f, T, H, W, C)
    elif op == "min":
        _reduce_min_impl(result, x_f, T, H, W, C)
    elif op == "std":
        _reduce_std_impl(result, x_f, T, H, W, C)
    else:
        return None
    return result


@jit(nopython=True, parallel=True, fastmath=True, cache=_JIT_CACHE)
def _reduce_mean_impl(result, x, T, H, W, C):
    for h in prange(H):
        for w in range(W):
            for c in range(C):
                s = 0.0
                for t in range(T):
                    s += x[t, h, w, c]
                result[0, h, w, c] = s / T


@jit(nopython=True, parallel=True, fastmath=True, cache=_JIT_CACHE)
def _reduce_max_impl(result, x, T, H, W, C):
    for h in prange(H):
        for w in range(W):
            for c in range(C):
                m = x[0, h, w, c]
                for t in range(1, T):
                    v = x[t, h, w, c]
                    if v > m:
                        m = v
                result[0, h, w, c] = m


@jit(nopython=True, parallel=True, fastmath=True, cache=_JIT_CACHE)
def _reduce_min_impl(result, x, T, H, W, C):
    for h in prange(H):
        for w in range(W):
            for c in range(C):
                m = x[0, h, w, c]
                for t in range(1, T):
                    v = x[t, h, w, c]
                    if v < m:
                        m = v
                result[0, h, w, c] = m


@jit(nopython=True, parallel=True, fastmath=True, cache=_JIT_CACHE)
def _reduce_std_impl(result, x, T, H, W, C):
    # Welford's online algorithm for single-pass mean + variance
    for h in prange(H):
        for w in range(W):
            for c in range(C):
                n = 0
                mean = 0.0
                m2 = 0.0
                for t in range(T):
                    v = x[t, h, w, c]
                    n += 1
                    delta = v - mean
                    mean += delta / n
                    m2 += delta * (v - mean)
                result[0, h, w, c] = np.sqrt(m2 / n) if n > 1 else 0.0
