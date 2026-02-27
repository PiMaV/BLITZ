import cv2
import numpy as np
# import numba
from . import ops, optimized


def _to_8bit_fixed_scale(arr: np.ndarray) -> np.ndarray:
    """Convert to uint8 using dtype range only (no per-image normalization). Preserves relative brightness."""
    if arr.dtype == np.uint8:
        return np.ascontiguousarray(arr)
    if arr.dtype == np.uint16:
        out = (arr.astype(np.float32) / 65535.0 * 255.0).clip(0, 255)
        return out.astype(np.uint8)
    if arr.dtype in (np.float32, np.float64):
        # Assume 0..1; values outside get clipped
        out = (arr.astype(np.float64) * 255.0).clip(0, 255)
        return np.nan_to_num(out, nan=0.0, posinf=255, neginf=0).astype(np.uint8)
    # Fallback: treat as 0..max(dtype)
    mx = np.iinfo(arr.dtype).max if np.issubdtype(arr.dtype, np.integer) else 1.0
    out = (arr.astype(np.float64) / mx * 255.0).clip(0, 255)
    return out.astype(np.uint8)


def _stretch_to_full_range(
    arr: np.ndarray,
    convert_to_8_bit: bool,
    original_dtype: np.dtype,
) -> np.ndarray:
    """Per-image min-max stretch to full range of target dtype. Works for 8, 16 bit and float."""
    with np.errstate(invalid="ignore"):
        mn, mx = np.nanmin(arr), np.nanmax(arr)
    if not np.isfinite(mn):
        mn = 0.0
    if not np.isfinite(mx) or mx <= mn:
        if convert_to_8_bit:
            return np.zeros_like(arr, dtype=np.uint8)
        return np.ascontiguousarray(arr).astype(original_dtype)
    arr = arr.astype(np.float64)
    arr = (arr - mn) / (mx - mn)
    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0).clip(0, 1)
    if convert_to_8_bit:
        return (arr * 255).astype(np.uint8)
    if original_dtype in (np.float32, np.float64):
        return arr.astype(original_dtype)
    if original_dtype == np.uint8:
        return (arr * 255).astype(np.uint8)
    return (arr * 65535).astype(np.uint16)


def resize_and_convert(
    image: np.ndarray,
    size: float,
    convert_to_8_bit: bool,
    normalize: bool = False,
) -> np.ndarray:
    """Resize and optionally convert. normalize=False: no per-image stretch (loading default).
    normalize=True: stretch each image to full range of target dtype (8/16/float)."""
    h, w = image.shape[:2]
    resized_image = cv2.resize(
        image,
        (int(w*size), int(h*size)),
        interpolation=cv2.INTER_AREA,
    )
    if normalize:
        return _stretch_to_full_range(
            resized_image, convert_to_8_bit, resized_image.dtype
        )
    if convert_to_8_bit:
        return _to_8bit_fixed_scale(resized_image)
    return resized_image


def resize_and_convert_to_8_bit(
    array: np.ndarray,
    size_ratio: float,
    convert_to_8_bit: bool,
    normalize: bool = False,
) -> np.ndarray:
    new_shape = tuple(int(dim * size_ratio) for dim in array.shape[:2])
    resized_array = cv2.resize(array, new_shape[::-1])
    if normalize:
        resized_array = _stretch_to_full_range(
            resized_array, convert_to_8_bit, resized_array.dtype
        )
    elif convert_to_8_bit:
        resized_array = _to_8bit_fixed_scale(resized_array)
    return resized_array


def adjust_ratio_for_memory(size_estimate: float, ram_size: float) -> float:
    if size_estimate <= 2**30 * ram_size:
        return 1.0
    adjusted_ratio = 2**30 * ram_size / size_estimate
    return adjusted_ratio


def smoothen(
    x: np.ndarray,
    y: np.ndarray,
    window_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    # smoothing the y data using convolution
    smoothed_y = np.convolve(
        y,
        np.ones(window_size) / window_size,
        mode='valid',
    )

    trim_start = window_size // 2
    trim_end = -(window_size // 2)

    if window_size % 2 == 0:
        trim_end = trim_end + 1

    # Trim the x data to align with smoothed y data
    trimmed_x = x[trim_start: trim_end]

    return trimmed_x, smoothed_y


def ensure_4d(images: np.ndarray) -> np.ndarray:
    if images.ndim == 4:
        return images
    return images[..., np.newaxis]


def sliding_mean_normalization(
    images: np.ndarray,
    window: int,
    lag: int,
) -> np.ndarray:
    """Computes sliding mean using Numba or cumulative sum (fallback).

    Used by ImageData when Divide source is "Sliding mean" (Ops tab).

    Optimized for CPU and Memory:
    - Uses Numba if available for parallel execution and minimal memory overhead.
    - Fallback: Uses np.cumsum for O(1) sliding window (vs O(window)), processing
      row-by-row to avoid allocating a full float32 cumsum array.
    """
    n = images.shape[0] - (lag + window)
    if n <= 0:
        return np.empty((0, *images.shape[1:]), dtype=np.float32)

    if optimized.HAS_NUMBA:
        # Ensure float32 for Numba
        if images.dtype != np.float32:
            # We must copy/convert because Numba expects specific type?
            # optimized.sliding_mean_numba expects float32
            # But creating a full float32 copy of 'images' (T,H,W,C) might OOM if images is huge uint8.
            # existing implementation: sl = images[:, i, ...].astype(np.float32)
            # It processes row by row, so it only allocates (T,W,C) float32 per row.

            # If we pass huge uint8 array to Numba function expecting float32, Numba might
            # 1. Reject it (signature mismatch if typed)
            # 2. Try to cast?

            # My Numba function signature is implicitly typed by usage or decorators?
            # @jit(nopython=True) handles types dynamically unless specified.
            # But inside I do `result = np.empty(..., dtype=np.float32)`.
            # And `current_sum = 0.0`.
            # If `images` is uint8, `images[...]` is uint8.
            # `current_sum += uint8` works (promotes to float).
            # So passing uint8 to Numba should work fine and be memory efficient!
            # It reads uint8, adds to float accumulator.
            # This is BETTER than astype(np.float32) which allocates huge array.

            # So I just pass it.
            pass

        return optimized.sliding_mean_numba(images, window, lag)

    # Pre-allocate result
    result = np.empty((n, *images.shape[1:]), dtype=np.float32)

    h = images.shape[1]

    # Process row by row to save memory (avoid full 4D cumsum allocation)
    for i in range(h):
        # Slice: (T, W, C)
        sl = images[:, i, ...].astype(np.float32)

        # Cumsum along time
        cs = np.cumsum(sl, axis=0)

        # Compute mean
        upper = cs[lag + window : lag + window + n]
        lower = cs[lag : lag + n]

        result[:, i, ...] = (upper - lower) / window

    return result


def sliding_mean_at_frame(
    images: np.ndarray,
    frame_idx: int,
    window: int,
) -> np.ndarray:
    """Sliding mean for a single frame's window. Returns (H, W, C)."""
    end = frame_idx + window
    if end > images.shape[0]:
        return np.zeros(images.shape[1:], dtype=np.float32)
    sl = images[frame_idx:end]
    return np.mean(sl.astype(np.float32), axis=0)


def _reduce_window(w: np.ndarray, method) -> np.ndarray:
    """Reduce (window,H,W,C) over axis 0. Returns (H,W,C)."""
    redop = ops.get(method)
    out = redop.reduce(w.astype(np.float32))
    return out[0]  # (1,H,W,C) -> (H,W,C)


def sliding_aggregate_at_frame(
    images: np.ndarray,
    frame_idx: int,
    window: int,
    method,
) -> np.ndarray:
    """Sliding aggregate for a single frame's window. Returns (H, W, C)."""
    from .ops import ReduceOperation
    if method == ReduceOperation.MEAN:
        return sliding_mean_at_frame(images, frame_idx, window)
    end = frame_idx + window
    if end > images.shape[0]:
        return np.zeros(images.shape[1:], dtype=np.float32)
    sl = images[frame_idx:end]
    return _reduce_window(sl, method)


def sliding_aggregate_normalization(
    images: np.ndarray,
    window: int,
    lag: int,
    method,
) -> np.ndarray:
    """Sliding aggregate over time. Uses Reduce method (MEAN, MAX, MIN, STD, MEDIAN)."""
    n = images.shape[0] - (lag + window)
    if n <= 0:
        return np.empty((0, *images.shape[1:]), dtype=np.float32)
    from .ops import ReduceOperation
    if method == ReduceOperation.MEAN and optimized.HAS_NUMBA:
        return optimized.sliding_mean_numba(images, window, lag)
    redop = ops.get(method)
    result = np.empty((n, *images.shape[1:]), dtype=np.float32)
    for i in range(n):
        start = lag + 1 + i
        end = start + window
        window_slice = images[start:end].astype(np.float32)
        reduced = redop.reduce(window_slice)
        result[i] = reduced[0]
    return result


def normalize(signal: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    min_ = signal.min()
    return (signal - min_) / (signal.max() - min_ + eps)


def unify_range(*signal: np.ndarray, eps: float = 1e-10) -> list[np.ndarray]:
    target_min = signal[0].min()
    target_max = signal[0].max()
    output = [signal[0]]
    for i in range(1, len(signal)):
        min_ = signal[i].min()
        output.append(
            target_min
            + (signal[i] - min_) / (signal[i].max() - min_ + eps)
            * (target_max - target_min)
        )
    return output
