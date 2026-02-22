import cv2
import numpy as np
# import numba
from . import optimized


def resize_and_convert(
    image: np.ndarray,
    size: float,
    convert_to_8_bit: bool,
) -> np.ndarray:
    h, w = image.shape[:2]
    resized_image = cv2.resize(
        image,
        (int(w*size), int(h*size)),
        interpolation=cv2.INTER_AREA,
    )
    if convert_to_8_bit:
        mx = resized_image.max()
        if mx <= 0:
            return np.zeros_like(resized_image, dtype=np.uint8)
        return (resized_image / mx * 255).astype(np.uint8)
    return resized_image


def resize_and_convert_to_8_bit(
    array: np.ndarray,
    size_ratio: float,
    convert_to_8_bit: bool,
) -> np.ndarray:
    new_shape = tuple(int(dim * size_ratio) for dim in array.shape[:2])
    resized_array = cv2.resize(array, new_shape[::-1])
    if convert_to_8_bit:
        mx = np.max(resized_array)
        if mx <= 0:
            return np.zeros_like(resized_array, dtype=np.uint8)
        resized_array = (resized_array / mx * 255).astype(np.uint8)
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
