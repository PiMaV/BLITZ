import cv2
import numpy as np
import numba


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
        return (resized_image / resized_image.max() * 255).astype(np.uint8)
    return resized_image


def resize_and_convert_to_8_bit(
    array: np.ndarray,
    size_ratio: float,
    convert_to_8_bit: bool,
) -> np.ndarray:
    new_shape = tuple(int(dim * size_ratio) for dim in array.shape[:2])
    resized_array = cv2.resize(array, new_shape[::-1])
    if convert_to_8_bit:
        resized_array = (
            resized_array / np.max(resized_array) * 255  # type: ignore
        ).astype('uint8')
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


@numba.njit(
    "f4[:,:,:,:](f4[:,:,:,:], i8, i8)",
    fastmath=True,
    parallel=True,
    cache=True,
)
def sliding_mean_normalization(
    images: np.ndarray,
    window: int,
    lag: int,
) -> np.ndarray:
    n = images.shape[0]-(lag+window)
    q, p, c = images.shape[1:]
    result = np.zeros((n, q, p, c), dtype=np.float32)
    for t in numba.prange(n):
        mean = np.zeros(images.shape[1:])
        for s in range(lag, lag+window):
            mean = mean + images[t+s+1]
        result[t] = mean / window
    return result
