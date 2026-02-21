import cv2
import numpy as np


def ensure_float32(image: np.ndarray) -> np.ndarray:
    if image.dtype != np.float32:
        return image.astype(np.float32)
    return image


def median(image: np.ndarray, ksize: int) -> np.ndarray:
    """Apply median filter to remove hot pixels/noise.
    ksize must be odd.
    """
    if ksize < 1:
        return image
    if ksize % 2 == 0:
        ksize += 1

    # Median Blur only supports uint8, uint16, int16, float32
    # But usually float32 is fine.
    # Note: cv2.medianBlur works on single channel or 3 channel images.
    # If image is 4D (T, H, W, C) or 3D (H, W, C), handle accordingly.
    # Since operations usually run on single image (H, W, C) or single frame (H, W) in the pipeline:
    if image.ndim == 3 and image.shape[2] == 1:
        return cv2.medianBlur(image, ksize)
    elif image.ndim == 2:
        return cv2.medianBlur(image, ksize)
    else:
        # Multi-channel logic if needed, but usually we process per channel or assume grayscale for scientific data
        # If float32, cv2.medianBlur is supported.
        return cv2.medianBlur(image, ksize)


def min_filter(image: np.ndarray, ksize: int) -> np.ndarray:
    """Apply minimum filter (Erosion)."""
    if ksize < 1:
        return image
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
    return cv2.erode(image, kernel)


def max_filter(image: np.ndarray, ksize: int) -> np.ndarray:
    """Apply maximum filter (Dilation)."""
    if ksize < 1:
        return image
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
    return cv2.dilate(image, kernel)


def gaussian_blur(image: np.ndarray, sigma: float) -> np.ndarray:
    """Apply Gaussian Blur (Lowpass)."""
    if sigma <= 0:
        return image
    # ksize=0 means computed from sigma
    return cv2.GaussianBlur(image, (0, 0), sigmaX=sigma, sigmaY=sigma)


def highpass(image: np.ndarray, sigma: float) -> np.ndarray:
    """Apply Highpass filter (Original - Lowpass)."""
    if sigma <= 0:
        return np.zeros_like(image)
    low = gaussian_blur(image, sigma)
    return image - low


def clahe(image: np.ndarray, clip_limit: float = 2.0, tile_grid_size: int = 8) -> np.ndarray:
    """Contrast Limited Adaptive Histogram Equalization.
    Requires uint8/uint16 input for cv2.createCLAHE.
    If input is float, we normalize to 0-65535 (uint16) apply CLAHE, then convert back.
    """
    if tile_grid_size < 1:
        return image

    orig_dtype = image.dtype
    if orig_dtype == np.float32 or orig_dtype == np.float64:
        # Normalize to 0-1 range first if not already
        min_val, max_val = image.min(), image.max()
        if max_val == min_val:
            return image

        # Temporarily convert to uint16 for CLAHE
        norm = (image - min_val) / (max_val - min_val)
        norm_uint16 = (norm * 65535).astype(np.uint16)

        clahe_obj = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))

        # CLAHE only works on single channel.
        if image.ndim == 3:
            res_channels = []
            for c in range(image.shape[2]):
                res_channels.append(clahe_obj.apply(norm_uint16[..., c]))
            res_uint16 = np.stack(res_channels, axis=-1)
        else:
            res_uint16 = clahe_obj.apply(norm_uint16)

        # Convert back to original range
        return (res_uint16.astype(np.float32) / 65535.0) * (max_val - min_val) + min_val

    elif orig_dtype == np.uint8:
        clahe_obj = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
        if image.ndim == 3:
            res_channels = []
            for c in range(image.shape[2]):
                res_channels.append(clahe_obj.apply(image[..., c]))
            return np.stack(res_channels, axis=-1)
        return clahe_obj.apply(image)

    return image


def local_normalize_mean(image: np.ndarray, ksize: int) -> np.ndarray:
    """Local Normalization: (Pixel - Mean) / (Std + eps) or similar.
    Here implementing a simpler version requested: divide by local mean (or blur).
    Let's implement: Image / (Gaussian(Image) + eps).
    This highlights local variations relative to background.
    """
    if ksize < 1:
        return image
    # Use Gaussian as "local mean" approximation
    # Sigma roughly ksize / 6 for 99% coverage or just let cv2 decide from ksize
    # ksize for GaussianBlur must be odd
    if ksize % 2 == 0:
        ksize += 1

    local_mean = cv2.GaussianBlur(image, (ksize, ksize), 0)

    # Avoid division by zero
    return image / (local_mean + 1e-6)


def threshold_binary(image: np.ndarray, thresh: float, maxval: float = 1.0, type_: int = cv2.THRESH_BINARY) -> np.ndarray:
    """Apply thresholding."""
    _, res = cv2.threshold(image, thresh, maxval, type_)
    return res


def clip_values(image: np.ndarray, min_val: float, max_val: float) -> np.ndarray:
    """Clip values to range."""
    return np.clip(image, min_val, max_val)

def histogram_clipping(image: np.ndarray, min_percentile: float, max_percentile: float) -> np.ndarray:
    """Clip image intensity based on percentiles."""
    # Compute percentiles
    low = np.percentile(image, min_percentile)
    high = np.percentile(image, max_percentile)
    return np.clip(image, low, high)
