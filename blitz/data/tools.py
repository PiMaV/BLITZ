from typing import Any

import cv2
import numpy as np


def resize_and_convert(
    image: np.ndarray,
    size: int = 1,
    convert_to_8_bit: bool = False,
) -> np.ndarray:
    h, w = image.shape[:2]
    resized_image = cv2.resize(
        image,
        (int(w*size), int(h*size)),
        interpolation=cv2.INTER_AREA,
    )
    if convert_to_8_bit:
        return (
            resized_image / np.max(resized_image) * 255  # type: ignore
        ).astype(np.uint8)
    return resized_image


def resize_and_convert_to_8_bit(
    array: np.ndarray,
    size_ratio: int,
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


def create_info_image(
    message: str = 'Loading failed',
    height: int = 50,
    width: int = 280,
    color: tuple[int, int, int] = (255, 0, 0),
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    img = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.putText(
        img=img,
        text=message,
        org=(5, height//2),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.4,
        color=color,
        thickness=1,
    )
    img = np.expand_dims(img, axis=0)
    all_metadata = [{
        "file_size_MB": img.nbytes / 2**20,  # In MB
        "file_name": "Error.png",  # Filename as frame number
    }]
    return img, all_metadata


def is_image_grayscale(image: np.ndarray) -> bool:
    if len(image.shape) == 2:
        return True
    if len(image.shape) == 3 and image.shape[2] == 1:
        return True
    if len(image.shape) == 3:
        return bool(
            np.all(image[:, :, 0] == image[:, :, 1])
            and np.all(image[:, :, 1] == image[:, :, 2])
        )
    return False


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
