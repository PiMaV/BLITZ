import json
import os
import warnings
from multiprocessing import Pool, cpu_count
from typing import Any, Optional

import cv2
import numpy as np
from natsort import natsorted

from ..tools import log
from .tools import (adjust_ratio_for_memory, create_info_image,
                    is_image_grayscale, resize_and_convert,
                    resize_and_convert_to_8_bit)

IMAGE_EXTENSIONS = (".jpg", ".png", ".jpeg", ".bmp", ".tiff", ".tif")
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov")
ARRAY_EXTENSIONS = (".npy", )
MULTICORE_THRESHOLD = 1.3 * (2**30)  # in GB


def from_file(
    filepath: Optional[str] = None,
    size: int = 1,
    ratio: int = 1,
    convert_to_8_bit: bool = False,
    ram_size: int = 1,
    grayscale: bool = False,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    if filepath is None:
        data, metadata = create_info_image(
            message="No data",
            color=(0, 0, 255),
        )
    else:
        dirname = os.path.dirname(filepath)
        file_extension = os.path.splitext(filepath)[1].lower()
        if file_extension in IMAGE_EXTENSIONS:
            data, metadata = _load_multiple_standard_images(
                filepath,
                size,
                ratio,
                convert_to_8_bit,
                ram_size,
                grayscale,
            )
        elif file_extension in ARRAY_EXTENSIONS:
            data, metadata = _load_numpy_array(
                dirname,
                size,
                ratio,
                convert_to_8_bit,
                ram_size,
                grayscale,
            )
        elif file_extension in VIDEO_EXTENSIONS:
            data, metadata = _load_video(
                filepath,
                size,
                ratio,
                convert_to_8_bit,
                ram_size,
                grayscale,
            )
        else:
            warnings.warn(
                f"Unsupported or unknown file type: {filepath}",
                UserWarning,
            )
            data, metadata = create_info_image(message="Filetype unsupported.")

    data = np.swapaxes(data, 1, 2)
    return data, metadata


def random(
    shape: tuple[int, ...] = (33, 150, 70),
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    data = np.random.normal(size=shape)
    all_metadata = []

    for i in range(shape[0]):
        metadata = {
            "file_size_MB": data[i].nbytes / 2**20,  # in MB
            "size": f"{data[i].shape[0]}x{data[i].shape[1]}",
            "dtype": str(data[i].dtype),
            "bit_depth": data[i].dtype.itemsize * 8,  # in bits
            "color_space": "Random",  # or whatever makes sense in your context
            "file_name": f"Frame_{i}.npy"  # filename as frame number
        }
        all_metadata.append(metadata)

    return data, all_metadata


def _load_image(
    file_path: str,
    format: int = cv2.IMREAD_UNCHANGED,
) -> Optional[np.ndarray]:
    try:
        img = cv2.imread(file_path, format)
    except Exception:
        return None
    if img is None:
        return None
    if len(img.shape) == 3:
        if img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    return img


def _load_image_with_metadata(
    f: str,
    dirname: str,
    size: int,
    convert_to_8_bit: bool,
    load_format: int,
) -> tuple[Optional[np.ndarray], Optional[dict[str, Any]]]:
    image_data = _load_image(os.path.join(dirname, f), load_format)
    if image_data is None:
        return None, None
    image_data = resize_and_convert(image_data, size, convert_to_8_bit)
    metadata = {
        "file_size_MB": os.path.getsize(os.path.join(dirname, f)) / 2**20,
        "size": f"{image_data.shape[0]}x{image_data.shape[1]}",
        "dtype": str(image_data.dtype),
        "bit_depth": image_data.dtype.itemsize * 8,
        "color_space": "RGB" if len(image_data.shape) == 3 else "Grayscale",
        "file_name": f,
    }
    return image_data, metadata


def _load_multiple_standard_images(
    filepath: str,
    size: int,
    ratio: float,
    convert_to_8_bit: bool,
    ram_size: int,
    grayscale: bool,
    nr_img_to_go_multicore: int = 333,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    dirname = os.path.dirname(filepath)
    file_extension = os.path.splitext(filepath)[1].lower()
    image_files = [
        f for f in os.listdir(dirname)
        if os.path.isfile(os.path.join(dirname, f))
            and os.path.splitext(f)[1].lower() == file_extension
    ]
    image_files = natsorted(image_files)

    if grayscale:
        selected_image = cv2.imread(
            filepath,
            cv2.IMREAD_ANYDEPTH | cv2.IMREAD_GRAYSCALE,
        )
    else:
        selected_image = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)

    selected_image = resize_and_convert(selected_image, size, convert_to_8_bit)
    is_grayscale = is_image_grayscale(selected_image)

    total_size_estimate = len(image_files) * size**2 * (
        selected_image.size if convert_to_8_bit else selected_image.nbytes
    )

    adjusted_ratio = adjust_ratio_for_memory(total_size_estimate, ram_size)

    # Use the smaller ratio
    ratio = min(ratio, adjusted_ratio)
    full_dataset_size = len(image_files)
    image_files = image_files[::int(np.ceil(1/ratio))]
    log(f"Loading {len(image_files)}/{full_dataset_size} images")

    load_format = (
        cv2.IMREAD_ANYDEPTH | cv2.IMREAD_GRAYSCALE
        if is_grayscale else cv2.IMREAD_UNCHANGED
    )

    faulty_images = []
    valid_data_with_metadata = []

    if (len(image_files) > nr_img_to_go_multicore
            or total_size_estimate > MULTICORE_THRESHOLD):
        pool = Pool(cpu_count())
        results = pool.starmap(
            _load_image_with_metadata,
            [(f, dirname, size, convert_to_8_bit, load_format)
             for f in image_files]
        )
        pool.close()
        pool.join()
    else:
        results = [
            _load_image_with_metadata(
                f, dirname, size, convert_to_8_bit, load_format
            ) for f in image_files
        ]

    for result in results:
        if (result[0] is not None
                and result[0].shape[:2] == selected_image.shape[:2]):
            valid_data_with_metadata.append(result)
        else:
            faulty_images.append(
                result[1]["file_name"] if result[1] is not None else "Unknown"
            )

    if valid_data_with_metadata:
        images = np.stack([img for img, _ in valid_data_with_metadata])
        all_metadata = [meta for _, meta in valid_data_with_metadata]
    else:
        log("Could not load any valid images")
        images, all_metadata = create_info_image()

    if faulty_images:
        log(f"Could not load {len(faulty_images)} images: {faulty_images}")

    return images, all_metadata


def _load_numpy_array(
    folder_path,
    size,
    ratio,
    convert_to_8_bit,
    ram_size,
    grayscale,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    if grayscale:
        log("Warning: Grayscale not implemented for numpy arrays yet")
    array_files = [f for f in os.listdir(folder_path) if f.endswith('.npy')]

    first_array = np.load(os.path.join(folder_path, array_files[0]))
    avg_array_size = first_array.nbytes  # in bytes

    total_size_estimate = avg_array_size * len(array_files)

    adjusted_ratio = adjust_ratio_for_memory(total_size_estimate, ram_size)

    # Use the smaller ratio
    ratio = min(ratio, adjusted_ratio)
    full_dataset_size = len(array_files)
    array_files = array_files[::int(np.ceil(1/ratio))]
    log(f"Loading {len(array_files)}/{full_dataset_size} arrays")

    matrix_list = []
    metadata_list = []

    if len(array_files) > 333 or total_size_estimate > MULTICORE_THRESHOLD:
        multiprocessing = True
    else:
        multiprocessing = False

    if multiprocessing:
        with Pool(cpu_count()) as pool:
            results = pool.starmap(
                _load_and_process_file,
                [(f, folder_path, size, convert_to_8_bit) for f in array_files]
            )
        matrix_list, metadata_sublists = zip(*results)
        metadata_list.extend(metadata_sublists)
    else:
        for f in array_files:
            array = np.load(os.path.join(folder_path, f))
            resized_array = resize_and_convert_to_8_bit(
                array,
                size,
                convert_to_8_bit,
            )
            matrix_list.append(resized_array)
            metadata = {
                'filename': f,
                'shape': resized_array.shape,
                'dtype': resized_array.dtype,
            }
            metadata_list.append(metadata)

    matrix = np.stack(matrix_list)  # type: ignore
    return matrix, metadata_list


def _load_and_process_file(
    f: str,
    folder_path: str,
    size: int,
    convert_to_8_bit: bool,
) -> tuple[np.ndarray, dict[str, Any]]:
    array = np.load(os.path.join(folder_path, f))
    resized_array = resize_and_convert_to_8_bit(array, size, convert_to_8_bit)
    metadata = {
        'filename': f,
        'shape': resized_array.shape,
        'dtype': resized_array.dtype,
    }
    return resized_array, metadata


def _load_video(
    filepath: str,
    size: int,
    ratio: float,
    convert_to_8_bit: bool,
    ram_size: int,
    grayscale: bool,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    cap = cv2.VideoCapture(filepath)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    codec = int(cap.get(cv2.CAP_PROP_FOURCC))
    fourcc_str = "".join([chr((codec >> 8 * i) & 0xFF) for i in range(4)])
    cap.release()

    memory_estimate = width * height * 3 * frame_count
    log(f"Estimated size to load: {memory_estimate / 2**20:.2f} MB")
    adjusted_ratio = adjust_ratio_for_memory(memory_estimate, ram_size)

    ratio = min(ratio, adjusted_ratio)
    if ratio == 1:
        log("No adjustment to ratio required, loading the full dataset")
    else:
        log(f"Adjusted ratio for subset extraction: {ratio:.4f}")

    skip_frames = int(1 / ratio) - 1

    video = cv2.VideoCapture(filepath)
    frames = []
    metadata = []

    while True:
        ret, frame = video.read()
        if not ret:
            break

        frame_number = len(frames)  # current frame number starting from 0
        frames.append(resize_and_convert(
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if not grayscale else (
                cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            ),
            size,
            convert_to_8_bit
        ))

        single_frame_metadata = {
            "file_size_MB": os.path.getsize(filepath) / 2**20,
            "file_name": str(frame_number),
            "size": f"{frame.shape[0]}x{frame.shape[1]}",
            "dtype": str(frame.dtype),
            "bit_depth": frame.dtype.itemsize * 8,
            "color_space": "RGB" if len(frame.shape) == 3 else "Grayscale",
            "fps": fps,
            "frame_count": frame_count,
            "reduced_frame_count": len(frames),
            "codec": fourcc_str
        }
        metadata.append(single_frame_metadata)

        # skip the next `skip_frames` number of frames without decoding.
        for _ in range(skip_frames):
            video.grab()

    video.release()
    return np.stack(frames), metadata


def tof_from_json(file_path: str) -> np.ndarray:
    with open(file_path, 'r') as file:
        raw_data = json.load(file)

    # Filter out data where offset is negative
    filtered_data = [item for item in raw_data if item["offset"] >= 0]

    if not filtered_data:
        # If all data was filtered out, return empty 2D numpy array
        return np.empty((0,2))

    # Extract time and tof data from the filtered list
    time = np.array([item["offset"] for item in filtered_data])
    tof = np.array([item["tof_data"] for item in filtered_data])

    # Stack the arrays to create a 2D array where
    # the first column is time and the second is tof_data
    data_array = np.column_stack((time, tof))

    return data_array
