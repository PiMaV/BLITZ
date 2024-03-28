import json
import os
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from .. import settings
from ..tools import log
from .image import ImageData
from .tools import (adjust_ratio_for_memory, resize_and_convert,
                    resize_and_convert_to_8_bit)

IMAGE_EXTENSIONS = (".jpg", ".png", ".jpeg", ".bmp", ".tiff", ".tif")
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov")
ARRAY_EXTENSIONS = (".npy", )


class DataLoader:

    @staticmethod
    def _is_array(file: Path) -> bool:
        return file.suffix.lower() in ARRAY_EXTENSIONS

    @staticmethod
    def _is_image(file: Path) -> bool:
        return file.suffix.lower() in IMAGE_EXTENSIONS

    @staticmethod
    def _is_video(file: Path) -> bool:
        return file.suffix.lower() in VIDEO_EXTENSIONS

    def __init__(
        self,
        size_ratio: float = 1.0,
        subset_ratio: float = 1.0,
        max_ram: float = 1.0,
        convert_to_8_bit: bool = False,
        grayscale: bool = False,
    ) -> None:
        self.max_ram = max_ram
        self.size_ratio = size_ratio
        self.subset_ratio = subset_ratio
        self.convert_to_8_bit = convert_to_8_bit
        self.grayscale = grayscale

    def load(self, path: Optional[Path] = None) -> ImageData:
        if path is None:
            return DataLoader.from_text("No data")

        if path.is_dir():
            return self._load_folder(path)
        else:
            return self._load_file(path)

    def _load_file(self, path: Path) -> ImageData:
        if DataLoader._is_array(path):
            return self._load_array(path)
        elif DataLoader._is_image(path):
            image, metadata = self._load_image(path)
            return ImageData(image[np.newaxis, ...], [metadata])
        elif DataLoader._is_video(path):
            return self._load_video(path)
        else:
            log("Error: Unsupported file type", color="red")
            return DataLoader.from_text(
                "Unsupported file type",
                color=(255, 0, 0),
            )

    def _load_folder(self, path: Path) -> ImageData:
        content = [f for f in path.iterdir() if not f.is_dir()]
        suffixes = {s: len([f for f in content if f.suffix == s])
                    for s in set(f.suffix for f in content)}
        most_frequent_suffix = max(suffixes, key=suffixes.get)  # type: ignore
        if len(suffixes) > 1:
            log("Warning: folder contains multiple file types; "
                f"Loading all {most_frequent_suffix!r} files")
            content = [f for f in content if f.suffix == most_frequent_suffix]

        if DataLoader._is_image(content[0]):
            sample, _ = self._load_image(content[0])
            total_size_estimate = len(content) * self.size_ratio**2 * (
                sample.size if self.convert_to_8_bit else sample.nbytes
            )
            load_function = self._load_image
        elif DataLoader._is_array(content[0]):
            sample, _ = self._load_single_array(content[0])
            total_size_estimate = sample.nbytes * len(content)
            load_function = self._load_single_array
        else:
            log("Error: Unknown file extension in folder", color="red")
            return DataLoader.from_text(
                "Unsupported file type",
                color=(255, 0, 0),
            )

        adjusted_ratio = adjust_ratio_for_memory(
            total_size_estimate, self.max_ram,
        )
        ratio = min(self.subset_ratio, adjusted_ratio)
        full_dataset_size = len(content)
        content = content[::int(np.ceil(1 / ratio))]
        log(f"Loading {len(content)}/{full_dataset_size} files", color="green")

        if (len(content) > settings.get("data/multicore_files_threshold")
                or total_size_estimate >
                    settings.get("data/multicore_size_threshold")):
            with Pool(cpu_count()) as pool:
                results = pool.starmap(
                    load_function,
                    [(f, ) for f in content],
                )
            matrices, metadata = zip(*results)
        else:
            matrices, metadata = [], []
            for f in content:
                matrix, meta = load_function(f)
                matrices.append(matrix)
                metadata.append(meta)

        try:
            matrices = np.stack(matrices)
        except:
            log(
                "Error loading files: shapes of images do not match",
                color="red",
            )
            return DataLoader.from_text(
                "Error loading files",
                color=(255, 0, 0),
            )
        return ImageData(matrices, metadata)

    def _load_image(self, path: Path) -> tuple[np.ndarray, dict[str, Any]]:
        image = cv2.imread(
            str(path),
            cv2.IMREAD_UNCHANGED if not self.grayscale else (
                cv2.IMREAD_ANYDEPTH | cv2.IMREAD_GRAYSCALE
            ),
        )
        if image.ndim == 3:
            if image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            elif image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)

        image = resize_and_convert(
            image,
            self.size_ratio,
            self.convert_to_8_bit,
        )
        metadata = {
            "file_name": path.name,
            "file_size_MB": os.path.getsize(path) / 2**20,
            "size": f"{image.shape[0]}x{image.shape[1]}",
            "dtype": str(image.dtype),
            "bit_depth": image.dtype.itemsize * 8,
            "grayscale": image.ndim == 2,
        }

        return np.swapaxes(image, 0, 1), metadata

    def _load_video(self, path: Path) -> ImageData:
        cap = cv2.VideoCapture(str(path))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        codec = int(cap.get(cv2.CAP_PROP_FOURCC))
        fourcc_str = "".join([chr((codec >> 8 * i) & 0xFF) for i in range(4)])
        cap.release()

        memory_estimate = width * height * 3 * frame_count
        log(f"Estimated size to load: {memory_estimate / 2**20:.2f} MB")
        adjusted_ratio = adjust_ratio_for_memory(
            memory_estimate,
            self.max_ram,
        )

        ratio = min(self.subset_ratio, adjusted_ratio)
        if ratio == 1:
            log("No adjustment to ratio required, loading the full dataset")
        else:
            log(f"Adjusted ratio for subset extraction: {ratio:.4f}",
                color="green")

        skip_frames = int(1 / ratio) - 1

        video = cv2.VideoCapture(str(path))
        frames = []
        metadata = []

        while True:
            ret, frame = video.read()
            if not ret:
                break

            frame_number = len(frames)  # current frame number starting from 0
            frames.append(resize_and_convert(
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    if not self.grayscale else
                    cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                self.size_ratio,
                self.convert_to_8_bit,
            ))

            metadata.append({
                "file_name": str(frame_number),
                "file_size_MB": os.path.getsize(path) / 2**20,
                "size": f"{frame.shape[0]}x{frame.shape[1]}",
                "dtype": str(frame.dtype),
                "bit_depth": frame.dtype.itemsize * 8,
                "color_space": "RGB" if len(frame.shape) == 3 else "Grayscale",
                "fps": fps,
                "frame_count": frame_count,
                "reduced_frame_count": len(frames),
                "codec": fourcc_str,
            })

            # skip the next `skip_frames` number of frames without decoding.
            for _ in range(skip_frames):
                video.grab()

        video.release()
        return ImageData(np.swapaxes(np.stack(frames), 1, 2), metadata)

    def _load_array(self, path: Path) -> ImageData:
        array: np.ndarray = np.load(path)
        match array.ndim:
            case 4:
                if self.grayscale:
                    weights = np.array([0.2989, 0.5870, 0.1140])
                    array = np.sum(array * weights, axis=-1)
            case 3:
                if not self.grayscale:
                    if array.shape[2] != 3:
                        log("Warning: File does not contain color images, "
                            "loading as grayscale", color="green")
                    else:
                        array = array[np.newaxis, ...]
            case 2:
                if not self.grayscale:
                    log("Warning: Loading files as grayscale images",
                        color="green")
                array = array[np.newaxis, ...]
            case _:
                log(f"Error: Unsupported array shape {array.shape}",
                    color="red")
                return DataLoader.from_text(
                    "Error loading files", color=(255, 0, 0)
                )
        # now the first dimension is time
        function_ = lambda x: resize_and_convert_to_8_bit(
            x,
            self.size_ratio,
            self.convert_to_8_bit,
        )
        total_size_estimate = array[0].nbytes * array.shape[0]
        if (array.shape[0] > settings.get("data/multicore_files_threshold")
                or total_size_estimate >
                    settings.get("data/multicore_size_threshold")):
            with Pool(cpu_count()) as pool:
                matrices = pool.starmap(
                    function_,
                    [(a, ) for a in array],
                )
        else:
            matrices = []
            for a in array:
                matrix = function_(a)
                matrices.append(matrix)

        array = np.swapaxes(np.stack(matrices), 1, 2)
        metadata = [{
            'file_name': path.name + f"-{i}",
            'shape': array.shape,
            'dtype': array.dtype,
        } for i in range(array.shape[0])]
        return ImageData(array, metadata)

    def _load_single_array(
        self,
        path: Path,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        array: np.ndarray = np.load(path)
        if array.ndim >= 4 or (array.ndim == 3 and array.shape[2] != 3):
            log(f"Error: Unsupported array shape {array.shape}", color="red")
            data = DataLoader.from_text(
                "Error loading files",
                color=(255, 0, 0),
            )
            return data.image[0], data.meta[0]

        if array.ndim == 3 and self.grayscale:
            weights = np.array([0.2989, 0.5870, 0.1140])
            array = np.sum(array * weights, axis=-1)

        array = np.swapaxes(resize_and_convert_to_8_bit(
            array,
            self.size_ratio,
            self.convert_to_8_bit,
        ), 0, 1)
        metadata = {
            'file_name': path.name,
            'shape': array.shape,
            'dtype': array.dtype,
        }
        return array, metadata

    @staticmethod
    def from_text(
        text: str,
        height: int = 50,
        width: int = 280,
        color: tuple[int, int, int] = (0, 0, 255),
    ) -> ImageData:
        img = np.zeros((height, width, 3), dtype=np.uint8)
        cv2.putText(
            img=img,
            text=text,
            org=(5, height//2),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.4,
            color=color,
            thickness=1,
        )
        metadata = [{
            "file_name": "-",
            "file_size_MB": img.nbytes / 2**20,  # In MB
        }]
        return ImageData(np.swapaxes(img[np.newaxis, ...], 1, 2), metadata)


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
