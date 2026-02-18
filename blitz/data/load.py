import json
import os
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
import pydicom
from natsort import natsorted

from .. import settings
from ..tools import log
from .image import DicomMetaData, ImageData, MetaData, VideoMetaData
from .tools import (adjust_ratio_for_memory, resize_and_convert,
                    resize_and_convert_to_8_bit)


def get_video_preview(
    path: Path,
    n_frames: int = 10,
    size_ratio: float = 0.2,
    grayscale: bool = False,
    mode: str = "max",
    normalize: bool = True,
) -> np.ndarray | None:
    """Laedt eine schnelle Vorschau aus dem Video.

    Args:
        path: Videopfad
        n_frames: Anzahl Frames (verteilt ueber das Video)
        size_ratio: Skalierung (0.2 = 20% fuer schnelles Laden)
        grayscale: Graustufen
        mode: "max" = MAX ueber Frames, "single" = nur mittlerer Frame
        normalize: Min-Max auf 0-255 strecken fuer bessere Sichtbarkeit

    Returns:
        uint8 array (H, W) oder (H, W, 3), oder None bei Fehler
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        cap.release()
        return None
    frames_to_read = (
        np.linspace(0, frame_count - 1, min(n_frames, frame_count), dtype=int)
        if mode == "max" and frame_count > 1
        else [frame_count // 2]
    )
    collected = []
    for fi in frames_to_read:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ret, frame = cap.read()
        if not ret:
            continue
        if grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = resize_and_convert(frame, size_ratio, convert_to_8_bit=False)
        collected.append(frame)
    cap.release()
    if not collected:
        return None
    if mode == "max" and len(collected) > 1:
        stack = np.stack(collected)
        out = np.max(stack, axis=0).astype(np.uint8)
    else:
        out = collected[0].astype(np.uint8)
    if normalize and out.size > 0:
        p_lo, p_hi = np.percentile(out, (2, 98))
        if p_hi > p_lo:
            out = np.clip(
                (out.astype(np.float32) - p_lo) / (p_hi - p_lo) * 255, 0, 255
            ).astype(np.uint8)
    return out


def get_image_preview(
    path: Path,
    size_ratio: float = 0.3,
    grayscale: bool = False,
    mode: str = "max",
    normalize: bool = True,
    n_samples: int = 10,
) -> np.ndarray | None:
    """Load a fast preview from an image file or folder.

    Args:
        path: Image file or folder path
        size_ratio: Scale factor for fast loading
        grayscale: Convert to grayscale
        mode: "max" = MAX across sampled images, "single" = first image only
        normalize: Min-max stretch to 0-255 for visibility
        n_samples: For folders: number of images to sample for MAX

    Returns:
        uint8 array (H, W) or (H, W, 3), or None on error
    """
    def _load_one(p: Path) -> np.ndarray | None:
        img = cv2.imread(
            str(p),
            cv2.IMREAD_UNCHANGED if not grayscale else (
                cv2.IMREAD_ANYDEPTH | cv2.IMREAD_GRAYSCALE
            ),
        )
        if img is None:
            return None
        if img.ndim == 3:
            if img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            elif img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        h, w = img.shape[:2]
        ratio = size_ratio
        if h > 0 and w > 0 and (int(h * ratio) < 1 or int(w * ratio) < 1):
            ratio = max(1.0 / h, 1.0 / w, size_ratio)
        return resize_and_convert(img, ratio, convert_to_8_bit=False)

    if path.is_file():
        if not DataLoader._is_image(path):
            return None
        img = _load_one(path)
        if img is None:
            return None
        out = img.astype(np.uint8)
    else:
        content = [f for f in path.iterdir() if not f.is_dir() and DataLoader._is_image(f)]
        content = natsorted(content)
        if not content:
            return None
        if mode == "single" or len(content) == 1:
            img = _load_one(content[0])
            if img is None:
                return None
            out = img.astype(np.uint8)
        else:
            indices = np.linspace(0, len(content) - 1, min(n_samples, len(content)), dtype=int)
            collected = []
            for i in indices:
                img = _load_one(content[i])
                if img is not None:
                    collected.append(img)
            if not collected:
                return None
            stack = np.stack(collected)
            out = np.max(stack, axis=0).astype(np.uint8)

    if normalize and out.size > 0:
        p_lo, p_hi = np.percentile(out, (2, 98))
        if p_hi > p_lo:
            out = np.clip(
                (out.astype(np.float32) - p_lo) / (p_hi - p_lo) * 255, 0, 255
            ).astype(np.uint8)
    return out


def get_sample_format(path: Path) -> tuple[bool, bool]:
    """Quick sample to detect if source is grayscale and uint8. Returns (is_grayscale, is_uint8)."""
    if path.is_file():
        if DataLoader._is_video(path):
            cap = cv2.VideoCapture(str(path))
            if not cap.isOpened():
                return False, False
            ret, frame = cap.read()
            cap.release()
            if not ret or frame is None:
                return False, False
            is_gray = frame.ndim == 2 or (frame.ndim == 3 and frame.shape[2] == 1)
            return is_gray, frame.dtype == np.uint8
        if DataLoader._is_image(path):
            img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if img is None:
                return False, False
            is_gray = img.ndim == 2 or (img.ndim == 3 and img.shape[2] == 1)
            return is_gray, img.dtype == np.uint8
    else:
        content = [f for f in path.iterdir() if not f.is_dir() and DataLoader._is_image(f)]
        content = natsorted(content)
        if not content:
            return False, False
        return get_sample_format(content[0])
    return False, False


def get_sample_format_display(path: Path) -> str:
    """Quick sample to get human-readable format string. E.g. 'Grayscale, 8 bit' or 'RGB, 16 bit'."""
    arr = None
    if path.is_file():
        if DataLoader._is_video(path):
            cap = cv2.VideoCapture(str(path))
            if not cap.isOpened():
                return "Unknown"
            ret, frame = cap.read()
            cap.release()
            if not ret or frame is None:
                return "Unknown"
            arr = frame
        elif DataLoader._is_image(path):
            arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        else:
            return "Unknown"
    else:
        content = [f for f in path.iterdir() if not f.is_dir() and DataLoader._is_image(f)]
        content = natsorted(content)
        if not content:
            return "Unknown"
        return get_sample_format_display(content[0])
    if arr is None:
        return "Unknown"
    color = "Grayscale" if arr.ndim == 2 or (arr.ndim == 3 and arr.shape[2] == 1) else "RGB"
    bits = arr.dtype.itemsize * 8
    return f"{color}, {bits} bit"


def get_sample_bytes_per_pixel(path: Path) -> int:
    """Bytes per pixel of the source (1 for uint8, 2 for uint16, etc.). Used for RAM estimate."""
    if path.is_file():
        if DataLoader._is_video(path):
            cap = cv2.VideoCapture(str(path))
            if not cap.isOpened():
                return 1
            ret, frame = cap.read()
            cap.release()
            if not ret or frame is None:
                return 1
            return int(frame.dtype.itemsize)
        if DataLoader._is_image(path):
            img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if img is None:
                return 1
            return int(img.dtype.itemsize)
        return 1
    content = [f for f in path.iterdir() if not f.is_dir() and DataLoader._is_image(f)]
    content = natsorted(content)
    if not content:
        return 1
    return get_sample_bytes_per_pixel(content[0])


def get_image_metadata(path: Path) -> dict | None:
    """Get metadata for image file or folder. Returns dict with file_name, size (h,w), file_count."""
    if path.is_file():
        if not DataLoader._is_image(path):
            return None
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            return None
        h, w = img.shape[:2]
        return {"file_name": path.name, "size": (h, w), "file_count": 1}
    content = [f for f in path.iterdir() if not f.is_dir() and DataLoader._is_image(f)]
    content = natsorted(content)
    if not content:
        return None
    img = cv2.imread(str(content[0]), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    h, w = img.shape[:2]
    return {"file_name": path.name, "size": (h, w), "file_count": len(content)}


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

    @staticmethod
    def _is_dicom(path: Path) -> bool:
        try:
            pydicom.dcmread(path)
            return True
        except Exception:
            return False

    @staticmethod
    def get_video_metadata(path: Path) -> VideoMetaData:
        cap = cv2.VideoCapture(str(path))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        codec = int(cap.get(cv2.CAP_PROP_FOURCC))
        fourcc_str = "".join([chr((codec >> 8 * i) & 0xFF) for i in range(4)])
        cap.release()

        return VideoMetaData(
            file_name=path.name,
            file_size_MB=os.path.getsize(path) / 2**20,
            size=(height, width),
            dtype=np.uint8, # Placeholder
            bit_depth=8, # Placeholder
            color_model="rgb",
            fps=fps,
            frame_count=frame_count,
            reduced_frame_count=0,
            codec=fourcc_str,
        )

    def __init__(
        self,
        size_ratio: float = 1.0,
        subset_ratio: float = 1.0,
        max_ram: float = 1.0,
        convert_to_8_bit: bool = False,
        grayscale: bool = False,
        mask: Optional[tuple[slice, slice]] = None,
        crop: Optional[tuple[int, int]] = None
    ) -> None:
        self.max_ram = max_ram
        self.size_ratio = size_ratio
        self.subset_ratio = subset_ratio
        self.convert_to_8_bit = convert_to_8_bit
        self.grayscale = grayscale
        self.mask = mask
        if mask is not None:
            self.mask = (mask[1], mask[0])
        self.crop = crop

    def load(
        self,
        path: Optional[Path] = None,
        progress_callback: Optional[Callable[[int], None]] = None,
        message_callback: Optional[Callable[[str], None]] = None,
        **kwargs,
    ) -> ImageData:
        if path is None:
            return DataLoader.from_text("  Load data", 50, 100)

        if path.is_dir():
            return self._load_folder(
                path,
                progress_callback=progress_callback,
                message_callback=message_callback,
            )
        else:
            if DataLoader._is_video(path):
                return self._load_video(path, progress_callback=progress_callback, **kwargs)
            return self._load_file(path, progress_callback=progress_callback)

    def _log_arguments(self, data: ImageData) -> None:
        args = [
            ("8bit", str(self.convert_to_8_bit)),
            ("grayscale", str(data.meta[0].color_model == "grayscale")),
            ("max. RAM", round(self.max_ram, 3)),
            ("Size-ratio", round(self.size_ratio, 3)),
            ("Subset-ratio", round(self.subset_ratio, 3)),
            ("Size (MB)", "{:.5s}".format(f"{data.image.nbytes/2**20:.5f}")),
        ]
        string = f"{'Dataset':-^20}" + "\n"
        for x, y in args:
            string += "  |{:<12} {:>5}|\n".format(x, y)
        string += "  " + 20*"-"
        log(string, color="green")

    def _load_file(
        self,
        path: Path,
        progress_callback: Optional[Callable[[int], None]] = None,
    ) -> ImageData:
        if progress_callback is not None:
            progress_callback(100)
        if DataLoader._is_array(path):
            return self._load_array(path)
        elif DataLoader._is_image(path):
            image, metadata = self._load_image(path)
            done = ImageData(image[np.newaxis, ...], [metadata])
            self._log_arguments(done)
            return done
        elif DataLoader._is_video(path):
            # This path is generally not reached if called via load() for video,
            # but kept for compatibility.
            return self._load_video(path)
        elif DataLoader._is_dicom(path):
            image, metadata = self._load_dicom_file(path)
            done = ImageData(image[np.newaxis, ...], [metadata])
            self._log_arguments(done)
            return done
        else:
            log("Error: Unsupported file type", color="red")
            return DataLoader.from_text(
                "Unsupported file type",
                color=(255, 0, 0),
            )

    def _load_folder(
        self,
        path: Path,
        progress_callback: Optional[Callable[[int], None]] = None,
        message_callback: Optional[Callable[[str], None]] = None,
    ) -> ImageData:
        if progress_callback is not None:
            progress_callback(0)
        content = [f for f in path.iterdir() if not f.is_dir()]
        content = natsorted(content)
        suffixes = {s: len([f for f in content if f.suffix == s])
                    for s in set(f.suffix for f in content)}
        most_frequent_suffix = max(suffixes, key=suffixes.get)  # type: ignore
        if len(suffixes) > 1:
            log("Warning: folder contains multiple file types; "
                f"Loading all {most_frequent_suffix!r} files")
            content = [f for f in content if f.suffix == most_frequent_suffix]
        if self.crop is not None:
            content = content[self.crop[0]:self.crop[1]+1]

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
        elif DataLoader._is_dicom(content[0]):
            sample, _ = self._load_dicom_file(content[0])
            total_size_estimate = sample.nbytes * len(content)
            load_function = self._load_dicom_file
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

        n_content = len(content)
        use_multicore = (
            len(content) > settings.get("default/multicore_files_threshold")
            or total_size_estimate
            > settings.get("default/multicore_size_threshold")
        )
        if use_multicore:
            if message_callback is not None:
                message_callback("Loading in parallel (progress not available)...")
            with Pool(cpu_count()) as pool:
                results = pool.starmap(
                    load_function,
                    [(f, ) for f in content],
                )
            matrices, metadata = zip(*results)
            if progress_callback is not None:
                progress_callback(100)
        else:
            matrices, metadata = [], []
            for i, f in enumerate(content):
                matrix, meta = load_function(f)
                matrices.append(matrix)
                metadata.append(meta)
                if progress_callback is not None and n_content > 0:
                    progress_callback(int(100 * (i + 1) / n_content))

        try:
            matrices = np.stack(matrices)
        except Exception:
            log(
                "Error loading files: shapes of images do not match",
                color="red",
            )
            return DataLoader.from_text(
                "Error loading files",
                color=(255, 0, 0),
            )
        done = ImageData(matrices, metadata)

        if isinstance(done.meta[0], DicomMetaData):
            sorted_indices = np.argsort(
                [meta.sequence_number for meta in done.meta]
            )
            done = ImageData(
                done.image[sorted_indices],
                [done.meta[i] for i in sorted_indices]
            )

        self._log_arguments(done)
        return done

    def _load_image(self, path: Path) -> tuple[np.ndarray, MetaData]:
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
        if self.mask is not None:
            image = image[self.mask]
        metadata = MetaData(
            file_name=path.name,
            file_size_MB=os.path.getsize(path) / 2**20,
            size=(image.shape[0], image.shape[1]),
            dtype=image.dtype,
            bit_depth=8*image.dtype.itemsize,
            color_model= "grayscale" if image.ndim == 2 else "rgb",
        )
        return np.swapaxes(image, 0, 1), metadata

    def _load_dicom_file(self, path: Path) -> tuple[np.ndarray, MetaData]:
        dicom_data = pydicom.dcmread(path)
        image = dicom_data.pixel_array
        image = np.swapaxes(image, 0, 1)
        metadata = DicomMetaData(
            file_name=path.name,
            file_size_MB=os.path.getsize(path) / 2**20,
            size=(image.shape[0], image.shape[1]),
            dtype=image.dtype,
            bit_depth=8 * image.dtype.itemsize,
            color_model="grayscale",
            sequence_number=int(dicom_data.InstanceNumber),
        )
        return image, metadata

    def _load_video(
        self,
        path: Path,
        frame_range: Optional[tuple[int, int]] = None,
        step: Optional[int] = None,
        size_ratio: Optional[float] = None,
        grayscale: Optional[bool] = None,
        progress_callback: Optional[Callable[[int], None]] = None,
        **_kwargs,
    ) -> ImageData:
        # Override defaults if provided
        _size_ratio = size_ratio if size_ratio is not None else self.size_ratio
        _grayscale = grayscale if grayscale is not None else self.grayscale

        # Determine subset ratio logic. If 'step' is provided, it overrides subset_ratio
        if step is not None:
            _subset_ratio = 1.0 / step
        else:
            _subset_ratio = self.subset_ratio

        cap = cv2.VideoCapture(str(path))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        codec = int(cap.get(cv2.CAP_PROP_FOURCC))
        fourcc_str = "".join([chr((codec >> 8 * i) & 0xFF) for i in range(4)])
        cap.release()

        # If manual frame_range not provided, use self.crop
        if frame_range is None:
            if self.crop is not None:
                start_f = self.crop[0]
                end_f = self.crop[1]
            else:
                start_f = 0
                end_f = frame_count - 1
        else:
            start_f = frame_range[0]
            end_f = frame_range[1]

        # Ensure bounds
        start_f = max(0, start_f)
        end_f = min(frame_count - 1, end_f)
        total_frames_target = max(0, end_f - start_f + 1)

        # If parameters were not overridden by user dialog, perform auto-adjust check
        if step is None and size_ratio is None:
            # ImageData keeps uint8 by default; account for resize and grayscale
            resized_h = int(height * _size_ratio)
            resized_w = int(width * _size_ratio)
            channels = 1 if _grayscale else 3
            memory_estimate = resized_h * resized_w * channels * total_frames_target
            log(f"Estimated size to load: {memory_estimate / 2**20:.2f} MB")
            adjusted_ratio = adjust_ratio_for_memory(
                memory_estimate,
                self.max_ram,
            )
            ratio = min(_subset_ratio, adjusted_ratio)
            if ratio == _subset_ratio:
                log("No adjustment to ratio required")
            else:
                _subset_ratio = ratio
                log(f"Adjusted ratio for subset extraction: {ratio:.4f}",
                    color="orange")

        skip_frames = int(1 / _subset_ratio) - 1
        step_val = skip_frames + 1

        video = cv2.VideoCapture(str(path))
        frames = None
        metadata = []

        # Seek logic
        # If start_f is large, seeking is better than reading
        if start_f > 0:
            video.set(cv2.CAP_PROP_POS_FRAMES, start_f)

        frame_number = start_f # Tracks the source video frame number

        # Expected frames based on step
        expected_frames = int(np.ceil(total_frames_target / step_val))
        
        current_idx = 0
        while frame_number <= end_f:
            ret, frame = video.read()
            if not ret:
                break

            image = resize_and_convert(
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    if not _grayscale else
                    cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                _size_ratio,
                self.convert_to_8_bit,
            )
            if self.mask is not None:
                image = image[self.mask]

            if frames is None:
                allocate_size = expected_frames if expected_frames > 0 else 100
                frames = np.empty((allocate_size, *image.shape), dtype=image.dtype)

            if current_idx >= len(frames):
                new_size = int(len(frames) * 1.5)
                frames.resize((new_size, *image.shape), refcheck=False)

            frames[current_idx] = image

            metadata.append(VideoMetaData(
                file_name=str(frame_number),
                file_size_MB=os.path.getsize(path) / 2**20,
                size=(frame.shape[0], frame.shape[1]),
                dtype=frame.dtype,  # type: ignore
                bit_depth=8*frame.dtype.itemsize,
                color_model="rgb" if len(frame.shape) == 3 else "grayscale",
                fps=fps,
                frame_count=frame_count,
                reduced_frame_count=current_idx + 1,
                codec=fourcc_str,
            ))
            current_idx += 1
            frame_number += 1

            # Progress reporting
            if progress_callback and expected_frames > 0 and current_idx % 10 == 0:
                progress = int((current_idx / expected_frames) * 100)
                progress_callback(min(100, progress))

            # Skip frames
            skipped_successfully = True
            for _ in range(skip_frames):
                if not video.grab():
                    skipped_successfully = False
                    break
                frame_number += 1

            if not skipped_successfully:
                break

        video.release()
        
        if frames is None:
            return DataLoader.from_text("No frames loaded", color=(255, 0, 0))

        if current_idx < len(frames):
            frames = frames[:current_idx]

        done = ImageData(np.swapaxes(frames, 1, 2), metadata)
        self._log_arguments(done)
        return done

    def _load_array(self, path: Path) -> ImageData:
        array: np.ndarray = np.load(path)
        gray = True
        match array.ndim:
            case 4:
                if self.grayscale:
                    weights = np.array([0.2989, 0.5870, 0.1140])
                    array = np.sum(array * weights, axis=-1)
                else:
                    gray = False
            case 3:
                if not self.grayscale and array.shape[2] == 3:
                    array = array[np.newaxis, ...]
                    gray = False
            case 2:
                array = array[np.newaxis, ...]
            case _:
                log(f"Error: Unsupported array shape {array.shape}",
                    color="red")
                return DataLoader.from_text(
                    "Error loading files", color=(255, 0, 0)
                )
        # now the first dimension is time
        def function_(x):
            return resize_and_convert_to_8_bit(
                x,
                self.size_ratio,
                self.convert_to_8_bit,
            )

        total_size_estimate = array[0].nbytes * array.shape[0]
        if (array.shape[0] > settings.get("default/multicore_files_threshold")
                or total_size_estimate >
                    settings.get("default/multicore_size_threshold")):
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

        array = np.stack(matrices)
        if self.mask is not None:
            array = array[:, *self.mask]
        if self.crop is not None:
            array = array[self.crop[0]:self.crop[1]+1]
        array = np.swapaxes(array, 1, 2)
        metadata = [MetaData(
            file_name=path.name + f"-{i}",
            file_size_MB=os.path.getsize(path)/2**20,
            size=array[i].shape[:2],
            dtype=array.dtype,
            color_model="grayscale" if gray else "rgb",
            bit_depth=8*array.dtype.itemsize,
        ) for i in range(array.shape[0])]
        done = ImageData(array, metadata)
        self._log_arguments(done)
        return done

    def _load_single_array(
        self,
        path: Path,
    ) -> tuple[np.ndarray, MetaData]:
        array: np.ndarray = np.load(path)
        if array.ndim >= 4 or (array.ndim == 3 and array.shape[2] != 3):
            log(f"Error: Unsupported array shape {array.shape}", color="red")
            data = DataLoader.from_text(
                "Error loading files",
                color=(255, 0, 0),
            )
            return data.image[0], data.meta[0]

        gray = False
        if array.ndim == 3 and self.grayscale:
            weights = np.array([0.2989, 0.5870, 0.1140])
            array = np.sum(array * weights, axis=-1)
            gray = True

        array = resize_and_convert_to_8_bit(
            array,
            self.size_ratio,
            self.convert_to_8_bit,
        )
        if self.mask is not None:
            array = array[self.mask]
        array = np.swapaxes(array, 0, 1)
        metadata = MetaData(
            file_name=path.name,
            file_size_MB=os.path.getsize(path)/2**20,
            size=array.shape[:2],  # type: ignore
            dtype=array.dtype,
            bit_depth=8*array.dtype.itemsize,
            color_model="grayscale" if gray else "rgb",
        )
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
        metadata = [MetaData(
            file_name="-",
            file_size_MB=img.nbytes/2**20,
            size=(height, width),
            dtype=np.uint8,
            bit_depth=8*img.dtype.itemsize,
            color_model="rgb",
        )]
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

def tof_from_csv(file_path: str) -> np.ndarray:
    try:
        # Load the first two columns, skipping invalid lines
        data = np.genfromtxt(file_path, delimiter=',', usecols=(0, 1), invalid_raise=False)
    except Exception:
        return np.empty((0, 2))

    if data is None or data.size == 0:
        return np.empty((0, 2))

    # Ensure 2D array even if there's only one valid row
    if data.ndim == 1:
        data = data.reshape(1, -1)
        
    # Filter for offset (first column) >= 0
    return data[data[:, 0] >= 0]