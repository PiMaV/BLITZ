"""ASCII Loader: tab/space/comma-separated numeric data (.asc, .dat) -> ImageData."""

import io
from multiprocessing import Pool

from ..._cpu import physical_cpu_count
from pathlib import Path
from typing import Any, Callable, Optional

import cv2
import numpy as np
from natsort import natsorted

from ... import settings
from ..image import ImageData, MetaData
from ...tools import log

ASCII_EXTENSIONS = (".asc", ".dat")
DELIMITERS = {"Tab": "\t", "Space": " ", "Comma": ","}
RAW_PREVIEW_MAX_CHARS = 80
RAW_PREVIEW_MAX_LINES = 5


def get_ascii_files(path: Path) -> list[Path]:
    """List ASCII files in folder, natsorted. If path is file, return [path]."""
    if path.is_file():
        return [path] if path.suffix.lower() in ASCII_EXTENSIONS else []
    content = [f for f in path.iterdir() if f.is_file() and f.suffix.lower() in ASCII_EXTENSIONS]
    if not content:
        return []
    suffixes = {s: len([f for f in content if f.suffix.lower() == s]) for s in ASCII_EXTENSIONS}
    most_frequent = max(suffixes, key=suffixes.get)  # type: ignore
    content = [f for f in content if f.suffix.lower() == most_frequent]
    return natsorted(content)


def estimate_ascii_datatype(
    path: Path,
    delimiter: str,
    first_col_is_row_number: bool,
) -> dict:
    """Estimate value type and stats from first file. Returns dict with dtype, min, max, median, mean."""
    empty = {"dtype": "float", "min": 0.0, "max": 0.0, "median": 0.0, "mean": 0.0}
    files = get_ascii_files(path)
    if not files:
        return empty
    arr = _parse_ascii(files[0], delimiter, first_col_is_row_number)
    if arr is None or arr.size == 0:
        return empty
    flat = np.asarray(arr).ravel()
    valid = flat[~(np.isnan(flat) | np.isinf(flat))]
    if valid.size == 0:
        return {**empty, "min": float("nan"), "max": float("nan"), "median": float("nan"), "mean": float("nan")}
    mn, mx = np.nanmin(flat), np.nanmax(flat)
    med = np.nanmedian(flat)
    mean = np.nanmean(flat)
    if np.any(np.isnan(flat)) or np.any(np.isinf(flat)):
        dtype = "float"
    elif not np.all(np.equal(np.mod(flat, 1), 0)):
        dtype = "float"
    elif mn >= 0:
        dtype = "uint8" if mx <= 255 else ("uint16" if mx <= 65535 else "int")
    else:
        dtype = "int"
    return {"dtype": dtype, "min": float(mn), "max": float(mx), "median": float(med), "mean": float(mean)}


def get_ascii_metadata(path: Path, delimiter: str = "\t") -> dict | None:
    """Get metadata for ASCII file or folder. Returns dict with file_name, size (h,w), file_count, format_display."""
    files = get_ascii_files(path)
    if not files:
        return None
    raw = _parse_ascii(files[0], delimiter, first_col_is_row_number=False)
    if raw is None:
        return None
    first_col = first_col_looks_like_row_number(raw) if raw.shape[1] > 1 else False
    arr = raw[:, 1:] if (first_col and raw.shape[1] > 1) else raw
    h, w = arr.shape[0], arr.shape[1]
    delim_name = next((k for k, v in DELIMITERS.items() if v == delimiter), "Tab")
    est = estimate_ascii_datatype(path, delimiter, first_col)
    # Nur vorschlagen wenn Daten in 8-bit passen (0–255); uint16 mit größerem max → Präzision behalten
    convert_8bit_suggest = est["dtype"] == "uint16" and est["max"] <= 255
    return {
        "file_name": path.name,
        "size": (h, w),
        "file_count": len(files),
        "format_display": f"ASCII, {delim_name}-sep",
        "datatype_est": est["dtype"],
        "convert_to_8_bit_suggest": convert_8bit_suggest,
        "value_stats": {k: est[k] for k in ("min", "max", "median", "mean")},
    }


def parse_ascii_raw(path: Path, delimiter: str) -> np.ndarray | None:
    """Parse ASCII file without stripping first column."""
    return _parse_ascii(path, delimiter, first_col_is_row_number=False)


def _parse_ascii(
    path: Path,
    delimiter: str,
    first_col_is_row_number: bool,
) -> np.ndarray | None:
    """Parse ASCII file to 2D array (rows, cols). Invalid cells (e.g. '...', empty) become NaN."""
    path_str = str(path.resolve() if path.exists() else path)
    data = None
    for encoding in ("utf-8", "latin-1"):
        try:
            with open(path_str, encoding=encoding, errors="replace") as f:
                text = f.read()
            data = np.genfromtxt(
                io.StringIO(text),
                delimiter=delimiter,
                dtype=np.float64,
                invalid_raise=False,
                filling_values=np.nan,
            )
            break
        except Exception:
            continue
    if data is None:
        return None
    if data.size == 0:
        return None
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if first_col_is_row_number and data.shape[1] > 1:
        data = data[:, 1:]
    return data


def first_col_looks_like_row_number(data: np.ndarray) -> bool:
    """True if first column is strictly ascending and matches 0,1,2.. or 1,2,3.."""
    if data.shape[1] < 2 or len(data) < 2:
        return False
    col = data[:, 0].astype(np.float64)
    diffs = np.diff(col)
    if not np.all(diffs > 0):
        return False
    n = len(col)
    return np.allclose(col, np.arange(n)) or np.allclose(col, np.arange(1, n + 1))


def get_ascii_preview(
    path: Path,
    delimiter: str,
    first_col_is_row_number: bool,
    size_ratio: float = 1.0,
    mode: str = "max",
    n_samples: int = 10,
    normalize: bool = True,
) -> np.ndarray | None:
    """Preview from first file or MAX across sampled files (for folder). Returns uint8 (H,W)."""
    files = get_ascii_files(path)
    if not files:
        return None
    if mode == "single" or len(files) == 1:
        arr = _parse_ascii(files[0], delimiter, first_col_is_row_number)
        if arr is None:
            return None
    else:
        ref = _parse_ascii(files[0], delimiter, first_col_is_row_number)
        if ref is None:
            return None
        ref_shape = ref.shape
        indices = np.linspace(0, len(files) - 1, min(n_samples, len(files)), dtype=int)
        arrays = [ref]
        for i in indices[1:]:
            a = _parse_ascii(files[i], delimiter, first_col_is_row_number)
            if a is not None and a.shape == ref_shape:
                arrays.append(a)
        arr = np.max(np.stack(arrays), axis=0)
    out = _array_to_preview_uint8(arr, normalize=normalize)
    if size_ratio < 1.0 and out.size > 0:
        h, w = out.shape[:2]
        new_shape = (max(1, int(h * size_ratio)), max(1, int(w * size_ratio)))
        out = cv2.resize(out.astype(np.float32), (new_shape[1], new_shape[0]), interpolation=cv2.INTER_AREA).astype(np.uint8)
    return out


def _load_single_ascii_file(
    path: Path,
    delimiter: str,
    first_col_is_row_number: bool,
    size_ratio: float,
    mask: Optional[tuple[slice, slice]],
    convert_to_8_bit: bool,
) -> Optional[tuple[np.ndarray, MetaData]]:
    """Load one ASCII file. Returns (matrix, metadata) or None on failure. Top-level for Pool."""
    arr = _parse_ascii(path, delimiter, first_col_is_row_number)
    if arr is None:
        return None
    try:
        if mask is not None:
            arr = arr[mask]
        new_shape = tuple(int(dim * size_ratio) for dim in arr.shape[:2])
        arr = cv2.resize(arr.astype(np.float64), (new_shape[1], new_shape[0]), interpolation=cv2.INTER_AREA)
        if convert_to_8_bit:
            lo, hi = np.nanpercentile(arr, (2, 98))
            if hi > lo:
                arr = np.clip((arr - lo) / (hi - lo) * 255, 0, 255)
                arr = np.nan_to_num(arr, nan=0.0, posinf=255, neginf=0)
                arr = arr.astype(np.uint8)
            else:
                arr = np.zeros_like(arr, dtype=np.uint8)
        h, w = arr.shape[0], arr.shape[1]
        meta = MetaData(
            file_name=path.name,
            file_size_MB=path.stat().st_size / 2**20,
            size=(h, w),
            dtype=arr.dtype,
            bit_depth=8 * arr.dtype.itemsize,
            color_model="grayscale",
        )
        return (np.swapaxes(arr, 0, 1), meta)
    except Exception:
        return None


def _array_to_preview_uint8(arr: np.ndarray, normalize: bool = True) -> np.ndarray:
    """Convert 2D array to 0-255 uint8 for preview display."""
    if arr.size == 0:
        return np.zeros((1, 1), dtype=np.uint8)
    if normalize:
        lo, hi = np.nanpercentile(arr, (2, 98))
    else:
        lo, hi = np.nanmin(arr), np.nanmax(arr)
    if np.isnan(lo) or np.isnan(hi) or hi <= lo:
        return np.zeros_like(arr, dtype=np.uint8)
    normalized = np.clip((arr.astype(np.float64) - lo) / (hi - lo) * 255, 0, 255)
    normalized = np.nan_to_num(normalized, nan=0.0, posinf=255, neginf=0)
    return normalized.astype(np.uint8)


def load_ascii(
    path: Path,
    size_ratio: float = 1.0,
    subset_ratio: float = 1.0,
    mask: Optional[tuple[slice, slice]] = None,
    convert_to_8_bit: bool = False,
    delimiter: str = "\t",
    first_col_is_row_number: bool = True,
    *,
    progress_callback: Optional[Callable[[int], None]] = None,
    message_callback: Optional[Callable[[str], None]] = None,
) -> ImageData:
    """Load ASCII file(s) and return ImageData. Uses multicore for many files."""
    files = get_ascii_files(path)
    if not files:
        return ImageData(
            np.zeros((1, 1, 1, 1), dtype=np.uint8),
            [MetaData(file_name="-", file_size_MB=0, size=(1, 1), dtype=np.uint8, bit_depth=8, color_model="grayscale")],
        )
    step = max(1, int(1.0 / subset_ratio))
    files = files[::step]
    n = len(files)

    sample = _load_single_ascii_file(
        files[0], delimiter, first_col_is_row_number, size_ratio, mask, convert_to_8_bit
    )
    total_size_estimate = 0
    if sample is not None:
        _, sample_meta = sample
        h, w = sample_meta.size
        bytes_per = 1 if convert_to_8_bit else 8
        total_size_estimate = n * h * w * bytes_per
    use_multicore = (
        n > settings.get("default/multicore_files_threshold")
        or total_size_estimate > settings.get("default/multicore_size_threshold")
    )

    args = [(fp, delimiter, first_col_is_row_number, size_ratio, mask, convert_to_8_bit) for fp in files]
    if use_multicore:
        if message_callback:
            message_callback("Loading in parallel (progress not available)...")
        with Pool(physical_cpu_count()) as pool:
            results = pool.starmap(_load_single_ascii_file, args)
        if progress_callback:
            progress_callback(100)
    else:
        results = []
        for i, fp in enumerate(files):
            if message_callback:
                message_callback(f"Loading {fp.name}...")
            results.append(_load_single_ascii_file(fp, delimiter, first_col_is_row_number, size_ratio, mask, convert_to_8_bit))
            if progress_callback and n > 0:
                progress_callback(int(100 * (i + 1) / n))

    failed = [files[i].name for i, r in enumerate(results) if r is None]
    if failed:
        log(f"Files {', '.join(failed)} could not be loaded (corrupt?)", color="orange")
    matrices = []
    metadata_list = []
    for r in results:
        if r is not None:
            matrices.append(r[0])
            metadata_list.append(r[1])
    if not matrices:
        return ImageData(
            np.zeros((1, 1, 1, 1), dtype=np.uint8),
            [MetaData(file_name="-", file_size_MB=0, size=(1, 1), dtype=np.uint8, bit_depth=8, color_model="grayscale")],
        )
    stack = np.stack(matrices)
    return ImageData(stack, metadata_list)
