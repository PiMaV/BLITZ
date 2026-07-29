"""
HIKMICRO radiometric JPEG → approximate °C float32 arrays.

Vendored for BLITZ in-app use (numpy only). Reference CLI:
  WETTER-Suite/converters/hikmicro_converter.py
Keep behavior aligned when changing trailer offsets / linear map.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
from natsort import natsorted

HDRI_MAGIC = b"HDRI"
HDRI_HEADER_SIZE = 44
HDRI_MAGIC_U16 = 0x1028
TRAILER_TMAX_OFF = 1016
TRAILER_TMIN_OFF = 1020

IMAGE_SUFFIXES = {".jpg", ".jpeg"}


def _is_vis_companion(path: Path) -> bool:
    stem = path.stem.lower()
    return stem.endswith(".vis") or stem.endswith("_vis")


def is_hikmicro_candidate(path: Path) -> bool:
    return (
        path.is_file()
        and path.suffix.lower() in IMAGE_SUFFIXES
        and not _is_vis_companion(path)
    )


def _find_hdri_offset(data: bytes) -> int:
    offset = 0
    while offset < len(data) - 2:
        if data[offset] == 0xFF and data[offset + 1] == 0xD9:
            offset += 2
            if data[offset : offset + 4] == HDRI_MAGIC:
                return offset
        else:
            offset += 1
    hdri_at = data.find(HDRI_MAGIC)
    if hdri_at >= 0:
        return hdri_at
    raise ValueError("No HDRI thermal data block found")


def _trailer_after_pixels(data: bytes, hdri_at: int, width: int, height: int) -> bytes:
    start = hdri_at + HDRI_HEADER_SIZE + width * height * 2
    trailer = data[start:]
    cut = trailer.find(b"RADIOMETRICIMAGE")
    if cut >= 0:
        trailer = trailer[:cut]
    return trailer


def extract_raw_and_trailer_temps(image_path: Path) -> tuple[np.ndarray, float, float]:
    """(raw_uint16 HxW, Tmin_C, Tmax_C) from a radiometric JPEG."""
    path = Path(image_path)
    data = path.read_bytes()
    hdri_at = _find_hdri_offset(data)
    header = data[hdri_at : hdri_at + HDRI_HEADER_SIZE]
    if len(header) < HDRI_HEADER_SIZE or header[:4] != HDRI_MAGIC:
        raise ValueError(f"{path.name}: invalid HDRI header at offset {hdri_at}")

    magic = struct.unpack_from("<H", header, 4)[0]
    if magic != HDRI_MAGIC_U16:
        raise ValueError(
            f"{path.name}: unknown HDRI magic 0x{magic:04x} (expected 0x1028)"
        )

    width = struct.unpack_from("<I", header, 12)[0]
    height = struct.unpack_from("<I", header, 16)[0]
    if width == 0 or height == 0 or width > 4096 or height > 4096:
        raise ValueError(f"{path.name}: implausible HDRI dims {width}x{height}")

    raw_nbytes = width * height * 2
    payload = data[
        hdri_at + HDRI_HEADER_SIZE : hdri_at + HDRI_HEADER_SIZE + raw_nbytes
    ]
    if len(payload) != raw_nbytes:
        raise ValueError(
            f"{path.name}: truncated HDRI payload "
            f"(need {raw_nbytes} bytes, got {len(payload)})"
        )

    raw = np.frombuffer(payload, dtype="<u2").reshape((height, width)).copy()
    trailer = _trailer_after_pixels(data, hdri_at, width, height)
    need = max(TRAILER_TMAX_OFF, TRAILER_TMIN_OFF) + 4
    if len(trailer) < need:
        raise ValueError(
            f"{path.name}: HDRI trailer too short for Tmin/Tmax "
            f"(need {need} bytes, got {len(trailer)})"
        )

    t_max = float(struct.unpack_from("<f", trailer, TRAILER_TMAX_OFF)[0])
    t_min = float(struct.unpack_from("<f", trailer, TRAILER_TMIN_OFF)[0])
    if not np.isfinite(t_min) or not np.isfinite(t_max):
        raise ValueError(f"{path.name}: non-finite trailer Tmin/Tmax")
    if t_max <= t_min:
        raise ValueError(
            f"{path.name}: implausible trailer range Tmin={t_min}, Tmax={t_max}"
        )
    return raw, t_min, t_max


def raw_to_approx_celsius(raw: np.ndarray, t_min: float, t_max: float) -> np.ndarray:
    """Linear map raw DN endpoints → trailer Tmin/Tmax (°C, float32)."""
    r = raw.astype(np.float64)
    r0 = float(r.min())
    r1 = float(r.max())
    if r1 <= r0:
        mid = 0.5 * (t_min + t_max)
        return np.full(raw.shape, mid, dtype=np.float32)
    temp = t_min + (r - r0) / (r1 - r0) * (t_max - t_min)
    return temp.astype(np.float32)


def jpeg_to_celsius(image_path: Path) -> np.ndarray:
    raw, t_min, t_max = extract_raw_and_trailer_temps(image_path)
    return raw_to_approx_celsius(raw, t_min, t_max)


def is_hikmicro_radiometric_jpeg(path: Path) -> bool:
    """Fast probe: candidate suffix and HDRI block present with valid trailer."""
    if not is_hikmicro_candidate(path):
        return False
    try:
        extract_raw_and_trailer_temps(path)
        return True
    except (ValueError, OSError):
        return False


def filter_hikmicro_jpegs(paths: list[Path]) -> list[Path]:
    """Keep paths that probe as radiometric HIKMICRO JPEGs."""
    out = [p for p in paths if is_hikmicro_radiometric_jpeg(p)]
    return natsorted(out)


def load_hikmicro_stack(
    paths: list[Path],
    *,
    progress_callback=None,
) -> tuple[np.ndarray, list[str]]:
    """
    Convert JPEGs → float32 °C stack (N,H,W) and matching file names.
    Skips failures; raises if nothing loaded.
    """
    matrices: list[np.ndarray] = []
    names: list[str] = []
    n = len(paths)
    for i, path in enumerate(paths):
        try:
            arr = jpeg_to_celsius(path)
        except (ValueError, OSError):
            continue
        matrices.append(arr)
        names.append(path.name)
        if progress_callback is not None and n > 0:
            progress_callback(int(100 * (i + 1) / n))
    if not matrices:
        raise ValueError("No HIKMICRO frames could be converted")
    shapes = {m.shape for m in matrices}
    if len(shapes) != 1:
        # Crop to common min so stack still works
        mh = min(m.shape[0] for m in matrices)
        mw = min(m.shape[1] for m in matrices)
        matrices = [m[:mh, :mw] for m in matrices]
    stack = np.stack(matrices, axis=0)
    return stack, names


def celsius_preview(path: Path, size_ratio: float = 0.3) -> np.ndarray | None:
    """uint8 preview of approx °C map for dialogs."""
    try:
        arr = jpeg_to_celsius(path)
    except (ValueError, OSError):
        return None
    lo, hi = float(np.nanmin(arr)), float(np.nanmax(arr))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return None
    norm = ((arr - lo) / (hi - lo) * 255.0).clip(0, 255).astype(np.uint8)
    if size_ratio < 1.0:
        import cv2

        h, w = norm.shape[:2]
        nh, nw = max(1, int(h * size_ratio)), max(1, int(w * size_ratio))
        norm = cv2.resize(norm, (nw, nh), interpolation=cv2.INTER_AREA)
    return norm
