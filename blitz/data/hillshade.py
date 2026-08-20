"""Hillshade (relief shading) from a height field. Pure NumPy, no Qt."""

from __future__ import annotations

import numpy as np

AZIMUTH_CACHE_STEP_DEG = 30


def height_from_frame(frame: np.ndarray) -> np.ndarray:
    """2D float height from a display frame (greyscale or RGB)."""
    arr = np.asarray(frame)
    if arr.ndim == 3:
        # Luminance of last axis as channels
        if arr.shape[-1] in (3, 4):
            rgb = arr[..., :3].astype(np.float64)
            arr = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
        else:
            arr = np.nanmean(arr, axis=-1)
    elif arr.ndim != 2:
        raise ValueError(f"Expected 2D or RGB frame, got shape {arr.shape}")
    return arr.astype(np.float64, copy=False)


def snap_azimuth_deg(
    azimuth_deg: float,
    step_deg: float = AZIMUTH_CACHE_STEP_DEG,
) -> int:
    """Snap azimuth to the cache raster with half-up rounding (315° → 330°)."""
    step = int(step_deg)
    if step <= 0:
        step = AZIMUTH_CACHE_STEP_DEG
    az = float(azimuth_deg) % 360.0
    snapped = int((az + step / 2.0) // step) * step
    return int(snapped % 360)


def azimuth_cache_bins(step_deg: float = AZIMUTH_CACHE_STEP_DEG) -> list[int]:
    """Azimuth raster in ``[0, 360)``, e.g. 0, 30, …, 330."""
    step = int(step_deg)
    if step <= 0:
        step = AZIMUTH_CACHE_STEP_DEG
    return list(range(0, 360, step))


def azimuth_cache_order(
    start_deg: float,
    step_deg: float = AZIMUTH_CACHE_STEP_DEG,
) -> list[int]:
    """Bins rotated so the snapped start azimuth is computed first."""
    bins = azimuth_cache_bins(step_deg)
    start = snap_azimuth_deg(start_deg, step_deg)
    try:
        i = bins.index(start)
    except ValueError:
        return bins
    return bins[i:] + bins[:i]


def scaled_height_gradients(
    height: np.ndarray,
    z_factor: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """``dx, dy = ∇(z · z_factor)`` after converting a display frame to height."""
    z = height_from_frame(height)
    if z.ndim != 2:
        raise ValueError(f"Height must be 2D after conversion, got {z.shape}")
    zf = float(z_factor) if np.isfinite(z_factor) and z_factor > 0 else 1.0
    z = np.nan_to_num(z * zf, nan=0.0, posinf=0.0, neginf=0.0)
    dx, dy = np.gradient(z, axis=(0, 1))
    return dx, dy


def shade_from_gradients(
    dx: np.ndarray,
    dy: np.ndarray,
    azimuth_deg: float,
    elevation_deg: float,
) -> np.ndarray:
    """Lambertian ``n·l`` from precomputed gradients → float32 in ``[0, 1]``."""
    az = np.deg2rad(float(azimuth_deg) % 360.0)
    el = np.deg2rad(float(np.clip(elevation_deg, 0.0, 90.0)))
    # Unit light vector pointing toward the surface from the sun
    lx = np.sin(az) * np.cos(el)
    ly = np.cos(az) * np.cos(el)
    lz = np.sin(el)

    nx = -dx
    ny = -dy
    nz = np.ones_like(dx)
    with np.errstate(invalid="ignore"):
        norm = np.sqrt(nx * nx + ny * ny + nz * nz)
        shade = (nx * lx + ny * ly + nz * lz) / np.maximum(norm, 1e-12)
    return np.clip(shade, 0.0, 1.0).astype(np.float32)


def shade_to_uint8(shade: np.ndarray) -> np.ndarray:
    """Quantize ``[0, 1]`` shade to uint8 for the azimuth atlas."""
    return np.clip(np.round(np.asarray(shade) * 255.0), 0, 255).astype(np.uint8)


def calculate_hillshade(
    height: np.ndarray,
    azimuth_deg: float,
    elevation_deg: float,
    z_factor: float = 1.0,
) -> np.ndarray:
    """Compute hillshade in [0, 1] from a 2D height field.

    Coordinate convention (BLITZ image axes: axis0 = x / right, axis1 = y / up):
    - Azimuth 0° = light from top of image (+y), clockwise toward +x (right).
    - Elevation 0° = horizon, 90° = overhead.
    - ``z_factor`` exaggerates vertical scale before gradients.
    """
    dx, dy = scaled_height_gradients(height, z_factor)
    return shade_from_gradients(dx, dy, azimuth_deg, elevation_deg)
