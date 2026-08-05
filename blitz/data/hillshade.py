"""Hillshade (relief shading) from a height field. Pure NumPy, no Qt."""

from __future__ import annotations

import numpy as np


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
    z = height_from_frame(height)
    if z.ndim != 2:
        raise ValueError(f"Height must be 2D after conversion, got {z.shape}")
    zf = float(z_factor) if np.isfinite(z_factor) and z_factor > 0 else 1.0
    z = np.nan_to_num(z * zf, nan=0.0, posinf=0.0, neginf=0.0)

    # dz/dx along axis 0, dz/dy along axis 1
    dx, dy = np.gradient(z, axis=(0, 1))

    az = np.deg2rad(float(azimuth_deg) % 360.0)
    el = np.deg2rad(float(np.clip(elevation_deg, 0.0, 90.0)))
    # Unit light vector pointing toward the surface from the sun
    lx = np.sin(az) * np.cos(el)
    ly = np.cos(az) * np.cos(el)
    lz = np.sin(el)

    nx = -dx
    ny = -dy
    nz = np.ones_like(z)
    with np.errstate(invalid="ignore"):
        norm = np.sqrt(nx * nx + ny * ny + nz * nz)
        shade = (nx * lx + ny * ly + nz * lz) / np.maximum(norm, 1e-12)
    return np.clip(shade, 0.0, 1.0).astype(np.float32)
