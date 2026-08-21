"""Hillshade (relief shading) from a height field. Pure NumPy, no Qt."""

from __future__ import annotations

from typing import NamedTuple, Sequence

import numpy as np

AZIMUTH_CACHE_STEP_DEG = 30
AZIMUTH_CACHE_STEP_MIN_DEG = 5
AZIMUTH_CACHE_STEP_MAX_DEG = 90
VIEWPORT_MAX_EDGE = 1600
VIEWPORT_HALO_PX = 1
COMBINED_AZIMUTH_OFFSETS = (0.0, 90.0, 180.0, 270.0)
FOUR_WAY_PRESET_COLORS: tuple[tuple[float, float, float], ...] = (
    (1.00, 0.72, 0.28),
    (1.00, 0.92, 0.55),
    (0.45, 0.78, 1.00),
    (0.55, 0.90, 0.62),
)
Z_FACTOR_MIN = 0.01
Z_FACTOR_MAX = 2.0


class ShadeLight(NamedTuple):
    """One artificial sun. Color is RGB in ``[0, 1]``."""

    azimuth: float
    elevation: float
    color: tuple[float, float, float] = (1.0, 1.0, 1.0)


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


def clamp_azimuth_cache_step(step_deg: float) -> int:
    """Clamp to ``[5, 90]`` and the nearest step that divides 360° (no 1° atlas)."""
    try:
        raw = int(round(float(step_deg)))
    except (TypeError, ValueError):
        return AZIMUTH_CACHE_STEP_DEG
    lo = AZIMUTH_CACHE_STEP_MIN_DEG
    hi = AZIMUTH_CACHE_STEP_MAX_DEG
    step = min(hi, max(lo, raw))
    if 360 % step == 0:
        return step
    candidates = [d for d in range(lo, hi + 1) if 360 % d == 0]
    return min(candidates, key=lambda d: (abs(d - raw), d))


def snap_azimuth_deg(
    azimuth_deg: float,
    step_deg: float = AZIMUTH_CACHE_STEP_DEG,
) -> int:
    """Snap azimuth to the cache raster with half-up rounding (315° → 330° at 30°)."""
    step = clamp_azimuth_cache_step(step_deg)
    az = float(azimuth_deg) % 360.0
    snapped = int((az + step / 2.0) // step) * step
    return int(snapped % 360)


def step_azimuth_deg(
    azimuth_deg: float,
    step_deg: float,
    steps: int,
) -> int:
    """Move ``steps`` bins on the cache raster (wraps 0° ↔ last bin, no nearest-snap)."""
    step = clamp_azimuth_cache_step(step_deg)
    cur = snap_azimuth_deg(azimuth_deg, step)
    return int((cur + int(steps) * step) % 360)


def azimuth_cache_bins(step_deg: float = AZIMUTH_CACHE_STEP_DEG) -> list[int]:
    """Azimuth raster in ``[0, 360)``, e.g. 0, 30, …, 330."""
    step = clamp_azimuth_cache_step(step_deg)
    return list(range(0, 360, step))


def azimuth_atlas_nbytes(
    height: int,
    width: int,
    step_deg: float = AZIMUTH_CACHE_STEP_DEG,
    channels: int = 1,
) -> int:
    """uint8 atlas size in bytes (one copy per azimuth bin)."""
    h, w = int(height), int(width)
    ch = max(1, int(channels))
    if h <= 0 or w <= 0:
        return 0
    return len(azimuth_cache_bins(step_deg)) * h * w * ch


def azimuth_atlas_peak_nbytes(
    height: int,
    width: int,
    step_deg: float = AZIMUTH_CACHE_STEP_DEG,
    channels: int = 1,
) -> int:
    """Peak while filling: atlas + float64 height/gradients + one float32 shade."""
    hw = max(0, int(height)) * max(0, int(width))
    ch = max(1, int(channels))
    workspace = hw * (8 + 8 + 8 + 4 * ch)  # z, dx, dy, shade
    return azimuth_atlas_nbytes(height, width, step_deg, channels=ch) + workspace


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
    # Unit light vector pointing toward the surface from the sun.
    # ImageViewer ViewBox is invertY: screen north / top = decreasing y, so
    # azimuth 0° (north) uses -cos(az) like gdaldem. +x / east is unflipped.
    lx = np.sin(az) * np.cos(el)
    ly = -np.cos(az) * np.cos(el)
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


def combined_azimuths(azimuth_deg: float) -> tuple[float, float, float, float]:
    """Four lights on a circle: ``azimuth`` plus 90° steps."""
    base = float(azimuth_deg) % 360.0
    return tuple((base + off) % 360.0 for off in COMBINED_AZIMUTH_OFFSETS)


def shadow_azimuth_deg(sun_azimuth_deg: float) -> float:
    """Direction a vertical peg's shadow falls: opposite the sun.

    0° is north / top of the image, clockwise (same as hillshade azimuth).
    Sun from the north (0°, top of image) → shadow toward the south (180°, bottom).
    """
    return (float(sun_azimuth_deg) + 180.0) % 360.0


def four_way_lights(
    azimuth_deg: float,
    elevation_deg: float,
) -> tuple[ShadeLight, ...]:
    """Preset: four lights 90° apart, same elevation, distinct colors."""
    el = float(np.clip(elevation_deg, 0.0, 90.0))
    return tuple(
        ShadeLight(az, el, col)
        for az, col in zip(combined_azimuths(azimuth_deg), FOUR_WAY_PRESET_COLORS)
    )


def rotate_lights_to_primary(
    lights: Sequence[ShadeLight],
    primary_azimuth_deg: float,
) -> list[ShadeLight]:
    """Keep relative azimuths / elevations / colors; put light 0 at ``primary``."""
    items = list(lights)
    if not items:
        return [ShadeLight(float(primary_azimuth_deg) % 360.0, 45.0)]
    delta = (float(primary_azimuth_deg) - float(items[0].azimuth)) % 360.0
    return [
        ShadeLight((float(L.azimuth) + delta) % 360.0, float(L.elevation), L.color)
        for L in items
    ]


def shade_rgb_from_gradients(
    dx: np.ndarray,
    dy: np.ndarray,
    lights: Sequence[ShadeLight],
) -> np.ndarray:
    """Weighted mean of coloured Lambertian lights → float32 RGB ``[0, 1]``."""
    acc = None
    n = 0
    for light in lights:
        part = shade_from_gradients(dx, dy, light.azimuth, light.elevation)
        rgb = np.empty(part.shape + (3,), dtype=np.float32)
        r, g, b = light.color
        rgb[..., 0] = part * float(r)
        rgb[..., 1] = part * float(g)
        rgb[..., 2] = part * float(b)
        acc = rgb if acc is None else acc + rgb
        n += 1
    if acc is None:
        return np.zeros(np.asarray(dx).shape + (3,), dtype=np.float32)
    return np.clip(acc / float(max(n, 1)), 0.0, 1.0).astype(np.float32)


def shade_rgb_to_uint8(shade: np.ndarray) -> np.ndarray:
    """Quantize shade (grey or RGB) to uint8."""
    arr = np.asarray(shade)
    return np.clip(np.round(arr * 255.0), 0, 255).astype(np.uint8)


def shade_from_gradients_combined(
    dx: np.ndarray,
    dy: np.ndarray,
    azimuth_deg: float,
    elevation_deg: float,
) -> np.ndarray:
    """Greyscale mean of the four-way preset (white lights)."""
    lights = tuple(
        ShadeLight(az, elevation_deg, (1.0, 1.0, 1.0))
        for az in combined_azimuths(azimuth_deg)
    )
    rgb = shade_rgb_from_gradients(dx, dy, lights)
    return rgb[..., 0]


def viewport_slices(
    shape_hw: tuple[int, ...],
    x0: float,
    x1: float,
    y0: float,
    y1: float,
    *,
    axis_order: str = "col-major",
    halo: int = VIEWPORT_HALO_PX,
) -> tuple[tuple[slice, slice], tuple[float, float, float, float]]:
    """Crop slices and data-space rect ``(x, y, w, h)`` for a ViewBox window.

    ``axis_order`` matches pyqtgraph ImageItem: col-major is ``[x, y]``.
    """
    spatial = tuple(int(s) for s in shape_hw[:2])
    if str(axis_order) == "row-major":
        ny, nx = spatial
    else:
        nx, ny = spatial
    xa, xb = float(min(x0, x1)), float(max(x0, x1))
    ya, yb = float(min(y0, y1)), float(max(y0, y1))
    ix0 = max(0, int(np.floor(xa)) - int(halo))
    ix1 = min(nx, int(np.ceil(xb)) + int(halo))
    iy0 = max(0, int(np.floor(ya)) - int(halo))
    iy1 = min(ny, int(np.ceil(yb)) + int(halo))
    if ix1 <= ix0:
        ix1 = min(nx, ix0 + 1)
    if iy1 <= iy0:
        iy1 = min(ny, iy0 + 1)
    rect = (float(ix0), float(iy0), float(ix1 - ix0), float(iy1 - iy0))
    if str(axis_order) == "row-major":
        return (slice(iy0, iy1), slice(ix0, ix1)), rect
    return (slice(ix0, ix1), slice(iy0, iy1)), rect


def downsample_xy(arr: np.ndarray, max_edge: int = VIEWPORT_MAX_EDGE) -> np.ndarray:
    """Integer-stride downsample so ``max(spatial) <= max_edge``."""
    z = np.asarray(arr)
    if z.ndim < 2:
        return z
    cap = int(max_edge)
    if cap < 8:
        cap = VIEWPORT_MAX_EDGE
    a0, a1 = int(z.shape[0]), int(z.shape[1])
    m = max(a0, a1)
    if m <= cap:
        return z
    sx = max(1, int(np.ceil(a0 / cap)))
    sy = max(1, int(np.ceil(a1 / cap)))
    if z.ndim == 2:
        return z[::sx, ::sy]
    return z[::sx, ::sy, ...]


def extract_viewport_patch(
    frame: np.ndarray,
    x0: float,
    x1: float,
    y0: float,
    y1: float,
    *,
    axis_order: str = "col-major",
    max_edge: int = VIEWPORT_MAX_EDGE,
    halo: int = VIEWPORT_HALO_PX,
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """Visible crop plus halo, downsampled for overlay paint."""
    sl, rect = viewport_slices(
        np.asarray(frame).shape,
        x0,
        x1,
        y0,
        y1,
        axis_order=axis_order,
        halo=halo,
    )
    patch = np.asarray(frame)[sl]
    return downsample_xy(patch, max_edge), rect


def calculate_hillshade(
    height: np.ndarray,
    azimuth_deg: float,
    elevation_deg: float,
    z_factor: float = 1.0,
    *,
    combined: bool = False,
    lights: Sequence[ShadeLight] | None = None,
) -> np.ndarray:
    """Compute hillshade from a 2D height field.

    Greyscale ``[0, 1]`` for a single white light; RGB ``[0, 1]`` when ``lights``
    is set (or when ``combined`` uses the four-way colour preset).

    Coordinate convention (col-major frame ``[x, y]``, ImageViewer ``invertY``):
    - Azimuth 0° = light from top of the image (screen north, decreasing y).
    - Clockwise toward +x (90° = from the right / east).
    - Elevation 0° = horizon, 90° = overhead.
    - ``z_factor`` exaggerates vertical scale before gradients.
    """
    dx, dy = scaled_height_gradients(height, z_factor)
    if lights:
        return shade_rgb_from_gradients(dx, dy, lights)
    if combined:
        return shade_rgb_from_gradients(
            dx, dy, four_way_lights(azimuth_deg, elevation_deg)
        )
    return shade_from_gradients(dx, dy, azimuth_deg, elevation_deg)
