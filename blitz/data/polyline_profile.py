"""Sample intensity along an open polyline in image coordinates.

Image layout matches BLITZ: frame[x, y] with shape (W, H).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PolylineProfileResult:
    s: np.ndarray  # path length along samples (px)
    intensity: np.ndarray  # mean along perp band (or single pixel)
    xs: np.ndarray  # sample x (float)
    ys: np.ndarray  # sample y (float)
    env_lo: np.ndarray | None = None  # perp-band low
    env_hi: np.ndarray | None = None  # perp-band high
    env_ds_lo: np.ndarray | None = None  # over-frames low
    env_ds_hi: np.ndarray | None = None  # over-frames high


def _as_gray(frame: np.ndarray) -> np.ndarray:
    """(W,H) or (W,H,C) -> float (W,H); RGB uses first channel (Probe-style)."""
    arr = np.asarray(frame)
    if arr.ndim == 3:
        return arr[..., 0].astype(np.float64, copy=False)
    return arr.astype(np.float64, copy=False)


def _segment_samples(
    x0: float, y0: float, x1: float, y1: float, step: float = 1.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Dense samples along a segment; returns xs, ys, ds (segment lengths)."""
    dx = x1 - x0
    dy = y1 - y0
    length = float(np.hypot(dx, dy))
    if length < 1e-9:
        return (
            np.array([x0], dtype=np.float64),
            np.array([y0], dtype=np.float64),
            np.array([0.0], dtype=np.float64),
        )
    n = max(1, int(np.ceil(length / step)))
    t = np.linspace(0.0, 1.0, n + 1, dtype=np.float64)
    xs = x0 + t * dx
    ys = y0 + t * dy
    # per-point arc length increment from previous
    ds = np.empty_like(t)
    ds[0] = 0.0
    if n >= 1:
        ds[1:] = length / n
    return xs, ys, ds


def roi_points_xy(roi) -> np.ndarray:
    """Local ROI handle points → image (x, y) array shape (N, 2)."""
    pts = roi.getLocalHandlePositions()
    if not pts:
        return np.zeros((0, 2), dtype=np.float64)
    out = np.empty((len(pts), 2), dtype=np.float64)
    for i, item in enumerate(pts):
        p = item[1] if isinstance(item, (tuple, list)) else item
        mapped = roi.mapToParent(p)
        out[i, 0] = float(mapped.x())
        out[i, 1] = float(mapped.y())
    return out


def vertex_path_lengths(points_xy: np.ndarray) -> np.ndarray:
    """Cumulative path length (px) at each vertex, starting at 0."""
    if points_xy is None or len(points_xy) == 0:
        return np.zeros(0, dtype=np.float64)
    s = np.zeros(len(points_xy), dtype=np.float64)
    for i in range(1, len(points_xy)):
        s[i] = s[i - 1] + float(
            np.hypot(
                points_xy[i, 0] - points_xy[i - 1, 0],
                points_xy[i, 1] - points_xy[i - 1, 1],
            )
        )
    return s


def sample_polyline_profile(
    frame: np.ndarray,
    points_xy: np.ndarray,
    *,
    width: int = 0,
    envelope_pct: float = 0.0,
    want_perp_envelope: bool = False,
    volume: np.ndarray | None = None,
    want_dataset_envelope: bool = False,
    step: float = 1.0,
) -> PolylineProfileResult | None:
    """Sample along open polyline through ``points_xy`` (N,2) in image x,y.

    ``width`` is half-width in pixels perpendicular to the local tangent.
    ``volume`` is (T,W,H[,C]) when dataset envelope is requested.
    """
    if points_xy is None or len(points_xy) < 2:
        return None
    gray = _as_gray(frame)
    w, h = gray.shape[0], gray.shape[1]

    xs_list: list[np.ndarray] = []
    ys_list: list[np.ndarray] = []
    ds_list: list[np.ndarray] = []
    for i in range(len(points_xy) - 1):
        x0, y0 = points_xy[i]
        x1, y1 = points_xy[i + 1]
        xs, ys, ds = _segment_samples(x0, y0, x1, y1, step=step)
        if i > 0:
            # drop duplicate joint
            xs, ys, ds = xs[1:], ys[1:], ds[1:]
        xs_list.append(xs)
        ys_list.append(ys)
        ds_list.append(ds)

    xs = np.concatenate(xs_list)
    ys = np.concatenate(ys_list)
    ds = np.concatenate(ds_list)
    if xs.size == 0:
        return None
    s = np.cumsum(ds)

    # Tangents for perpendicular sampling
    tx = np.gradient(xs)
    ty = np.gradient(ys)
    norm = np.hypot(tx, ty)
    norm = np.where(norm < 1e-9, 1.0, norm)
    tx /= norm
    ty /= norm
    nx, ny = -ty, tx  # unit normal

    half = max(0, int(width))
    offsets = np.arange(-half, half + 1, dtype=np.float64)
    if offsets.size == 0:
        offsets = np.array([0.0])

    n = xs.size
    band = np.full((n, offsets.size), np.nan, dtype=np.float64)
    for j, off in enumerate(offsets):
        px = np.rint(xs + off * nx).astype(np.int64)
        py = np.rint(ys + off * ny).astype(np.int64)
        valid = (px >= 0) & (px < w) & (py >= 0) & (py < h)
        vals = np.full(n, np.nan, dtype=np.float64)
        vals[valid] = gray[px[valid], py[valid]]
        band[:, j] = vals

    count = np.sum(~np.isnan(band), axis=1)
    intensity = np.full(n, np.nan, dtype=np.float64)
    good = count > 0
    if np.any(good):
        intensity[good] = np.nansum(band[good], axis=1) / count[good]

    env_lo = env_hi = None
    if want_perp_envelope:
        env_lo = np.full(n, np.nan, dtype=np.float64)
        env_hi = np.full(n, np.nan, dtype=np.float64)
        if np.any(good):
            sub = band[good]
            if envelope_pct <= 0:
                env_lo[good] = np.nanmin(sub, axis=1)
                env_hi[good] = np.nanmax(sub, axis=1)
            else:
                env_lo[good] = np.nanpercentile(sub, envelope_pct, axis=1)
                env_hi[good] = np.nanpercentile(sub, 100.0 - envelope_pct, axis=1)

    env_ds_lo = env_ds_hi = None
    if want_dataset_envelope and volume is not None and volume.shape[0] > 0:
        # Sample centerline (and optional perp band) over all frames — heavy.
        # Use centerline pixels only for dataset envelope (speed).
        px = np.clip(np.rint(xs).astype(np.int64), 0, w - 1)
        py = np.clip(np.rint(ys).astype(np.int64), 0, h - 1)
        # volume[t, x, y] or [t,x,y,c]
        if volume.ndim == 4:
            stack = volume[:, px, py, 0].astype(np.float64, copy=False)
        else:
            stack = volume[:, px, py].astype(np.float64, copy=False)
        # stack shape (T, n)
        if envelope_pct <= 0:
            env_ds_lo = np.min(stack, axis=0)
            env_ds_hi = np.max(stack, axis=0)
        else:
            env_ds_lo = np.percentile(stack, envelope_pct, axis=0)
            env_ds_hi = np.percentile(stack, 100.0 - envelope_pct, axis=0)

    return PolylineProfileResult(
        s=s,
        intensity=intensity,
        xs=xs,
        ys=ys,
        env_lo=env_lo,
        env_hi=env_hi,
        env_ds_lo=env_ds_lo,
        env_ds_hi=env_ds_hi,
    )


    return out


def nearest_sample_index(
    xs: np.ndarray, ys: np.ndarray, x: float, y: float
) -> int | None:
    if xs is None or xs.size == 0:
        return None
    d2 = (xs - x) ** 2 + (ys - y) ** 2
    return int(np.argmin(d2))
