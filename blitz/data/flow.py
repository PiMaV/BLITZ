"""D8 flow accumulation from a height field. Pure NumPy (+ optional Numba)."""

from __future__ import annotations

import sys

import numpy as np

from .hillshade import height_from_frame

try:
    from numba import jit

    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

    def jit(*_args, **_kwargs):  # type: ignore[misc]
        def decorator(func):
            return func

        return decorator


_JIT_CACHE = not getattr(sys, "frozen", False)
_SQRT2 = float(np.sqrt(2.0))
# (dx, dy, distance) in col-major [x, y]; cardinals first so they win ties.
_NEIGHBOURS: tuple[tuple[int, int, float], ...] = (
    (1, 0, 1.0),
    (0, 1, 1.0),
    (-1, 0, 1.0),
    (0, -1, 1.0),
    (1, 1, _SQRT2),
    (-1, 1, _SQRT2),
    (-1, -1, _SQRT2),
    (1, -1, _SQRT2),
)


@jit(nopython=True, cache=_JIT_CACHE)
def _accumulate_jit(acc: np.ndarray, order: np.ndarray, down: np.ndarray) -> None:
    for i in range(order.size):
        src = order[i]
        dst = down[src]
        if dst >= 0:
            acc[dst] += acc[src]


def _accumulate_py(acc: np.ndarray, order: np.ndarray, down: np.ndarray) -> None:
    for src in order:
        dst = int(down[src])
        if dst >= 0:
            acc[dst] += acc[src]


def d8_downslope_flat(height: np.ndarray) -> np.ndarray:
    """Raveled index of the steepest downslope neighbour, or ``-1`` at sinks."""
    z = np.asarray(height, dtype=np.float64)
    if z.ndim != 2:
        raise ValueError(f"Expected 2D height, got shape {z.shape}")
    nx, ny = int(z.shape[0]), int(z.shape[1])
    pad = np.full((nx + 2, ny + 2), np.inf, dtype=np.float64)
    pad[1:-1, 1:-1] = np.nan_to_num(z, nan=np.inf, posinf=np.inf, neginf=np.inf)
    best_drop = np.zeros((nx, ny), dtype=np.float64)
    best_dx = np.zeros((nx, ny), dtype=np.int32)
    best_dy = np.zeros((nx, ny), dtype=np.int32)
    has = np.zeros((nx, ny), dtype=bool)
    for dx, dy, dist in _NEIGHBOURS:
        neigh = pad[1 + dx : 1 + dx + nx, 1 + dy : 1 + dy + ny]
        drop = (z - neigh) / dist
        better = drop > best_drop
        best_drop = np.where(better, drop, best_drop)
        best_dx = np.where(better, dx, best_dx)
        best_dy = np.where(better, dy, best_dy)
        has |= better
    xs = np.arange(nx, dtype=np.int32)[:, None]
    ys = np.arange(ny, dtype=np.int32)[None, :]
    down = ((xs + best_dx) * ny + (ys + best_dy)).astype(np.int32)
    return np.where(has, down, np.int32(-1))


def d8_accumulation(height: np.ndarray) -> np.ndarray:
    """Upstream cell count (each cell starts at 1). Sinks keep their total."""
    z = height_from_frame(height)
    down = d8_downslope_flat(z)
    n = int(z.size)
    acc = np.ones(n, dtype=np.float64)
    order = np.argsort(-z.ravel(), kind="mergesort").astype(np.int64, copy=False)
    down_flat = down.ravel().astype(np.int64, copy=False)
    if HAS_NUMBA:
        try:
            _accumulate_jit(acc, order, down_flat)
        except Exception:
            _accumulate_py(acc, order, down_flat)
    else:
        _accumulate_py(acc, order, down_flat)
    return acc.reshape(z.shape)


def accumulation_rgba(acc: np.ndarray, *, log_scale: bool = True) -> np.ndarray:
    """Yellow–green–cyan veins; low accumulation stays fully transparent.

    Percentile cut hides the catchment slab so hillshade or the LUT remain
    readable underneath. High values are bright, not a dark fill.
    """
    a = np.asarray(acc, dtype=np.float64)
    a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    vis = np.log1p(a) if log_scale else a
    finite = vis[np.isfinite(vis)]
    empty = np.zeros(a.shape + (4,), dtype=np.uint8)
    if finite.size == 0:
        return empty
    cut = float(np.percentile(finite, 72.0))
    hi = float(np.percentile(finite, 99.7))
    span = max(hi - cut, 1e-12)
    t = np.clip((vis - cut) / span, 0.0, 1.0).astype(np.float32)
    rgba = empty
    # Land-to-water: gold → lime → cyan → white
    rgba[..., 0] = np.clip(255.0 * (1.0 - 0.85 * t), 0.0, 255.0).astype(np.uint8)
    rgba[..., 1] = np.clip(140.0 + 115.0 * t, 0.0, 255.0).astype(np.uint8)
    rgba[..., 2] = np.clip(40.0 + 215.0 * np.power(t, 0.85), 0.0, 255.0).astype(np.uint8)
    rgba[..., 3] = np.where(
        t < 0.04,
        np.uint8(0),
        np.clip(30.0 + 150.0 * np.power(t, 0.7), 0.0, 175.0).astype(np.uint8),
    )
    return rgba
