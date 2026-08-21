"""Fit a load-dialog crop to content in a preview (threshold + margin)."""

from __future__ import annotations

import numpy as np

# Fraction of the 1st–99th percentile span; dark letterbox stays below this.
DEFAULT_REL_THRESHOLD = 0.05
# Extra pixels as a fraction of each axis, so the cut is not flush to content.
DEFAULT_MARGIN_FRAC = 0.03


def content_bbox_xywh(
    image: np.ndarray,
    *,
    rel_threshold: float = DEFAULT_REL_THRESHOLD,
    margin_frac: float = DEFAULT_MARGIN_FRAC,
) -> tuple[int, int, int, int]:
    """Return ``(x, y, w, h)`` for content in an ``(H, W[, C])`` preview.

    Coordinates match the load-dialog ROI (origin top-left, ``y`` down).
    A uniform or empty frame returns the full image. Never auto-applies a load;
    the dialog only moves the existing crop rectangle.
    """
    img = np.asarray(image)
    if img.ndim < 2:
        raise ValueError(f"Preview must be at least 2D, got shape {img.shape}")
    if img.ndim >= 3:
        lum = np.mean(img.reshape(img.shape[0], img.shape[1], -1), axis=2)
    else:
        lum = img
    lum = np.asarray(lum, dtype=np.float64)
    h, w = int(lum.shape[0]), int(lum.shape[1])
    if h < 1 or w < 1:
        return 0, 0, max(w, 1), max(h, 1)

    finite = lum[np.isfinite(lum)]
    if finite.size == 0:
        return 0, 0, w, h
    lo = float(np.percentile(finite, 1))
    hi = float(np.percentile(finite, 99))
    span = hi - lo
    if not np.isfinite(span) or span <= 0:
        return 0, 0, w, h

    mask = lum >= (lo + float(rel_threshold) * span)
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not np.any(rows) or not np.any(cols):
        return 0, 0, w, h

    y0 = int(np.argmax(rows))
    y1 = int(h - np.argmax(rows[::-1]))
    x0 = int(np.argmax(cols))
    x1 = int(w - np.argmax(cols[::-1]))

    my = int(round(float(margin_frac) * h))
    mx = int(round(float(margin_frac) * w))
    y0 = max(0, y0 - my)
    x0 = max(0, x0 - mx)
    y1 = min(h, y1 + my)
    x1 = min(w, x1 + mx)
    return x0, y0, max(1, x1 - x0), max(1, y1 - y0)
