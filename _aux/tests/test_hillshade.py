"""Unit tests for hillshade (numpy only)."""

from __future__ import annotations

import numpy as np

from blitz.data.hillshade import calculate_hillshade, height_from_frame


def test_height_from_rgb_luminance() -> None:
    rgb = np.zeros((4, 4, 3), dtype=np.float64)
    rgb[..., 0] = 1.0
    h = height_from_frame(rgb)
    assert h.shape == (4, 4)
    assert np.allclose(h, 0.299)


def test_flat_surface_overhead_is_bright() -> None:
    z = np.ones((32, 32), dtype=np.float64)
    shade = calculate_hillshade(z, azimuth_deg=315.0, elevation_deg=90.0)
    assert shade.shape == z.shape
    assert float(np.nanmin(shade)) > 0.95


def test_ramp_changes_with_azimuth() -> None:
    # Ramp increasing in +x: light from -x (270°) faces the slope → brighter
    x = np.linspace(0, 10, 64)
    z = np.tile(x[:, None], (1, 64))
    left = calculate_hillshade(z, azimuth_deg=270.0, elevation_deg=45.0)
    right = calculate_hillshade(z, azimuth_deg=90.0, elevation_deg=45.0)
    assert float(np.nanmean(left)) > float(np.nanmean(right))


def test_z_factor_deepens_relief_on_bump() -> None:
    yy, xx = np.mgrid[-1:1:48j, -1:1:48j]
    z = np.exp(-3.0 * (xx * xx + yy * yy))
    soft = calculate_hillshade(z, 315.0, 35.0, z_factor=0.5)
    hard = calculate_hillshade(z, 315.0, 35.0, z_factor=8.0)
    assert float(np.nanstd(hard)) > float(np.nanstd(soft))
