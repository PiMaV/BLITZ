"""Unit tests for hillshade (numpy only)."""

from __future__ import annotations

import numpy as np

from blitz.data.hillshade import (
    azimuth_cache_bins,
    azimuth_cache_order,
    calculate_hillshade,
    height_from_frame,
    scaled_height_gradients,
    shade_from_gradients,
    shade_to_uint8,
    snap_azimuth_deg,
)


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


def test_shade_from_gradients_matches_calculate_hillshade() -> None:
    rng = np.random.default_rng(0)
    z = rng.random((24, 32))
    dx, dy = scaled_height_gradients(z, z_factor=2.5)
    split = shade_from_gradients(dx, dy, azimuth_deg=210.0, elevation_deg=40.0)
    wrapped = calculate_hillshade(z, 210.0, 40.0, z_factor=2.5)
    assert split.dtype == np.float32
    assert wrapped.dtype == np.float32
    np.testing.assert_allclose(split, wrapped, rtol=0, atol=0)


def test_snap_azimuth_half_up_not_bankers() -> None:
    assert snap_azimuth_deg(315.0) == 330
    assert snap_azimuth_deg(360.0) == 0
    assert snap_azimuth_deg(0.0) == 0
    assert snap_azimuth_deg(14.9) == 0
    assert snap_azimuth_deg(15.0) == 30
    assert snap_azimuth_deg(345.0) == 0
    assert snap_azimuth_deg(329.9) == 330


def test_azimuth_cache_bins_and_order() -> None:
    bins = azimuth_cache_bins()
    assert bins == [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]
    assert len(bins) == 12
    order = azimuth_cache_order(315.0)
    assert order[0] == 330
    assert order == [330, 0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300]
    assert set(order) == set(bins)


def test_clamp_azimuth_cache_step_rejects_one_degree() -> None:
    from blitz.data.hillshade import (
        AZIMUTH_CACHE_STEP_MIN_DEG,
        clamp_azimuth_cache_step,
    )

    assert clamp_azimuth_cache_step(1) == AZIMUTH_CACHE_STEP_MIN_DEG
    assert clamp_azimuth_cache_step(0) == 5
    assert clamp_azimuth_cache_step(5) == 5
    assert clamp_azimuth_cache_step(10) == 10
    assert clamp_azimuth_cache_step(7) in (6, 8)
    assert 360 % clamp_azimuth_cache_step(25) == 0
    assert clamp_azimuth_cache_step(30) == 30
    assert clamp_azimuth_cache_step(180) == 90


def test_azimuth_cache_bins_ten_degree() -> None:
    bins = azimuth_cache_bins(10)
    assert bins[0] == 0
    assert bins[-1] == 350
    assert len(bins) == 36
    assert snap_azimuth_deg(315.0, 10) == 320
    order = azimuth_cache_order(315.0, 10)
    assert order[0] == 320


def test_azimuth_atlas_nbytes() -> None:
    from blitz.data.hillshade import (
        azimuth_atlas_nbytes,
        azimuth_atlas_peak_nbytes,
    )

    atlas = azimuth_atlas_nbytes(10, 20, 30)
    assert atlas == 12 * 10 * 20
    peak = azimuth_atlas_peak_nbytes(10, 20, 30)
    assert peak > atlas
    assert azimuth_atlas_nbytes(0, 10, 30) == 0
