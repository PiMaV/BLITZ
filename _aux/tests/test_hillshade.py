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


def test_azimuth_zero_lights_screen_north_not_plus_y() -> None:
    """ViewBox invertY: screen top is low y. Azimuth 0° must light from there.

    A ramp z = y faces north (decreasing y). North light is brighter than south.
    East/west must stay unflipped (see test_ramp_changes_with_azimuth).
    """
    ny = 64
    y = np.linspace(0.0, 10.0, ny)
    z = np.broadcast_to(y, (64, ny)).copy()
    north = calculate_hillshade(z, azimuth_deg=0.0, elevation_deg=45.0)
    south = calculate_hillshade(z, azimuth_deg=180.0, elevation_deg=45.0)
    assert float(np.nanmean(north)) > float(np.nanmean(south))


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


def test_step_azimuth_wraps_below_zero_on_cache_raster() -> None:
    from blitz.data.hillshade import step_azimuth_deg

    # Nearest-snap of 355° is 0° — that must not be used for a backward wheel tick.
    assert snap_azimuth_deg(355.0, 30) == 0
    assert step_azimuth_deg(0.0, 30, -1) == 330
    assert step_azimuth_deg(330.0, 30, 1) == 0
    assert step_azimuth_deg(0.0, 30, 1) == 30
    assert step_azimuth_deg(360.0, 5, -1) == 355


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


def test_shade_to_uint8_range() -> None:
    z = np.linspace(0, 1, 16, dtype=np.float32).reshape(4, 4)
    u8 = shade_to_uint8(z)
    assert u8.dtype == np.uint8
    assert int(u8.min()) == 0
    assert int(u8.max()) == 255


def test_combined_azimuths_quarter_turns() -> None:
    from blitz.data.hillshade import combined_azimuths

    assert combined_azimuths(315.0) == (315.0, 45.0, 135.0, 225.0)
    assert combined_azimuths(0.0) == (0.0, 90.0, 180.0, 270.0)


def test_combined_hillshade_white_lights_match_mean() -> None:
    from blitz.data.hillshade import (
        combined_azimuths,
        shade_from_gradients_combined,
    )

    yy, xx = np.mgrid[-1:1:32j, -1:1:32j]
    z = np.exp(-3.0 * (xx * xx + yy * yy))
    combo = shade_from_gradients_combined(
        *scaled_height_gradients(z, 1.0), 315.0, 40.0
    )
    parts = [
        calculate_hillshade(z, az, 40.0, combined=False)
        for az in combined_azimuths(315.0)
    ]
    np.testing.assert_allclose(combo, np.mean(parts, axis=0), rtol=0, atol=1e-6)


def test_combined_colour_preset_is_rgb() -> None:
    yy, xx = np.mgrid[-1:1:24j, -1:1:24j]
    z = np.exp(-3.0 * (xx * xx + yy * yy))
    rgb = calculate_hillshade(z, 315.0, 40.0, combined=True)
    assert rgb.ndim == 3 and rgb.shape[-1] == 3
    assert rgb.dtype == np.float32


def test_rotate_lights_wraps_primary_azimuth() -> None:
    from blitz.data.hillshade import ShadeLight, rotate_lights_to_primary

    lights = [
        ShadeLight(0.0, 35.0, (1.0, 0.0, 0.0)),
        ShadeLight(90.0, 20.0, (0.0, 1.0, 0.0)),
    ]
    rotated = rotate_lights_to_primary(lights, 350.0)
    assert abs(rotated[0].azimuth - 350.0) < 1e-9
    assert abs(rotated[1].azimuth - 80.0) < 1e-9
    assert rotated[1].elevation == 20.0


def test_azimuth_wraps_below_zero() -> None:
    assert float((-5.0) % 360.0) == 355.0
    assert float((0.0 - 5.0) % 360.0) == 355.0


def test_shadow_falls_opposite_the_sun() -> None:
    from blitz.data.hillshade import shadow_azimuth_deg

    assert shadow_azimuth_deg(0.0) == 180.0
    assert shadow_azimuth_deg(315.0) == 135.0
    assert shadow_azimuth_deg(90.0) == 270.0
    assert shadow_azimuth_deg(180.0) == 0.0


def test_viewport_slices_col_major_includes_halo() -> None:
    from blitz.data.hillshade import extract_viewport_patch, viewport_slices

    sl, rect = viewport_slices(
        (100, 80), 10, 20, 5, 15, axis_order="col-major", halo=1
    )
    assert sl[0] == slice(9, 21)
    assert sl[1] == slice(4, 16)
    assert rect == (9.0, 4.0, 12.0, 12.0)
    z = np.arange(100 * 80).reshape(100, 80)
    patch, prect = extract_viewport_patch(
        z, 10, 20, 5, 15, axis_order="col-major", max_edge=1600, halo=1
    )
    assert patch.shape == (12, 12)
    assert prect == rect


def test_downsample_xy_caps_long_edge() -> None:
    from blitz.data.hillshade import downsample_xy

    z = np.zeros((4000, 500), dtype=np.float32)
    d = downsample_xy(z, 1000)
    assert max(d.shape) <= 1000
    assert d.shape[0] == 1000
