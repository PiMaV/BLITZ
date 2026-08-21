"""Unit tests for D8 flow accumulation (numpy only)."""

from __future__ import annotations

import numpy as np

from blitz.data.flow import accumulation_rgba, d8_accumulation, d8_downslope_flat


def test_flat_surface_is_all_sinks() -> None:
    z = np.ones((6, 7), dtype=np.float64)
    down = d8_downslope_flat(z)
    assert np.all(down == -1)
    acc = d8_accumulation(z)
    assert np.allclose(acc, 1.0)


def test_corner_drain_collects_all_cells() -> None:
    # z = x + y: unique downslope toward (0, 0).
    x = np.arange(8)
    y = np.arange(9)
    z = x[:, None] + y[None, :]
    acc = d8_accumulation(z.astype(np.float64))
    assert acc.shape == z.shape
    assert float(acc[0, 0]) == float(z.size)
    assert float(acc[-1, -1]) == 1.0


def test_trench_collects_from_both_sides() -> None:
    z = np.ones((12, 16), dtype=np.float64) * 10.0
    z[6, :] = 0.0
    acc = d8_accumulation(z)
    assert float(np.mean(acc[6, :])) > float(np.mean(acc[0, :]))
    assert float(np.mean(acc[6, :])) > float(np.mean(acc[-1, :]))


def test_accumulation_rgba_is_transparent_on_ridges() -> None:
    x = np.arange(10)
    z = np.broadcast_to(x[:, None], (10, 10)).copy().astype(np.float64)
    acc = d8_accumulation(z)
    rgba = accumulation_rgba(acc, log_scale=True)
    assert rgba.shape == (10, 10, 4)
    assert rgba.dtype == np.uint8
    # High-x ridge cells start at 1 upstream → transparent.
    assert int(rgba[-1, 5, 3]) == 0
    assert int(rgba[0, 5, 3]) > 0
    # Catchment must not become an opaque slab.
    assert float(np.median(rgba[..., 3])) < 40.0
