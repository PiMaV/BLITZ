"""Tests for LUT panel: level calculation (unit) and manual integration procedure.

Run unit tests: pytest tests/test_lut.py -v
Unit tests need only numpy (no Qt).
"""
from __future__ import annotations

import numpy as np
import pytest

from blitz.lut_levels import calculate_lut_levels


class TestCalculateLutLevels:
    """Unit tests for LUT level computation."""

    def test_minmax_percentile_zero(self) -> None:
        img = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        mn, mx = calculate_lut_levels(img, 0)
        assert mn == 10.0
        assert mx == 50.0

    def test_minmax_percentile_negative(self) -> None:
        img = np.array([1.0, 2.0, 3.0])
        mn, mx = calculate_lut_levels(img, -1)
        assert mn == 1.0
        assert mx == 3.0

    def test_percentile_1pct_clips_extrema(self) -> None:
        img = np.concatenate([
            [0.0, 100.0],
            np.linspace(10, 90, 98),
        ])
        mn, mx = calculate_lut_levels(img, 1.0)
        assert 0.0 < mn < 10.0
        assert 90.0 < mx < 100.0

    def test_percentile_handles_nans(self) -> None:
        img = np.array([1.0, np.nan, 5.0, np.nan, 10.0], dtype=np.float32)
        mn, mx = calculate_lut_levels(img, 0)
        assert mn == 1.0
        assert mx == 10.0

    def test_percentile_3d_stack(self) -> None:
        np.random.seed(42)
        img = np.random.rand(50, 64, 64).astype(np.float32)
        mn, mx = calculate_lut_levels(img, 1.0)
        global_min, global_max = float(np.nanmin(img)), float(np.nanmax(img))
        assert mn >= global_min
        assert mx <= global_max
        assert mn < mx

    def test_percentile_symmetric_bipolar(self) -> None:
        img = np.array([-100.0, -10.0, 0.0, 10.0, 100.0])
        mn, mx = calculate_lut_levels(img, 0)
        assert mn == -100.0
        assert mx == 100.0
