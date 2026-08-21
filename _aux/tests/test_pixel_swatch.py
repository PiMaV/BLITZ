"""Cursor swatch: RGB uses the source pixel, not the LUT."""

from __future__ import annotations

import numpy as np

from blitz.tools import pixel_to_swatch_rgb8


def test_uint8_rgb_is_passthrough() -> None:
    assert pixel_to_swatch_rgb8(
        np.array([12, 200, 255], dtype=np.uint8),
        dtype=np.uint8,
    ) == (12, 200, 255)


def test_uint16_rgb_scales_to_8bit() -> None:
    assert pixel_to_swatch_rgb8(
        np.array([0, 32768, 65535], dtype=np.uint16),
        dtype=np.uint16,
    ) == (0, 128, 255)


def test_unit_float_rgb() -> None:
    assert pixel_to_swatch_rgb8(
        np.array([0.0, 0.5, 1.0], dtype=np.float32),
        dtype=np.float32,
    ) == (0, 128, 255)


def test_gray_pixel_has_no_rgb_swatch() -> None:
    assert pixel_to_swatch_rgb8(np.array([40.0]), dtype=np.float32) is None
