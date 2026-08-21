"""Load-dialog auto-crop: content bounding box on a preview."""

from __future__ import annotations

import numpy as np

from blitz.data.autocrop import content_bbox_xywh


def test_letterbox_finds_inner_block() -> None:
    img = np.zeros((80, 100), dtype=np.uint8)
    img[10:70, 15:85] = 200
    x, y, w, h = content_bbox_xywh(img, margin_frac=0.0)
    assert (x, y, w, h) == (15, 10, 70, 60)


def test_margin_grows_the_box() -> None:
    img = np.zeros((80, 100), dtype=np.uint8)
    img[10:70, 15:85] = 200
    x, y, w, h = content_bbox_xywh(img, margin_frac=0.03)
    assert x < 15 and y < 10
    assert x + w > 85 and y + h > 70
    assert x >= 0 and y >= 0
    assert x + w <= 100 and y + h <= 80


def test_uniform_frame_is_full_image() -> None:
    img = np.ones((40, 50), dtype=np.float32) * 7.0
    assert content_bbox_xywh(img) == (0, 0, 50, 40)


def test_rgb_uses_luminance() -> None:
    img = np.zeros((30, 40, 3), dtype=np.uint8)
    img[5:25, 8:32] = (10, 200, 10)
    x, y, w, h = content_bbox_xywh(img, margin_frac=0.0)
    assert (x, y, w, h) == (8, 5, 24, 20)
