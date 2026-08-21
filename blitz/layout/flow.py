"""View-only D8 accumulation overlay. Analysis buffer stays original height."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import QRectF, QTimer
from PyQt6.QtWidgets import QLabel

from ..data.flow import accumulation_rgba, d8_accumulation
from ..data.hillshade import VIEWPORT_MAX_EDGE, extract_viewport_patch
from .viewer import ImageViewer


class FlowAdapter:
    """Paint D8 accumulation on a separate ImageItem; never replaces viewer.image."""

    def __init__(
        self,
        viewer: ImageViewer,
        status_label: Optional[QLabel] = None,
    ) -> None:
        self.viewer = viewer
        self._status = status_label
        self._preview = False
        self._log_scale = True
        self._overlay_rect: Optional[tuple[float, float, float, float]] = None

        self._item = pg.ImageItem()
        self._item.setZValue(0.15)
        self._item.setOpacity(0.92)
        self._item.setVisible(False)
        try:
            self._item.setOpts(axisOrder=viewer.imageItem.axisOrder)
        except Exception:
            pass
        viewer.view.addItem(self._item)

        self._timer = QTimer()
        self._timer.setSingleShot(True)
        self._timer.setInterval(80)
        self._timer.timeout.connect(self._refresh_now)

        viewer.timeLine.sigPositionChanged.connect(self._schedule)
        viewer.image_changed.connect(self._schedule)
        viewer.image_size_changed.connect(self._schedule)
        viewer.destroyed.connect(self._on_viewer_destroyed)
        try:
            viewer.view.getViewBox().sigRangeChanged.connect(self._schedule)
        except Exception:
            pass

    def set_preview(self, on: bool) -> None:
        self._preview = bool(on)
        if not self._preview:
            self._timer.stop()
            self._item.clear()
            self._item.setVisible(False)
            self._overlay_rect = None
            self._set_status("Flow off · analysis = height")
            return
        self._schedule()

    def set_log_scale(self, on: bool) -> None:
        want = bool(on)
        if want == self._log_scale:
            return
        self._log_scale = want
        if self._preview:
            self._schedule()

    def _schedule(self, *_args) -> None:
        if not self._preview:
            return
        self._timer.start()

    def _on_viewer_destroyed(self, *_args) -> None:
        self._preview = False
        self._timer.stop()

    def _height_frame(self) -> Optional[np.ndarray]:
        img = self.viewer.image
        if img is None:
            return None
        try:
            frame = img[int(self.viewer.currentIndex)]
        except Exception:
            return None
        if frame is None or np.size(frame) == 0:
            return None
        return np.asarray(frame)

    def _view_window(
        self,
        frame: np.ndarray,
    ) -> tuple[float, float, float, float, int, str]:
        order = str(getattr(self.viewer.imageItem, "axisOrder", "col-major"))
        spatial = np.asarray(frame).shape[:2]
        if order == "row-major":
            ny, nx = int(spatial[0]), int(spatial[1])
        else:
            nx, ny = int(spatial[0]), int(spatial[1])
        max_edge = VIEWPORT_MAX_EDGE
        x0, x1, y0, y1 = 0.0, float(nx), 0.0, float(ny)
        try:
            vb = self.viewer.view.getViewBox()
            (vx0, vx1), (vy0, vy1) = vb.viewRange()
            x0, x1, y0, y1 = float(vx0), float(vx1), float(vy0), float(vy1)
            px = max(int(vb.width()), int(vb.height()), 1)
            max_edge = int(max(64, min(VIEWPORT_MAX_EDGE, px)))
        except Exception:
            pass
        return x0, x1, y0, y1, max_edge, order

    def _viewport_patch(
        self,
    ) -> Optional[tuple[np.ndarray, tuple[float, float, float, float]]]:
        frame = self._height_frame()
        if frame is None:
            return None
        x0, x1, y0, y1, max_edge, order = self._view_window(frame)
        return extract_viewport_patch(
            frame,
            x0,
            x1,
            y0,
            y1,
            axis_order=order,
            max_edge=max_edge,
        )

    def _refresh_now(self) -> None:
        if not self._preview:
            return
        spec = self._viewport_patch()
        if spec is None:
            self._item.setVisible(False)
            self._set_status("No image · load a height map first")
            return
        patch, rect = spec
        self._overlay_rect = rect
        try:
            acc = d8_accumulation(patch)
            rgba = accumulation_rgba(acc, log_scale=self._log_scale)
        except Exception as e:
            self._item.setVisible(False)
            self._set_status(f"Flow failed: {e}")
            return
        self._item.setImage(rgba, autoLevels=False)
        rx, ry, rw, rh = rect
        self._item.setRect(QRectF(rx, ry, rw, rh))
        self._item.setVisible(True)
        scale = "log1p" if self._log_scale else "linear"
        h, w = int(acc.shape[0]), int(acc.shape[1])
        self._set_status(
            f"D8 accumulation {scale} · {h}×{w} viewport · analysis = height"
        )

    def _set_status(self, text: str) -> None:
        if self._status is not None:
            self._status.setText(text)
