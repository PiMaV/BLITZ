"""View-only hillshade overlay. Analysis buffer stays original height."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QLabel

from ..data.hillshade import calculate_hillshade
from .viewer import ImageViewer


class ShadeAdapter:
    """Preview hillshade on a separate ImageItem; never replaces viewer.image."""

    def __init__(
        self,
        viewer: ImageViewer,
        status_label: Optional[QLabel] = None,
    ) -> None:
        self.viewer = viewer
        self._status = status_label
        self._preview = False
        self._azimuth = 315.0
        self._elevation = 45.0
        self._z_factor = 1.0

        self._item = pg.ImageItem()
        # Just above the base ImageItem (z≈0); crosshair/ROI sit much higher.
        self._item.setZValue(0.1)
        self._item.setVisible(False)
        # Match main image axis order so overlay aligns
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
        viewer.image_size_changed.connect(self._on_size_changed)

    def set_preview(self, on: bool) -> None:
        self._preview = bool(on)
        if not self._preview:
            self._item.clear()
            self._item.setVisible(False)
            try:
                self.viewer.imageItem.setOpacity(1.0)
            except Exception:
                pass
            self._set_status("Preview off · analysis = height")
            return
        self._refresh_now()

    def set_params(
        self,
        *,
        azimuth: Optional[float] = None,
        elevation: Optional[float] = None,
        z_factor: Optional[float] = None,
    ) -> None:
        if azimuth is not None:
            self._azimuth = float(azimuth) % 360.0
        if elevation is not None:
            self._elevation = float(np.clip(elevation, 0.0, 90.0))
        if z_factor is not None:
            self._z_factor = float(max(0.01, z_factor))
        if self._preview:
            self._schedule()

    def _schedule(self, *_args) -> None:
        if self._preview and not self._timer.isActive():
            self._timer.start()

    def _on_size_changed(self, *_args) -> None:
        if self._preview:
            self._refresh_now()

    def _height_frame(self) -> Optional[np.ndarray]:
        """Authoritative height: ImageData / ImageView buffer (not the overlay)."""
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

    def _refresh_now(self) -> None:
        if not self._preview:
            return
        frame = self._height_frame()
        if frame is None:
            self._item.setVisible(False)
            self._set_status("No image · load a height map first")
            return
        try:
            shade = calculate_hillshade(
                frame,
                self._azimuth,
                self._elevation,
                self._z_factor,
            )
        except Exception as e:
            self._item.setVisible(False)
            self._set_status(f"Shade failed: {e}")
            return

        # Keep analysis image fully intact; only hide it visually.
        try:
            self.viewer.imageItem.setOpacity(0.0)
        except Exception:
            pass
        self._item.setImage(shade, autoLevels=False)
        self._item.setLevels((0.0, 1.0))
        self._item.setVisible(True)
        self._set_status(
            f"Hillshade preview · az {self._azimuth:.0f}° elev {self._elevation:.0f}° "
            f"Z×{self._z_factor:g} · analysis = height"
        )

    def _set_status(self, text: str) -> None:
        if self._status is not None:
            self._status.setText(text)
