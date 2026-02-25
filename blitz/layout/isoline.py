"""Isoline overlay: contour lines at intensity levels. Independent of RoSEE."""

import numpy as np
import pyqtgraph as pg

from .viewer import ImageViewer


def _gaussian_filter(img: np.ndarray, ksize: int) -> np.ndarray:
    """Fast Gaussian blur via OpenCV (faster than pg.gaussianFilter for large images)."""
    import cv2

    k = (ksize, ksize)
    sigma = (ksize - 1) / 6.0
    return cv2.GaussianBlur(img, k, sigmaX=sigma)


class IsolineAdapter:

    def __init__(self, viewer: ImageViewer) -> None:
        self.viewer = viewer
        self._show_iso: bool = False
        self._n_iso = 1
        self._isocurves: list[pg.IsocurveItem] = []
        self._isolines: list[pg.InfiniteLine] = []
        self._reset_iso()

    def _reset_iso(self) -> None:
        while len(self._isocurves) > 0:
            curve = self._isocurves.pop()
            self.viewer.view.removeItem(curve)
            line = self._isolines.pop()
            self.viewer.getHistogramWidget().vb.removeItem(line)

        cmap: pg.ColorMap = pg.colormap.get("CET-C6")  # type: ignore
        for i in range(self._n_iso):
            curve = pg.IsocurveItem(
                level=0,
                pen=pg.mkPen(cmap[(i + 1) / self._n_iso]),
            )
            # Horizontal LUT: x=level, y=counts; vertical lines (angle=90) to select levels
            line = pg.InfiniteLine(
                angle=90,
                movable=True,
                pen=pg.mkPen(
                    cmap[(i + 1) / self._n_iso],
                    width=3,
                ),
            )
            line.setZValue(1000)
            line.sigPositionChanged.connect(self._on_level_changed)
            self._isocurves.append(curve)
            self._isolines.append(line)
            self.viewer.view.addItem(curve)
            self.viewer.getHistogramWidget().vb.addItem(line)
            if not self._show_iso:
                curve.hide()
                line.hide()

        try:
            img = self.viewer.now
            mean_val = float(np.mean(img))
            if self._n_iso > 1:
                std_val = float(np.std(img))
                levels = np.linspace(
                    mean_val - std_val,
                    mean_val + std_val,
                    self._n_iso,
                )
                levels = np.clip(levels, 0, float(np.max(img)))
            else:
                levels = [mean_val]
        except (ValueError, TypeError, IndexError, AttributeError):
            levels = [0.5] * self._n_iso

        for iso_line, level in zip(self._isolines, levels):
            iso_line.setValue(level)

    def update(
        self,
        on: bool,
        n: int = 1,
        smoothing: int = 0,
        downsample: int = 1,
    ) -> None:
        if on != self._show_iso:
            if self._show_iso:
                for curve, line in zip(self._isocurves, self._isolines):
                    curve.hide()
                    line.hide()
            else:
                for curve, line in zip(self._isocurves, self._isolines):
                    curve.show()
                    line.show()
            self._show_iso = not self._show_iso
        if n != self._n_iso:
            self._n_iso = n
            self._reset_iso()
        if self._show_iso:
            self._refresh(smoothing=smoothing, downsample=downsample)

    def _refresh(self, smoothing: int = 0, downsample: int = 1) -> None:
        try:
            img = np.asarray(self.viewer.now[..., 0], dtype=np.float32)
        except (ValueError, TypeError, IndexError, AttributeError):
            return
        if smoothing > 0:
            k = max(1, int(smoothing) | 1)  # odd kernel
            img = _gaussian_filter(img, k)

        effective_downsample = 1
        if downsample > 1:
            import cv2
            h, w = img.shape
            new_h, new_w = int(h / downsample), int(w / downsample)
            if new_h > 0 and new_w > 0:
                img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                effective_downsample = downsample

        for iso, line in zip(self._isocurves, self._isolines):
            iso.setLevel(line.value())
            iso.setData(img)
            if effective_downsample > 1:
                from PyQt6.QtGui import QTransform
                iso.setTransform(QTransform().scale(effective_downsample, effective_downsample))
            else:
                iso.resetTransform()

    def _on_level_changed(self) -> None:
        for curve, line in zip(self._isocurves, self._isolines):
            curve.setLevel(line.value())
