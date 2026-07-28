"""Open-polyline intensity profile: ROI, dock plot, envelopes, CSV, linked cursor."""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..data.polyline_profile import (
    PolylineProfileResult,
    nearest_sample_index,
    roi_points_xy,
    sample_polyline_profile,
    vertex_path_lengths,
)
from ..theme import get_plot_bg
from .viewer import ImageViewer

_PATH_PEN = pg.mkPen((80, 200, 220), width=2)
_ENV_PEN = pg.mkPen((60, 140, 160), width=1)
_DS_PEN = pg.mkPen((100, 130, 220), width=1)
# Amber sync cursor — distinct from magenta linked-cursor on H/V
_SYNC_COLOR = (255, 170, 40)
_SYNC_PEN = pg.mkPen(_SYNC_COLOR, width=2)
_ROI_PEN = pg.mkPen((80, 200, 220, 200), width=2)
_BAND_PEN = pg.mkPen((80, 200, 220, 160), width=1, style=Qt.PenStyle.DotLine)
_BAND_FILL = pg.mkBrush(80, 200, 220, 40)
_HANDLE_LABEL_COLOR = (255, 230, 140)
_VERTEX_LINE_PEN = pg.mkPen((255, 210, 80, 220), width=2)
_TIP_COLOR = (255, 220, 160)


def _fmt_tip(val: float) -> str:
    if not np.isfinite(val):
        return "—"
    if float(val) == int(val):
        return str(int(val))
    return f"{val:.4g}"


class PolylineProfileController(QWidget):
    """Tools-driven open polyline + intensity-vs-path-length plot."""

    def __init__(
        self,
        viewer: ImageViewer,
        *,
        on_probe: Optional[Callable[[int, int], None]] = None,
        get_envelope_pct: Optional[Callable[[], float]] = None,
        get_mm_scale: Optional[Callable[[], tuple[bool, float, float]]] = None,
        linked_cursor_enabled: Optional[Callable[[], bool]] = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._viewer = viewer
        self._on_probe = on_probe
        self._get_envelope_pct = get_envelope_pct or (lambda: 0.0)
        self._get_mm_scale = get_mm_scale or (lambda: (False, 1.0, 1.0))
        self._linked_cursor_enabled = linked_cursor_enabled or (lambda: False)

        self._active = False
        self._last: PolylineProfileResult | None = None
        self._s_display: np.ndarray | None = None  # after mm scale

        # --- ROI (open) ---
        self._roi = pg.PolyLineROI(
            [[0, 0], [20, 0], [20, 20]],
            closed=False,
            pen=_ROI_PEN,
        )
        self._roi.handleSize = 9
        self._roi.hide()
        viewer.view.addItem(self._roi)

        self._pixel = pg.ROI(
            pos=[0, 0],
            size=[1, 1],
            pen=_SYNC_PEN,
            movable=False,
            rotatable=False,
            resizable=False,
        )
        self._pixel.setZValue(10_001)
        self._pixel.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        while self._pixel.handles:
            self._pixel.removeHandle(0)
        self._pixel.hide()
        viewer.view.addItem(self._pixel)

        self._pixel_tip = pg.TextItem("", color=_TIP_COLOR, anchor=(0.0, 1.0))
        self._pixel_tip.setZValue(10_003)
        self._pixel_tip.hide()
        viewer.view.addItem(self._pixel_tip)

        # Perpendicular band outline in the image (±width)
        self._band_lo = pg.PlotDataItem(pen=_BAND_PEN)
        self._band_hi = pg.PlotDataItem(pen=_BAND_PEN)
        self._band_fill = pg.FillBetweenItem(
            self._band_lo, self._band_hi, brush=_BAND_FILL
        )
        self._band_lo.setZValue(9_990)
        self._band_hi.setZValue(9_990)
        self._band_fill.setZValue(9_989)
        for item in (self._band_lo, self._band_hi, self._band_fill):
            item.hide()
            viewer.view.addItem(item)

        self._handle_labels: list[pg.TextItem] = []
        self._axis_vertex_lines: list[pg.InfiniteLine] = []
        self._axis_vertex_labels: list[pg.TextItem] = []

        # --- Plot + controls ---
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)

        row = QHBoxLayout()
        self.spin_width = QSpinBox()
        self.spin_width.setPrefix("Width: ")
        self.spin_width.setRange(0, 99)
        self.spin_width.setValue(0)
        self.spin_width.setToolTip(
            "Half-width ⊥ to the path (px).\n"
            "Main curve = mean intensity across that band "
            "(Width 0 = single centerline pixel)."
        )
        row.addWidget(self.spin_width)
        self.cb_env = QCheckBox("Envelope ⊥")
        self.cb_env.setToolTip(
            "Also plot min/max (or View→Envelope %) across the same band. "
            "Main curve stays the band mean. Band outline on image when Width > 0."
        )
        row.addWidget(self.cb_env)
        self.cb_env_ds = QCheckBox("Envelope frames")
        self.cb_env_ds.setToolTip(
            "Needs T>1. At each path sample: temporal min (lower) and max (upper) "
            "over all frames.\n"
            "• Upper high → somewhere in time this spot was bright (event).\n"
            "• Lower≈upper low → quiet/dark the whole time.\n"
            "• Wide gap → strong temporal variability at that path position."
        )
        row.addWidget(self.cb_env_ds)
        self.cb_mm = QCheckBox("Path in au")
        self.cb_mm.setToolTip("Scale path axis with Tools → Measure calibration")
        row.addWidget(self.cb_mm)
        self.btn_csv = QPushButton("CSV")
        self.btn_csv.setToolTip("Export path profile to CSV")
        row.addWidget(self.btn_csv)
        row.addStretch(1)
        stats_box = QLabel("Stats")
        stats_box.setStyleSheet("color: #888; font-weight: bold;")
        row.addWidget(stats_box)
        self._status = QLabel("—")
        self._status.setStyleSheet("color: #aaa;")
        self._status.setToolTip("Path profile statistics")
        row.addWidget(self._status)
        layout.addLayout(row)

        self.plot = pg.PlotWidget(background=get_plot_bg())
        self.plot.showGrid(x=True, y=True, alpha=0.4)
        # Compact axes — avoid huge “path (px)” label band eating dock height
        self.plot.setLabel("bottom", "s", units="px")
        self.plot.setLabel("left", "I")
        for name, size_fn, size in (
            ("bottom", "setHeight", 18),
            ("left", "setWidth", 36),
        ):
            ax = self.plot.getAxis(name)
            ax.setStyle(tickTextOffset=1, autoExpandTextSpace=False)
            getattr(ax, size_fn)(size)
            try:
                ax.label.setMaximumHeight(12)
            except Exception:
                pass
        try:
            self.plot.plotItem.layout.setContentsMargins(1, 1, 1, 1)
            self.plot.plotItem.setContentsMargins(0, 0, 0, 0)
        except Exception:
            pass
        # Manual Y range — enableAutoRange grows unbounded on repeated refresh
        # (same pyqtgraph quirk as H/V extraction plots).
        self.plot.getViewBox().enableAutoRange(x=True, y=False)
        self._curve = self.plot.plot(pen=_PATH_PEN)
        self._env_lo_c = self.plot.plot(pen=_ENV_PEN)
        self._env_hi_c = self.plot.plot(pen=_ENV_PEN)
        self._ds_lo_c = self.plot.plot(pen=_DS_PEN)
        self._ds_hi_c = self.plot.plot(pen=_DS_PEN)
        self._path_marker = pg.ScatterPlotItem(
            size=10,
            symbol="o",
            pen=_SYNC_PEN,
            brush=pg.mkBrush(*_SYNC_COLOR, 220),
        )
        self._path_marker.setZValue(20)
        self._path_marker.hide()
        self.plot.addItem(self._path_marker)
        self._path_tip = pg.TextItem("", color=_TIP_COLOR, anchor=(0.0, 1.0))
        self._path_tip.setZValue(21)
        self._path_tip.hide()
        self.plot.addItem(self._path_tip)
        layout.addWidget(self.plot, 1)

        self._draw_timer = QTimer(self)
        self._draw_timer.setSingleShot(True)
        self._draw_timer.setInterval(40)
        self._draw_timer.timeout.connect(self.refresh)

        self.spin_width.valueChanged.connect(self.refresh)
        self.cb_env.stateChanged.connect(self._on_env_toggled)
        self.cb_env_ds.stateChanged.connect(self.refresh)
        self.cb_mm.stateChanged.connect(self.refresh)
        self.btn_csv.clicked.connect(self.export_csv)

        self._roi.sigRegionChanged.connect(self._schedule_refresh)
        self._roi.sigRegionChangeFinished.connect(self.refresh)
        viewer.timeLine.sigPositionChanged.connect(self._on_frame)
        viewer.image_changed.connect(self._on_image_changed)

        self._proxy_plot = None
        self._proxy_main = None

    # --- public API ---

    def set_active(self, active: bool) -> None:
        self._active = bool(active)
        if self._active:
            self._reshape_roi()
            self._roi.show()
            self.refresh()
            self._connect_linked()
        else:
            self._roi.hide()
            self._pixel.hide()
            self._pixel_tip.hide()
            self._path_marker.hide()
            self._path_tip.hide()
            self._hide_band()
            self._clear_handle_markers()
            self._curve.clear()
            self._env_lo_c.clear()
            self._env_hi_c.clear()
            self._ds_lo_c.clear()
            self._ds_hi_c.clear()
            self._last = None
            self._status.setText("—")

    def is_active(self) -> bool:
        return self._active

    def _clear_handle_markers(self) -> None:
        for lab in self._handle_labels:
            try:
                self._viewer.view.removeItem(lab)
            except Exception:
                pass
        self._handle_labels.clear()
        for line in self._axis_vertex_lines:
            try:
                self.plot.removeItem(line)
            except Exception:
                pass
        self._axis_vertex_lines.clear()
        for lab in self._axis_vertex_labels:
            try:
                self.plot.removeItem(lab)
            except Exception:
                pass
        self._axis_vertex_labels.clear()

    def _update_handle_markers(
        self, points: np.ndarray, s_vertices: np.ndarray
    ) -> None:
        """Number handles in the image and mark the same indices on the path axis."""
        n = len(points)
        while len(self._handle_labels) > n:
            lab = self._handle_labels.pop()
            try:
                self._viewer.view.removeItem(lab)
            except Exception:
                pass
        for i in range(n):
            text = str(i + 1)
            x, y = float(points[i, 0]), float(points[i, 1])
            if i < len(self._handle_labels):
                lab = self._handle_labels[i]
                lab.setText(text)
                lab.setPos(x, y)
                lab.show()
            else:
                lab = pg.TextItem(
                    text,
                    color=_HANDLE_LABEL_COLOR,
                    anchor=(0.5, 1.2),
                )
                lab.setZValue(10_002)
                try:
                    from PyQt6.QtGui import QFont
                    f = QFont()
                    f.setPointSize(11)
                    f.setBold(True)
                    lab.setFont(f)
                except Exception:
                    pass
                lab.setPos(x, y)
                self._viewer.view.addItem(lab)
                self._handle_labels.append(lab)

        while len(self._axis_vertex_lines) > n:
            line = self._axis_vertex_lines.pop()
            lab = self._axis_vertex_labels.pop()
            try:
                self.plot.removeItem(line)
                self.plot.removeItem(lab)
            except Exception:
                pass
        for i in range(n):
            s = float(s_vertices[i])
            text = str(i + 1)
            if i < len(self._axis_vertex_lines):
                self._axis_vertex_lines[i].setPos(s)
                self._axis_vertex_lines[i].show()
                self._axis_vertex_labels[i].setText(text)
                self._axis_vertex_labels[i].show()
            else:
                line = pg.InfiniteLine(
                    pos=s, angle=90, pen=_VERTEX_LINE_PEN, movable=False
                )
                line.setZValue(5)
                self.plot.addItem(line)
                self._axis_vertex_lines.append(line)
                lab = pg.TextItem(
                    text,
                    color=_HANDLE_LABEL_COLOR,
                    anchor=(0.5, 0.5),
                )
                lab.setZValue(6)
                try:
                    from PyQt6.QtGui import QFont
                    f = QFont()
                    f.setPointSize(11)
                    f.setBold(True)
                    lab.setFont(f)
                except Exception:
                    pass
                self.plot.addItem(lab)
                self._axis_vertex_labels.append(lab)
        self._place_axis_labels()

    def _place_axis_labels(self) -> None:
        """Vertex numbers mid-height so they stay clear of the curve baseline."""
        try:
            (_, _), (y0, y1) = self.plot.viewRange()
        except Exception:
            return
        y = 0.5 * (y0 + y1)
        for lab, line in zip(self._axis_vertex_labels, self._axis_vertex_lines):
            lab.setPos(float(line.value()), y)

    def _on_env_toggled(self) -> None:
        if self.cb_env.isChecked() and self.spin_width.value() == 0:
            self.spin_width.setValue(1)
        self.refresh()

    def _hide_band(self) -> None:
        self._band_lo.hide()
        self._band_hi.hide()
        self._band_fill.hide()

    def _update_band_overlay(self, result: PolylineProfileResult, width: int) -> None:
        """Draw ±width corridor in the image (what Envelope ⊥ samples)."""
        if width <= 0 or result.xs.size < 2:
            self._hide_band()
            return
        xs, ys = result.xs, result.ys
        tx = np.gradient(xs)
        ty = np.gradient(ys)
        norm = np.hypot(tx, ty)
        norm = np.where(norm < 1e-9, 1.0, norm)
        nx, ny = -ty / norm, tx / norm
        w = float(width)
        lo_x, lo_y = xs - w * nx, ys - w * ny
        hi_x, hi_y = xs + w * nx, ys + w * ny
        self._band_lo.setData(lo_x, lo_y)
        self._band_hi.setData(hi_x, hi_y)
        self._band_lo.show()
        self._band_hi.show()
        self._band_fill.show()

    def _sync_frames_envelope_enabled(self) -> None:
        img = self._viewer.image
        t = int(img.shape[0]) if img is not None else 0
        ok = t > 1
        self.cb_env_ds.setEnabled(ok)
        if not ok and self.cb_env_ds.isChecked():
            self.cb_env_ds.blockSignals(True)
            self.cb_env_ds.setChecked(False)
            self.cb_env_ds.blockSignals(False)
            self._ds_lo_c.clear()
            self._ds_hi_c.clear()
        if not ok:
            self.cb_env_ds.setToolTip(
                "Needs a time series (T>1): temporal min/max along the path."
            )
        else:
            self.cb_env_ds.setToolTip(
                "At each path sample: temporal min (lower) and max (upper) over frames.\n"
                "• Upper high → bright event sometime at this spot.\n"
                "• Lower≈upper low → quiet/dark the whole time.\n"
                "• Wide gap → strong temporal variability."
            )
    # --- internals ---

    def _on_frame(self, *_args) -> None:
        if self._active:
            self.refresh()

    def _on_image_changed(self) -> None:
        if self._active:
            self._reshape_roi()
            self.refresh()

    def _reshape_roi(self) -> None:
        img = self._viewer.image
        if img is None:
            return
        w, h = int(img.shape[1]), int(img.shape[2])
        self._roi.setPos((0, 0))
        # Keep relative shape if already edited; only init when tiny/default
        pts = roi_points_xy(self._roi)
        span = 0.0
        if len(pts) >= 2:
            span = float(np.hypot(*(pts[-1] - pts[0])))
        if span < 2 or pts.max() > max(w, h) * 1.5:
            x0, y0 = w * 0.2, h * 0.5
            x1, y1 = w * 0.8, h * 0.5
            xm, ym = w * 0.5, h * 0.35
            self._roi.setPoints([[x0, y0], [xm, ym], [x1, y1]])

    def _schedule_refresh(self) -> None:
        if self._active and not self._draw_timer.isActive():
            self._draw_timer.start()

    def refresh(self) -> None:
        if not self._active:
            return
        img = self._viewer.image
        if img is None:
            return
        self._sync_frames_envelope_enabled()
        frame = img[int(self._viewer.currentIndex)]
        points = roi_points_xy(self._roi)
        width = int(self.spin_width.value())
        pct = float(self._get_envelope_pct())
        want_env = self.cb_env.isChecked()
        want_ds = self.cb_env_ds.isChecked() and self.cb_env_ds.isEnabled()
        # Skip heavy over-frames envelope while ROI is mid-drag (timer path).
        from_drag_timer = self.sender() is self._draw_timer
        do_ds = want_ds and not from_drag_timer
        volume = img if do_ds else None

        result = sample_polyline_profile(
            frame,
            points,
            width=width,
            envelope_pct=pct,
            want_perp_envelope=want_env,
            volume=volume,
            want_dataset_envelope=do_ds,
        )
        if result is None:
            self._last = None
            self._curve.clear()
            self._hide_band()
            self._clear_handle_markers()
            return
        self._last = result
        self._update_band_overlay(result, width)
        use_mm, n_px, px_in_mm = self._get_mm_scale()
        use_mm = use_mm and self.cb_mm.isChecked()
        if use_mm and n_px > 0:
            scale = px_in_mm / float(n_px)
            s = result.s * scale
            s_vert = vertex_path_lengths(points) * scale
            self.plot.setLabel("bottom", "s", units="au")
        else:
            s = result.s
            s_vert = vertex_path_lengths(points)
            self.plot.setLabel("bottom", "s", units="px")
        # Keep bottom axis compact after setLabel (pyqtgraph expands it again)
        try:
            ax = self.plot.getAxis("bottom")
            ax.setStyle(autoExpandTextSpace=False)
            ax.setHeight(18)
        except Exception:
            pass
        self._s_display = s

        inten = result.intensity
        env_lo, env_hi = result.env_lo, result.env_hi
        ds_lo, ds_hi = result.env_ds_lo, result.env_ds_hi
        self._display_intensity = inten
        self._display_env = (env_lo, env_hi) if want_env else (None, None)
        self._display_ds = (ds_lo, ds_hi) if do_ds else (None, None)

        self._curve.setData(s, inten)
        if want_env and env_lo is not None and env_hi is not None:
            self._env_lo_c.setData(s, env_lo)
            self._env_hi_c.setData(s, env_hi)
        else:
            self._env_lo_c.clear()
            self._env_hi_c.clear()
        if do_ds and ds_lo is not None and ds_hi is not None:
            self._ds_lo_c.setData(s, ds_lo)
            self._ds_hi_c.setData(s, ds_hi)
        elif not want_ds:
            self._ds_lo_c.clear()
            self._ds_hi_c.clear()
        # else: keep previous frame-envelope while dragging
        self._update_handle_markers(points, s_vert)
        self._set_intensity_range(inten, env_lo if want_env else None, env_hi if want_env else None,
                                  ds_lo if do_ds else None, ds_hi if do_ds else None)
        self._place_axis_labels()
        finite = inten[np.isfinite(inten)]
        if finite.size:
            self._status.setText(
                f"n={len(s)}  L={float(s[-1]):.1f}  "
                f"I=[{float(np.min(finite)):.4g} … {float(np.max(finite)):.4g}]"
            )
        else:
            self._status.setText(f"n={len(s)}  (out of image)")

    def _set_intensity_range(
        self,
        inten: np.ndarray,
        env_lo: np.ndarray | None,
        env_hi: np.ndarray | None,
        ds_lo: np.ndarray | None,
        ds_hi: np.ndarray | None,
    ) -> None:
        chunks = [inten]
        if env_lo is not None and env_hi is not None:
            chunks.extend([env_lo, env_hi])
        if ds_lo is not None and ds_hi is not None:
            chunks.extend([ds_lo, ds_hi])
        stacked = np.concatenate([np.asarray(c, dtype=np.float64).ravel() for c in chunks])
        finite = stacked[np.isfinite(stacked)]
        if finite.size == 0:
            return
        lo = float(np.min(finite))
        hi = float(np.max(finite))
        if hi <= lo:
            pad = max(abs(lo) * 0.05, 1.0)
            lo, hi = lo - pad, hi + pad
        else:
            pad = 0.05 * (hi - lo)
            # Extra headroom at top so vertex numbers stay readable
            lo, hi = lo - pad, hi + pad * 2.5
        self.plot.setYRange(lo, hi, padding=0)
    def export_csv(self) -> None:
        if self._last is None or self._s_display is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export polyline profile",
            "polyline_profile.csv",
            "CSV (*.csv)",
        )
        if not path:
            return
        r = self._last
        s = self._s_display
        inten = getattr(self, "_display_intensity", r.intensity)
        cols = ["s", "intensity", "x", "y"]
        data = [s, inten, r.xs, r.ys]
        env_lo, env_hi = getattr(self, "_display_env", (r.env_lo, r.env_hi))
        if env_lo is not None and env_hi is not None:
            cols += ["env_lo", "env_hi"]
            data += [env_lo, env_hi]
        ds_lo, ds_hi = getattr(self, "_display_ds", (r.env_ds_lo, r.env_ds_hi))
        if ds_lo is not None and ds_hi is not None:
            cols += ["env_frames_lo", "env_frames_hi"]
            data += [ds_lo, ds_hi]
        arr = np.column_stack(data)
        header = ",".join(cols)
        np.savetxt(path, arr, delimiter=",", header=header, comments="")

    # --- linked cursor ---

    def _connect_linked(self) -> None:
        if self._proxy_plot is None:
            self._proxy_plot = pg.SignalProxy(
                self.plot.scene().sigMouseMoved,
                rateLimit=40,
                slot=self._on_plot_moved,
            )
        if self._proxy_main is None:
            self._proxy_main = pg.SignalProxy(
                self._viewer.scene.sigMouseMoved,
                rateLimit=40,
                slot=self._on_main_moved,
            )

    def _clear_link_markers(self) -> None:
        self._pixel.hide()
        self._pixel_tip.hide()
        self._path_marker.hide()
        self._path_tip.hide()

    def _show_sample(self, idx: int) -> None:
        if self._last is None or self._s_display is None:
            return
        idx = int(np.clip(idx, 0, len(self._last.xs) - 1))
        x = int(np.floor(self._last.xs[idx]))
        y = int(np.floor(self._last.ys[idx]))
        self._pixel.setPos([x, y])
        self._pixel.setSize([1, 1])
        self._pixel.show()
        # Point ON the profile curve (correct here: s maps to sampled intensity)
        inten = getattr(self, "_display_intensity", self._last.intensity)
        if inten is not None and idx < len(inten) and np.isfinite(inten[idx]):
            s = float(self._s_display[idx])
            iv = float(inten[idx])
            self._path_marker.setData([s], [iv])
            self._path_marker.show()
            tip = _fmt_tip(iv)
            self._path_tip.setText(tip)
            self._path_tip.setPos(s, iv)
            self._path_tip.show()
            self._pixel_tip.setText(tip)
            self._pixel_tip.setPos(x + 1.2, y)
            self._pixel_tip.show()
        else:
            self._path_marker.hide()
            self._path_tip.hide()
            self._pixel_tip.hide()
        if self._on_probe is not None:
            self._on_probe(x, y)

    def _on_plot_moved(self, args) -> None:
        if not self._active or not self._linked_cursor_enabled():
            return
        if self._last is None or self._s_display is None:
            return
        pos = args[0] if isinstance(args, (tuple, list)) else args
        try:
            pt = self.plot.plotItem.vb.mapSceneToView(pos)
        except Exception:
            return
        s = float(pt.x())
        idx = int(np.argmin(np.abs(self._s_display - s)))
        self._show_sample(idx)

    def _on_main_moved(self, args) -> None:
        if not self._active or not self._linked_cursor_enabled():
            return
        if self._last is None:
            return
        pos = args[0] if isinstance(args, (tuple, list)) else args
        pt = self._viewer.view.vb.mapSceneToView(pos)
        x, y = float(pt.x()), float(pt.y())
        idx = nearest_sample_index(self._last.xs, self._last.ys, x, y)
        if idx is None:
            return
        dist = float(
            np.hypot(self._last.xs[idx] - x, self._last.ys[idx] - y)
        )
        # Near path: within width+3 px
        thresh = max(3.0, float(self.spin_width.value()) + 3.0)
        if dist > thresh:
            self._clear_link_markers()
            return
        self._show_sample(idx)

    def leaveEvent(self, event) -> None:
        self._clear_link_markers()
        super().leaveEvent(event)
