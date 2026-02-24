from typing import Optional

import numpy as np
import pyqtgraph as pg

# Line colors: green (per image), teal (per crosshair), blue (per dataset)
_PEN_MIN_PER_IMAGE = pg.mkPen((60, 140, 60), width=2)
_PEN_MAX_PER_IMAGE = pg.mkPen((100, 200, 100), width=2)
_PEN_MIN_PER_CROSSHAIR = pg.mkPen((60, 160, 160), width=2)
_PEN_MAX_PER_CROSSHAIR = pg.mkPen((100, 200, 200), width=2)
_PEN_MIN_PER_DATASET = pg.mkPen((60, 80, 160), width=2)
_PEN_MAX_PER_DATASET = pg.mkPen((100, 130, 220), width=2)
from PyQt6.QtCore import QPointF, QSize, Qt, pyqtSignal
from PyQt6.QtGui import QKeyEvent, QMouseEvent, QShowEvent, QWheelEvent
from PyQt6.QtWidgets import QFrame, QHBoxLayout, QLabel, QSizePolicy, QVBoxLayout, QWidget, QLineEdit

from .viewer import ImageViewer
from .. import settings
from ..theme import get_plot_bg, get_timeline_line_color, get_agg_band_bg, get_agg_separator_stylesheet

AGG_RANGE_BAND_HEIGHT = 32
_AGG_RANGE_BRUSH = pg.mkBrush(100, 220, 130, 140)
_AGG_RANGE_PEN = pg.mkPen((100, 220, 130), width=6)


class AggregateRangeBand(pg.PlotWidget):
    """
    Second timeline: aggregate range only. No plot, no axes (uses first timeline).
    X-linked to main timeline. Green-tinted bg for clear separation from frame.
    """

    def __init__(self, parent: QWidget, main_viewbox, **kargs) -> None:
        super().__init__(parent, background=get_agg_band_bg(), **kargs)
        for ax in ("left", "bottom", "top", "right"):
            self.plotItem.hideAxis(ax)  # type: ignore
        self.setMaximumHeight(AGG_RANGE_BAND_HEIGHT)
        self.setMinimumHeight(AGG_RANGE_BAND_HEIGHT)
        self.plotItem.vb.setYRange(0, 1, padding=0)  # type: ignore
        self.plotItem.vb.setMouseEnabled(x=True, y=False)  # type: ignore
        self.getViewBox().setXLink(main_viewbox)  # type: ignore
        self.crop_range = pg.LinearRegionItem(
            brush=_AGG_RANGE_BRUSH,
            pen=_AGG_RANGE_PEN,
        )
        if hasattr(self.crop_range, "handleSize"):
            self.crop_range.handleSize = 22
        self.crop_range.setZValue(0)
        self.addItem(self.crop_range)
        self.crop_range.hide()


class TimePlot(pg.PlotWidget):
    """First timeline: curve, cursor, frame selection only."""

    clicked_to_set_frame = pyqtSignal(int)  # frame index from click

    def __init__(
        self,
        parent: QWidget,
        image_viewer: pg.ImageView,
        **kargs,
    ) -> None:
        super().__init__(parent, background=get_plot_bg(), **kargs)
        self.hideAxis('left')
        self.addItem(image_viewer.timeLine)
        self.timeline = image_viewer.timeLine
        self.timeline.setMovable(False)
        self.timeline.setPen(pg.mkPen(get_timeline_line_color(), width=6))
        self.addItem(image_viewer.frameTicks)
        self.addItem(image_viewer.normRgn)
        self.old_roi_plot = image_viewer.ui.roiPlot
        self.old_roi_plot.setParent(None)
        self.image_viewer = image_viewer
        sizePolicy = QSizePolicy(
            QSizePolicy.Policy.Preferred,
            QSizePolicy.Policy.Preferred,
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.sizePolicy().hasHeightForWidth())
        self.setSizePolicy(sizePolicy)
        self.setMinimumSize(QSize(0, 40))
        _tl = get_timeline_line_color()
        self.norm_range = pg.LinearRegionItem(
            brush=pg.mkBrush(*_tl, 80),
            pen=pg.mkPen(_tl, width=5),
        )
        self.norm_range.setZValue(0)
        if hasattr(self.norm_range, "handleSize"):
            self.norm_range.handleSize = 12
        self.addItem(self.norm_range)
        self.norm_range.hide()
        self._accept_all_events = False

    def toggle_norm_range(self) -> None:
        if self.norm_range.isVisible():
            self.norm_range.hide()
        else:
            self.norm_range.show()

    def wheelEvent(self, event: QWheelEvent) -> None:
        if event.modifiers() == Qt.KeyboardModifier.ControlModifier:
            super().wheelEvent(event)
        else:
            rotation = event.angleDelta().y()
            if rotation < 0:
                pos = self.timeline.getPos()
                self.timeline.setPos((pos[0]+1, pos[1]))
            if rotation > 0:
                pos = self.timeline.getPos()
                self.timeline.setPos((pos[0]-1, pos[1]))

    def mousePressEvent(self, event: QMouseEvent) -> None:
        """Click timeline -> set frame."""
        if (event.modifiers() == Qt.KeyboardModifier.NoModifier
                and event.button() == Qt.MouseButton.LeftButton
                and not (self.norm_range.isVisible()
                    and (self.norm_range.mouseHovering
                        or self.norm_range.childItems()[0].mouseHovering
                        or self.norm_range.childItems()[1].mouseHovering))):
            x = self.plotItem.vb.mapSceneToView(  # type: ignore
                event.position()
            ).x()
            idx = max(0, int(round(x)))
            self.image_viewer.setCurrentIndex(idx)
            self.clicked_to_set_frame.emit(idx)
            event.accept()
        else:
            super().mousePressEvent(event)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        super().keyPressEvent(event)
        self.image_viewer.keyPressEvent(event)

    def keyReleaseEvent(self, event: QKeyEvent) -> None:
        super().keyReleaseEvent(event)
        self.image_viewer.keyReleaseEvent(event)


class TimelineStack(QWidget):
    """Top: frame timeline (curve, cursor). Bottom: aggregate range only. Linked X."""

    def __init__(self, parent: QWidget, image_viewer: pg.ImageView) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        top_row = QHBoxLayout()
        top_row.addStretch()
        self.label_timeline_mode = QLabel("Frame")
        self.label_timeline_mode.setStyleSheet("font-size: 16pt; font-weight: bold;")
        top_row.addWidget(self.label_timeline_mode)
        layout.addLayout(top_row)
        self.main_plot = TimePlot(self, image_viewer)
        self.agg_sep = QFrame()
        self.agg_sep.setFixedHeight(4)
        self.agg_sep.setStyleSheet(get_agg_separator_stylesheet())
        self.agg_sep_spacer = QWidget()
        self.agg_sep_spacer.setFixedHeight(6)
        self.agg_band = AggregateRangeBand(
            self, self.main_plot.getViewBox(),
        )
        layout.addWidget(self.main_plot, 1)
        layout.addWidget(self.agg_sep, 0)
        layout.addWidget(self.agg_sep_spacer, 0)
        layout.addWidget(self.agg_band, 0)
        self.main_plot.crop_range = self.agg_band.crop_range  # backward compat


class MeasureROI(pg.PolyLineROI):

    def __init__(
        self,
        viewer: ImageViewer,
        circ_box: QLineEdit,
        area_box: QLineEdit,
        bounding_rect_box: QLineEdit,
    ) -> None:
        self._viewer = viewer
        super().__init__([[0, 0], [0, 20], [10, 10]], closed=True)
        self.handleSize = 10
        self.sigRegionChanged.connect(self.update_labels)

        self.n_px: int = 1
        self.px_in_mm: float = 1
        self.show_in_mm = False

        self.line_labels = []
        self.angle_labels = []
        self.circ_box = circ_box
        self.area_box = area_box
        self.bounding_rect_box = bounding_rect_box

        self.bounding_rect = pg.RectROI([0, 0], [10, 20], movable=False)
        self._viewer.view.addItem(self.bounding_rect)
        self.bounding_rect.setPen(color=(180, 180, 0, 80), width=2)
        self.bounding_rect.removeHandle(0)
        self._show_bounding_rect = False
        self.sigRegionChanged.connect(self.update_bounding_rect)

        self._viewer.view.addItem(self)
        self.setPen(color=(128, 128, 0, 100), width=3)
        self._visible = True
        self.toggle()
        self._viewer.image_changed.connect(self.reshape)
        self.reshape()

    def reshape(self) -> None:
        height = self._viewer.image.shape[2]
        width = self._viewer.image.shape[1]
        self.toggle()
        self.setPoints([[0, 0], [0, 0.5*height], [0.5*width, 0.25*height]])
        self.update_bounding_rect()
        self.toggle()

    def toggle_bounding_rect(self) -> None:
        self._show_bounding_rect = not self._show_bounding_rect
        self.bounding_rect.setVisible(
            self._visible and self._show_bounding_rect
        )

    def toggle(self) -> None:
        self._visible = not self._visible
        self.setVisible(self._visible)
        self.bounding_rect.setVisible(
            self._visible and self._show_bounding_rect
        )
        for label in self.line_labels:
            label.setVisible(self._visible)
        for label in self.angle_labels:
            label.setVisible(self._visible)
        if self._visible:
            self.update_labels()

    def update_labels(self) -> None:
        self.update_angles()
        self.update_lines()

    def update_bounding_rect(self) -> None:
        bound = self.boundingRect()
        pos = self.getLocalHandlePositions()
        if len(pos) > 1:
            left = min([p[1].x() for p in pos])
            top = min([p[1].y() for p in pos])
            self.bounding_rect.setPos(self.mapToView(QPointF(left, top)))
            self.bounding_rect.setSize(bound.size())

    def update_angles(self) -> None:
        positions = self.getSceneHandlePositions()

        while len(self.angle_labels) > len(positions):
            label = self.angle_labels.pop()
            self._viewer.view.removeItem(label)

        for i, (_, pos) in enumerate(positions):
            _, prev_pos = positions[(i - 1) % len(positions)]
            _, next_pos = positions[(i + 1) % len(positions)]
            angle = pg.Point(pos - next_pos).angle(
                pg.Point(pos - prev_pos)
            )
            pos = self._viewer.view.mapToView(pos)
            if i < len(self.angle_labels):
                self.angle_labels[i].setPos(pos.x(), pos.y())
                self.angle_labels[i].setText(f"{angle:.2f}°")
            else:
                angle_label = pg.TextItem(f"{angle:.2f}°")
                angle_label.setPos(pos.x(), pos.y())
                self._viewer.view.addItem(angle_label)
                self.angle_labels.append(angle_label)

    def update_lines(self) -> None:
        positions = self.getSceneHandlePositions()

        while len(self.line_labels) > len(positions):
            label = self.line_labels.pop()
            self._viewer.view.removeItem(label)

        total_length = 0
        for i, (_, pos) in enumerate(positions):
            _, next_pos = positions[(i + 1) % len(positions)]
            pos = self._viewer.view.mapToView(pos)
            next_pos = self._viewer.view.mapToView(next_pos)
            length = pg.Point(pos - next_pos).length()
            mid = (pos + next_pos) / 2
            if self.show_in_mm:
                length = length * self.px_in_mm / self.n_px
            total_length += length
            pos = self._viewer.view.mapToView(pos)
            if i < len(self.line_labels):
                self.line_labels[i].setPos(mid.x(), mid.y())
                self.line_labels[i].setText(f"{length:.2f}")
            else:
                line_label = pg.TextItem(f"{length:.2f}")
                line_label.setPos(mid.x(), mid.y())
                self._viewer.view.addItem(line_label)
                self.line_labels.append(line_label)

        points = [x for pol in self.shape().toFillPolygons() for x in pol]
        area = 0
        for i in range(len(points)-1):
            area += points[i].x()*points[i+1].y()-points[i+1].x()*points[i].y()
        if self.show_in_mm:
            area = area * (self.px_in_mm / self.n_px)**2
        self.circ_box.setText(f"Circ: {total_length:.2f}")
        self.area_box.setText(f"Area: {-0.5 * area:,.2f}")
        rect = self.boundingRect()
        w, h = rect.width(), rect.height()
        if self.show_in_mm:
            w, h = w * self.px_in_mm / self.n_px, h * self.px_in_mm / self.n_px
        self.bounding_rect_box.setText(f"HxW: {w:.2f} x {h:.2f}")


class ExtractionLine(pg.InfiniteLine):

    def __init__(
        self,
        viewer: ImageViewer,
        vertical: bool = False,
    ) -> None:
        self._vertical = vertical
        super().__init__(
            angle=90 if vertical else 0,
            pen=pg.mkPen(
                color=(120, 120, 120, 200),
                style=Qt.PenStyle.DashLine,
                width=3,
            ),
            movable=True,
        )
        self._viewer = viewer
        self._viewer.view.addItem(self)
        self._upper = pg.InfiniteLine(
            angle=90 if self._vertical else 0,
            pen=pg.mkPen(
                color=(75, 75, 75, 200),
                style=Qt.PenStyle.DotLine,
                width=3,
            ),
            movable=False,
            hoverPen=pg.mkPen(
                color=(255, 0, 0, 200),
                style=Qt.PenStyle.DotLine,
                width=3,
            ),
        )
        self._lower = pg.InfiniteLine(
            angle=90 if self._vertical else 0,
            pen=pg.mkPen(
                color=(75, 75, 75, 200),
                style=Qt.PenStyle.DotLine,
                width=3,
            ),
            movable=False,
            hoverPen=pg.mkPen(
                color=(255, 0, 0, 200),
                style=Qt.PenStyle.DotLine,
                width=3,
            ),
        )
        self._width: int = 0
        self.sigPositionChanged.connect(self._move_bounds)
        self._coupled: None | ExtractionLine = None

    def paint(self, p, *args):
        if self._coupled is not None:
            value = self._coupled.value()
            value = (
                value - self.viewRect().x()  # type: ignore
            ) / self.viewRect().width()  # type: ignore
            self.markers[0] = (
                self.markers[0][0], value, self.markers[0][2],
            )
        super().paint(p, *args)

    def value(self) -> int:
        return super().value()  # type: ignore

    def couple(self, line: "ExtractionLine") -> None:
        self._coupled = line

    @property
    def width(self) -> int:
        return self._width

    def setMouseHover(self, hover) -> None:
        self._lower.setMouseHover(hover)
        self._upper.setMouseHover(hover)
        super().setMouseHover(hover)

    def setPos(self, p) -> None:
        if isinstance(p, (list, tuple, np.ndarray)) and not np.ndim(p) == 0:
            p = p[0] if self._vertical else p[1]
        elif isinstance(p, QPointF):
            p = p.x() if self._vertical else p.y()
        p = int(p) + 0.5  # type: ignore
        super().setPos(p)

    def _move_bounds(self) -> None:
        if self._bounds is not None:
            self._upper.setPos(self.value() - self._width - 0.5)
            self._lower.setPos(self.value() + self._width + 0.5)

    def change_width(self, width: int) -> None:
        if width == self._width or width < 0:
            return
        elif width == 0:
            self._viewer.view.removeItem(self._upper)
            self._viewer.view.removeItem(self._lower)
            self._width = 0
            return

        if self._width == 0:
            self._viewer.view.addItem(self._upper)
            self._viewer.view.addItem(self._lower)
        self._width = width
        self._move_bounds()

    def toggle(self) -> None:
        if self.movable:
            self.setMovable(False)
            self._viewer.view.removeItem(self)
            if self._width > 0:
                self._viewer.view.removeItem(self._upper)
                self._viewer.view.removeItem(self._lower)
        else:
            self.setMovable(True)
            self._viewer.view.addItem(self)
            if self._width > 0:
                self._viewer.view.addItem(self._upper)
                self._viewer.view.addItem(self._lower)


class ExtractionPlot(pg.PlotWidget):

    plotItem: pg.PlotItem

    def __init__(
        self,
        viewer: ImageViewer,
        vertical: bool = False,
        **kwargs,
    ) -> None:
        self._viewer = viewer
        self._vert = vertical
        self._width = 0  # n pixels above and below line to mean
        v_plot_viewbox = pg.ViewBox()
        if vertical:
            v_plot_viewbox.invertX()
            v_plot_viewbox.invertY()
        v_plot_item = pg.PlotItem(viewBox=v_plot_viewbox)
        v_plot_item.showGrid(x=True, y=True, alpha=0.4)
        super().__init__(plotItem=v_plot_item, background=get_plot_bg(), **kwargs)

        self._extractionline = ExtractionLine(viewer=viewer, vertical=vertical)
        self._extractionline.sigPositionChanged.connect(self.draw_line)
        self._viewer.timeLine.sigPositionChanged.connect(self.draw_line)
        self._viewer.image_changed.connect(self.draw_line)
        self._viewer.image_changed.connect(self._invalidate_dataset_envelope_cache)
        self._viewer.image_size_changed.connect(self.center_line)
        self._coupled: ExtractionPlot | None = None
        self._mark_coupled_position: bool = True
        self._show_minmax_per_image: bool = False
        self._show_envelope_per_crosshair: bool = False
        self._show_envelope_per_dataset: bool = False
        self._envelope_pct: float = 0.0  # 0 = min/max, >0 = percentile
        self._dataset_envelope_cache: tuple[np.ndarray, np.ndarray] | None = None
        self._dataset_envelope_cache_key: tuple[int, ...] | None = None
        self._stale: bool = False
        self.center_line()

    def set_envelope_percentile(self, pct: float) -> None:
        self._envelope_pct = pct
        self._dataset_envelope_cache = None

    def set_show_minmax_per_image(self, show: bool) -> None:
        self._show_minmax_per_image = show

    def set_show_envelope_per_crosshair(self, show: bool) -> None:
        self._show_envelope_per_crosshair = show

    def set_show_envelope_per_dataset(self, show: bool) -> None:
        self._show_envelope_per_dataset = show
        if not show:
            self._dataset_envelope_cache = None

    def _invalidate_dataset_envelope_cache(self) -> None:
        self._dataset_envelope_cache = None
        self._dataset_envelope_cache_key = None

    def couple(self, plot: "ExtractionPlot") -> None:
        self._extractionline.addMarker(
            'o',
            size=settings.get("viewer/intersection_point_size"),
        )
        self._extractionline.couple(plot._extractionline)
        plot._extractionline.sigPositionChanged.connect(self.draw_line)
        self._coupled = plot

    def center_line(self) -> None:
        self._extractionline.setPos(
            self._viewer.image.shape[1 if self._vert else 2] / 2
        )

    def toggle_mark_position(self) -> None:
        self._mark_coupled_position = not self._mark_coupled_position

    def change_width(self, width: int) -> None:
        self._extractionline.change_width(width)
        self.draw_line()

    def extract_data(self) -> np.ndarray | None:
        p = int(self._extractionline.value())  # type: ignore
        if not (0 <= p < self._viewer.image.shape[1 if self._vert else 2]):
            return None
        sp = slice(
            max(p - self._extractionline.width, 0),
            min(p + self._extractionline.width + 1,
                self._viewer.image.shape[1 if self._vert else 2])
        )
        if self._vert:
            image = self._viewer.now[sp, :].mean(axis=0)
        else:
            image = self._viewer.now[:, sp].mean(axis=1)
        return image

    def _extract_slice_raw(self) -> np.ndarray | None:
        """Return the raw 2D slice (no averaging) for envelope computation."""
        return self._extract_slice_raw_for_frame(self._viewer.currentIndex)

    def _extract_slice_raw_for_frame(
        self, frame_index: int
    ) -> np.ndarray | None:
        """Return the raw 2D slice for a given frame."""
        p = int(self._extractionline.value())  # type: ignore
        if not (0 <= p < self._viewer.image.shape[1 if self._vert else 2]):
            return None
        sp = slice(
            max(p - self._extractionline.width, 0),
            min(p + self._extractionline.width + 1,
                self._viewer.image.shape[1 if self._vert else 2])
        )
        frame = self._viewer.image[frame_index, ...]
        if self._vert:
            return frame[sp, :]  # (line_width, width)
        return frame[:, sp]  # (height, line_width)

    def _extract_slice_params(self) -> tuple[int, slice] | None:
        p = int(self._extractionline.value())  # type: ignore
        if not (0 <= p < self._viewer.image.shape[1 if self._vert else 2]):
            return None
        sp = slice(
            max(p - self._extractionline.width, 0),
            min(p + self._extractionline.width + 1,
                self._viewer.image.shape[1 if self._vert else 2])
        )
        return p, sp

    def _extract_data_for_frame(self, frame_index: int) -> np.ndarray | None:
        params = self._extract_slice_params()
        if params is None:
            return None
        _, sp = params
        frame = self._viewer.image[frame_index, ...]
        if self._vert:
            return frame[sp, :].mean(axis=0).squeeze()
        return frame[:, sp].mean(axis=1).squeeze()

    def _compute_full_axis_envelope(
        self,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Min/max (or percentile) per position over FULL axis.
        Horizontal: at each x, take all y values -> min(x), max(x).
        Vertical: at each y, take all x values -> min(y), max(y).
        """
        frame = self._viewer.now
        if self._vert:
            # v_plot: profile along y -> min/max at each row -> reduce axis 0 (cols)
            axis_reduce = (0, 2) if frame.ndim == 3 else 0
            if self._envelope_pct <= 0:
                min_curve = np.min(frame, axis=axis_reduce)
                max_curve = np.max(frame, axis=axis_reduce)
            else:
                min_curve = np.percentile(frame, self._envelope_pct, axis=axis_reduce)
                max_curve = np.percentile(frame, 100 - self._envelope_pct, axis=axis_reduce)
        else:
            # h_plot: profile along x -> min/max at each column -> reduce axis 1 (rows)
            axis_reduce = (1, 2) if frame.ndim == 3 else 1
            if self._envelope_pct <= 0:
                min_curve = np.min(frame, axis=axis_reduce)
                max_curve = np.max(frame, axis=axis_reduce)
            else:
                min_curve = np.percentile(frame, self._envelope_pct, axis=axis_reduce)
                max_curve = np.percentile(frame, 100 - self._envelope_pct, axis=axis_reduce)
        return min_curve, max_curve

    def _compute_crosshair_envelope(
        self, slice_raw: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Envelope per position across crosshair thickness.
        pct=0: min/max. pct>0: percentile over line_width pixels.
        """
        line_width_axis = 0 if self._vert else 1
        if slice_raw.ndim == 3:
            axis = (line_width_axis, 2)
        else:
            axis = line_width_axis
        if self._envelope_pct <= 0:
            min_curve = np.min(slice_raw, axis=axis).squeeze()
            max_curve = np.max(slice_raw, axis=axis).squeeze()
        else:
            min_curve = np.percentile(slice_raw, self._envelope_pct, axis=axis).squeeze()
            max_curve = np.percentile(slice_raw, 100 - self._envelope_pct, axis=axis).squeeze()
        return min_curve, max_curve

    def _compute_dataset_envelope(self) -> tuple[np.ndarray, np.ndarray] | None:
        params = self._extract_slice_params()
        if params is None:
            return None
        p, sp = params
        width = self._extractionline.width
        shape = self._viewer.image.shape
        cache_key = (
            p, width, shape[0], shape[1], shape[2], self._envelope_pct, self._vert,
        )
        if (self._dataset_envelope_cache is not None
                and self._dataset_envelope_cache_key == cache_key):
            return self._dataset_envelope_cache
        n_frames = shape[0]
        if self._envelope_pct <= 0:
            slices_raw = []
            for i in range(n_frames):
                s = self._extract_slice_raw_for_frame(i)
                if s is not None:
                    slices_raw.append(s)
            if not slices_raw:
                return None
            stack = np.stack(slices_raw)  # (n_frames, ...)
            if self._vert:
                axis = (0, 1)  # reduce frames and line_width, keep width
            else:
                axis = (0, 2)  # reduce frames and line_width, keep height
            if stack.ndim == 4:
                axis = axis + (3,)  # also reduce over RGB
            min_curve = np.min(stack, axis=axis).squeeze()
            max_curve = np.max(stack, axis=axis).squeeze()
        else:
            extractions = []
            for i in range(n_frames):
                ext = self._extract_data_for_frame(i)
                if ext is not None:
                    extractions.append(ext)
            if not extractions:
                return None
            stack = np.stack(extractions)
            p_lo = self._envelope_pct
            p_hi = 100.0 - p_lo
            min_curve = np.percentile(stack, p_lo, axis=0)
            max_curve = np.percentile(stack, p_hi, axis=0)
        self._dataset_envelope_cache = (min_curve, max_curve)
        self._dataset_envelope_cache_key = cache_key
        return self._dataset_envelope_cache

    def showEvent(self, event: QShowEvent) -> None:
        super().showEvent(event)
        if self._stale:
            self._stale = False
            self.draw_line()

    def draw_line(self) -> None:
        if not self.isVisible():
            self._stale = True
            return
        self.clear()
        if (image := self.extract_data()) is not None:
            self.plot(image)
            self.draw_indicator(image)
            self._draw_envelope_lines(image)
            ds_env = self._compute_dataset_envelope() if self._show_envelope_per_dataset else None
            self._fit_unlinked_axis(image, ds_env)

    def plot(
        self,
        image: np.ndarray,
        x_values: Optional[np.ndarray] = None,
        **kwargs,
    ) -> None:
        if x_values is None:
            x_values = np.arange(image.shape[0]) + 0.5
        image = np.atleast_1d(image.squeeze())
        x_values = np.atleast_1d(x_values.squeeze())
        self.plot_x_y(x_values, image, **kwargs)

    def plot_x_y(self, x: np.ndarray, y: np.ndarray, **kwargs) -> None:
        if y.ndim == 2:
            kwargs.pop("pen", "")
            if self._vert:
                self.plotItem.plot(y[:, 0], x, pen='r', **kwargs)
                self.plotItem.plot(y[:, 1], x, pen='g', **kwargs)
                self.plotItem.plot(y[:, 2], x, pen='b', **kwargs)
            else:
                self.plotItem.plot(x, y[:, 0], pen='r', **kwargs)
                self.plotItem.plot(x, y[:, 1], pen='g', **kwargs)
                self.plotItem.plot(x, y[:, 2], pen='b', **kwargs)
        else:
            pen = kwargs.pop("pen", "gray")
            if self._vert:
                self.plotItem.plot(y, x, pen=pen, **kwargs)
            else:
                self.plotItem.plot(x, y, pen=pen, **kwargs)

    def _draw_envelope_lines(self, image: np.ndarray) -> None:
        n = image.shape[0]
        x_values = np.arange(n) + 0.5
        if self._show_minmax_per_image:
            min_curve, max_curve = self._compute_full_axis_envelope()
            min_curve = np.atleast_1d(min_curve.squeeze())
            max_curve = np.atleast_1d(max_curve.squeeze())
            if min_curve.ndim == 2:
                min_curve = min_curve.mean(axis=1)
                max_curve = max_curve.mean(axis=1)
            nx = min_curve.shape[0]
            x_vals = np.arange(nx) + 0.5
            self.plot_x_y(x_vals, min_curve, pen=_PEN_MIN_PER_IMAGE)
            self.plot_x_y(x_vals, max_curve, pen=_PEN_MAX_PER_IMAGE)
        if self._show_envelope_per_crosshair:
            slice_raw = self._extract_slice_raw()
            if slice_raw is not None:
                min_curve, max_curve = self._compute_crosshair_envelope(slice_raw)
                min_curve = np.atleast_1d(min_curve.squeeze())
                max_curve = np.atleast_1d(max_curve.squeeze())
                if min_curve.ndim == 2:
                    min_curve = min_curve.mean(axis=1)
                    max_curve = max_curve.mean(axis=1)
                nx = min_curve.shape[0]
                x_vals = np.arange(nx) + 0.5
                self.plot_x_y(x_vals, min_curve, pen=_PEN_MIN_PER_CROSSHAIR)
                self.plot_x_y(x_vals, max_curve, pen=_PEN_MAX_PER_CROSSHAIR)
        if self._show_envelope_per_dataset:
            result = self._compute_dataset_envelope()
            if result is not None:
                min_curve, max_curve = result
                min_curve = np.atleast_1d(min_curve.squeeze())
                max_curve = np.atleast_1d(max_curve.squeeze())
                if min_curve.ndim == 2:
                    min_curve = min_curve.mean(axis=1)
                    max_curve = max_curve.mean(axis=1)
                n = min_curve.shape[0]
                x_values = np.arange(n) + 0.5
                self.plot_x_y(x_values, min_curve, pen=_PEN_MIN_PER_DATASET)
                self.plot_x_y(x_values, max_curve, pen=_PEN_MAX_PER_DATASET)

    def _fit_unlinked_axis(
        self,
        image: np.ndarray,
        dataset_envelope: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> None:
        # Explicitly set range of unlinked axis to data bounds to prevent
        # cumulative zoom-out (pyqtgraph bug with setXLink/setYLink + autoRange)
        val_min = float(np.min(image))
        val_max = float(np.max(image))
        if dataset_envelope is not None:
            val_min = min(val_min, float(np.min(dataset_envelope[0])))
            val_max = max(val_max, float(np.max(dataset_envelope[1])))
        if self._show_minmax_per_image:
            fe = self._compute_full_axis_envelope()
            if fe[0] is not None:
                val_min = min(val_min, float(np.min(fe[0])))
                val_max = max(val_max, float(np.max(fe[1])))
        if self._show_envelope_per_crosshair:
            slice_raw = self._extract_slice_raw()
            if slice_raw is not None:
                fe_min, fe_max = self._compute_crosshair_envelope(slice_raw)
                val_min = min(val_min, float(np.min(fe_min)))
                val_max = max(val_max, float(np.max(fe_max)))
        if np.isnan(val_min) or np.isnan(val_max):
            return
        if val_min == val_max:
            val_min -= 0.5
            val_max += 0.5
        pad = 0.02 * (val_max - val_min) or 1.0
        vb = self.plotItem.getViewBox()
        if self._vert:
            vb.setXRange(val_min - pad, val_max + pad, padding=0)
        else:
            vb.setYRange(val_min - pad, val_max + pad, padding=0)

    def draw_indicator(self, image: np.ndarray) -> None:
        if self._coupled is None or not self._mark_coupled_position:
            return
        # Use data bounds instead of viewRange() to avoid feedback loop
        # where indicator spans current view -> autoRange expands -> next
        # indicator spans larger view -> repeated zoom-out on unlinked axis
        val_min = float(np.min(image))
        val_max = float(np.max(image))
        if val_min == val_max:
            val_min -= 0.5
            val_max += 0.5
        coupled_val = self._coupled._extractionline.value()
        pen = pg.mkPen("r", style=Qt.PenStyle.DashLine)
        if self._vert:
            self.plotItem.plot(
                np.array([val_min, val_max]),
                np.array([coupled_val, coupled_val]),
                pen=pen,
            )
        else:
            self.plotItem.plot(
                np.array([coupled_val, coupled_val]),
                np.array([val_min, val_max]),
                pen=pen,
            )

    def get_translated_pos(self, x: int, y: int) -> tuple[int, int]:
        return (x, y) if not self._vert else (y, x)

    def toggle_line(self) -> None:
        self._extractionline.toggle()
