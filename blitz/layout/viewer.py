from pathlib import Path
from typing import Any, Optional

import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QDropEvent, QMouseEvent
from pyqtgraph import RectROI, mkPen

from .. import settings
from ..data.load import DataLoader, ImageData
from ..tools import format_pixel_value, log, wrap_text


class ImageViewer(pg.ImageView):

    image: np.ndarray

    AVAILABLE_OPERATIONS = {
        "All Images": "org",
        "Minimum": "min",
        "Maximum": "max",
        "Mean": "mean",
        "Standard Deviation": "std",
    }

    file_dropped = pyqtSignal(str)

    def __init__(
        self,
        h_plot: pg.PlotWidget,
        v_plot: pg.PlotWidget,
    ) -> None:
        view = pg.PlotItem()
        view.showGrid(x=True, y=True, alpha=0.4)
        roi = pg.ROI(pos=(0, 0), size=10)  # type: ignore
        roi.handleSize = 9
        roi.addScaleHandle([1, 1], [0, 0])
        roi.addRotateHandle([0, 0], [0.5, 0.5])
        super().__init__(view=view, roi=roi)
        self.ui.graphicsView.setBackground(pg.mkBrush(20, 20, 20))

        self.ui.roiBtn.setChecked(True)
        self.roiClicked()
        self.ui.histogram.setMinimumWidth(220)
        self.ui.roiPlot.plotItem.showGrid(  # type: ignore
            x=True, y=True, alpha=0.6,
        )
        self.ui.histogram.gradient.loadPreset('greyclip')

        self.measure_roi = MeasureROI(self.view)

        self.mask: None | RectROI = None
        self.pixel_value: Optional[np.ndarray] = None

        self.h_plot = h_plot
        self.v_plot = v_plot

        self.crosshair_vline = pg.InfiniteLine(angle=90, movable=True)
        self.crosshair_hline = pg.InfiniteLine(angle=0, movable=True)
        self.view.addItem(self.crosshair_vline)
        self.view.addItem(self.crosshair_hline)
        self.pen = mkPen(
            color=(200, 200, 200, 140),
            style=Qt.PenStyle.DashDotDotLine,
            width=1,
        )
        self.setup_connections()
        self.crosshair_state = False
        self.toggle_crosshair()

        self.ui.roiPlot.mousePressEvent = self.on_roi_plot_clicked
        self.roi.sigRegionChanged.connect(
            lambda: self.ui.roiPlot.plotItem.vb.autoRange()  # type: ignore
        )
        self.roi.sigRegionChangeFinished.connect(
            lambda: self.ui.roiPlot.plotItem.vb.autoRange()  # type: ignore
        )
        self.setAcceptDrops(True)
        self._fit_levels = True
        self._background_image: ImageData | None = None
        self.load_data()

    def dragEnterEvent(self, e: QDropEvent):
        if e.mimeData().hasUrls():
            e.accept()
        else:
            e.ignore()

    def dropEvent(self, e: QDropEvent) -> None:
        file_path = e.mimeData().urls()[0].toLocalFile()
        self.file_dropped.emit(file_path)

    def toggle_fit_levels(self) -> None:
        self._fit_levels = not self._fit_levels

    def load_data(self, path: Optional[Path] = None, **kwargs) -> None:
        self.data = DataLoader(**kwargs).load(path)
        self.setImage(self.data.image)
        self.init_roi_and_crosshair()
        self.update_profiles()
        self.autoRange()

    def load_background_file(self, path: Path) -> bool:
        self._background_image = DataLoader().load(path)
        if not self._background_image.is_single_image():
            log("Error: Background is not a single image")
            self._background_image = None
            return False
        return True

    def unload_background_file(self) -> None:
        self._background_image = None

    def init_roi_and_crosshair(self) -> None:
        height = self.image.shape[2]
        width = self.image.shape[1]
        self.crosshair_vline.setPos(width / 2)
        self.crosshair_hline.setPos(height / 2)
        self.roi.setSize((.1*width, .1*height))
        self.roi.setPos((width*9/20, height*9/20))
        self.measure_roi.toggle()
        self.measure_roi.setPoints(
            [[0, 0], [0, 0.5*height], [0.5*width, 0.25*height]]
        )
        self.measure_roi.toggle()
        on_drop_roi_update = (
            self.data.n_images * np.prod(self.roi.size())
            > settings.get("viewer/ROI_on_drop_threshold")
        )
        self.toggle_roi_update_frequency(on_drop_roi_update)

    def norm(
        self,
        operation: str,
        beta: float = 1.0,
        left: Optional[int] = None,
        right: Optional[int] = None,
        background: bool = False,
    ) -> None:
        self.data.normalize(
            operation=operation,  # type: ignore
            beta=beta,
            left=left,
            right=right,
            reference=self._background_image if background else None,
        )
        pos = self.timeLine.pos()
        self.setImage(
            self.data.image,
            autoRange=False,
            autoLevels=self._fit_levels,
        )
        self.timeLine.setPos(pos)
        self.ui.roiPlot.plotItem.vb.autoRange()  # type: ignore
        self.update_profiles()

    def reduce(self, operation: str) -> None:
        match operation:
            case 'min' | 'max' | 'mean' | 'std':
                self.data.reduce(operation)
            case 'org':
                self.data.unravel()
            case _:
                log("Operation not implemented")
                return
        self.setImage(
            self.data.image,
            autoRange=False,
            autoLevels=self._fit_levels,
        )
        self.init_roi_and_crosshair()
        self.update_profiles()

    def manipulate(self, operation: str) -> None:
        if operation in ['transpose', 'flip_x', 'flip_y']:
            getattr(self.data, operation)()
        else:
            log("Operation not implemented")
            return
        pos = self.timeLine.pos()
        self.setImage(
            self.data.image,
            autoRange=False,
            autoLevels=self._fit_levels,
        )
        self.timeLine.setPos(pos)
        self.init_roi_and_crosshair()
        self.update_profiles()

    def reset(self) -> None:
        self.data.reset()
        self.setImage(
            self.data.image,
            autoRange=True,
            autoLevels=self._fit_levels,
        )
        self.init_roi_and_crosshair()

    def apply_mask(self) -> None:
        if self.mask is None:
            return
        self.data.mask(self.mask)
        self.toggle_mask()
        pos = self.timeLine.pos()
        self.setImage(
            self.data.image,
            autoRange=True,
            autoLevels=self._fit_levels,
        )
        self.timeLine.setPos(pos)
        self.init_roi_and_crosshair()

    def toggle_mask(self) -> None:
        if self.mask is None:
            img = self.getImageItem().image
            width, height = img.shape[0], img.shape[1]  # type: ignore
            self.mask = RectROI((0, 0), (width, height), pen=(0, 9))
            self.mask.handleSize = 10
            self.mask.addScaleHandle((0, 0), (1, 1))
            self.mask.addScaleHandle((1, 1), (0, 0))
            self.mask.addScaleHandle((0, 1), (1, 0))
            self.mask.addScaleHandle((1, 0), (0, 1))
            self.view.addItem(self.mask)
            # removeHandle has to be called after adding mask to view
            self.mask.removeHandle(0)
        else:
            self.view.removeItem(self.mask)
            self.mask = None

    def toggle_roi_update_frequency(
        self,
        on_drop: Optional[bool] = None,
    ) -> None:
        if self.roi.receivers(self.roi.sigRegionChanged) > 1 and (
            (on_drop is not None and on_drop) or (on_drop is None)
        ):
            self.roi.sigRegionChanged.disconnect()
            self.roi.sigRegionChangeFinished.connect(self.roiChanged)
            self.roi.sigRegionChangeFinished.connect(
                lambda: self.ui.roiPlot.plotItem.vb.autoRange()  # type: ignore
            )
        elif self.roi.receivers(self.roi.sigRegionChangeFinished) > 1 and (
            (on_drop is not None and not on_drop) or (on_drop is None)
        ):
            self.roi.sigRegionChangeFinished.disconnect()
            self.roi.sigRegionChanged.connect(self.roiChanged)
            self.roi.sigRegionChanged.connect(
                lambda: self.ui.roiPlot.plotItem.vb.autoRange()  # type: ignore
            )

    def is_roi_on_drop_update(self) -> bool:
        return self.roi.receivers(self.roi.sigRegionChangeFinished) > 1

    def get_position_info(
        self,
        pos: tuple[int, int],
    ) -> tuple[float, float, str | None]:
        img_coords = self.view.vb.mapSceneToView(pos)
        x, y = int(img_coords.x()), int(img_coords.y())
        if (0 <= x < self.image.shape[-2] and 0 <= y < self.image.shape[-1]):
            pixel_value = self.image[self.currentIndex, x, y]
        else:
            pixel_value = None
        return x, y, format_pixel_value(pixel_value)

    def get_frame_info(self) -> tuple[int, int, str]:
        current_image = int(self.currentIndex)
        name = wrap_text(
            self.data.meta[self.currentIndex].get(
                'file_name', str(current_image)
            ),
            max_length=40,
        )
        return current_image, self.image.shape[0]-1, name

    def on_roi_plot_clicked(self, ev) -> None:
        if isinstance(ev, QMouseEvent):
            if ev.button() == Qt.MouseButton.MiddleButton:
                x = self.ui.roiPlot.plotItem.vb.mapSceneToView(  # type: ignore
                    ev.pos()
                ).x()
                index = int(x)
                index = max(0, min(index, self.image.shape[0]-1))
                self.setCurrentIndex(index)
            else:
                pg.PlotWidget.mousePressEvent(self.ui.roiPlot, ev)

    def load_lut_config(self, lut: dict[str, Any]) -> None:
        self.ui.histogram.restoreState(lut)

    def get_lut_config(self) -> dict[str, Any]:
        return self.ui.histogram.saveState()

    def setup_connections(self) -> None:
        self.crosshair_vline.sigPositionChanged.connect(self.update_profiles)
        self.crosshair_hline.sigPositionChanged.connect(self.update_profiles)
        self.timeLine.sigPositionChanged.connect(self.update_profiles)
        self.items_added = False

    def toggle_crosshair(self) -> None:
        self.crosshair_state = not self.crosshair_state
        if self.crosshair_state:
            if self.crosshair_vline not in self.view.items:
                self.view.addItem(self.crosshair_vline)

            if self.crosshair_hline not in self.view.items:
                self.view.addItem(self.crosshair_hline)

            self.crosshair_hline.setPen(self.pen)
            self.crosshair_vline.setPen(self.pen)
            self.crosshair_vline.setMovable(True)
            self.crosshair_hline.setMovable(True)
        else:
            self.crosshair_vline.setMovable(False)
            self.crosshair_hline.setMovable(False)
            self.view.removeItem(self.crosshair_hline)
            self.view.removeItem(self.crosshair_vline)

    def update_profiles(self) -> None:
        x = int(self.crosshair_vline.getPos()[0])
        y = int(self.crosshair_hline.getPos()[1])
        frame_idx = int(self.timeLine.value())  # type: ignore

        if frame_idx > self.image.shape[0] - 1:
            log("Frame index out of bounds, skipping plotting.")
            return

        x_constrained = min(max(0, x), self.image.shape[1] - 1)
        y_constrained = min(max(0, y), self.image.shape[2] - 1)

        if x_constrained != x or y_constrained != y:
            return

        x_values = np.arange(self.image.shape[2])

        self.v_plot.clear()
        self.h_plot.clear()

        if (len(self.image.shape) == 4
                and self.image.shape[3] == 3):
            # colored image
            r_data_v = self.image[frame_idx, x, :, 0]
            g_data_v = self.image[frame_idx, x, :, 1]
            b_data_v = self.image[frame_idx, x, :, 2]

            self.v_plot.plot(r_data_v, x_values, pen='r')
            self.v_plot.plot(g_data_v, x_values, pen='g')
            self.v_plot.plot(b_data_v, x_values, pen='b')

            r_data_h = self.image[frame_idx, :, y, 0]
            g_data_h = self.image[frame_idx, :, y, 1]
            b_data_h = self.image[frame_idx, :, y, 2]

            self.h_plot.plot(r_data_h, pen='r')
            self.h_plot.plot(g_data_h, pen='g')
            self.h_plot.plot(b_data_h, pen='b')
        else:
            # grayscale image
            v_data = self.image[frame_idx, x, :]
            h_data = self.image[frame_idx, :, y]
            self.v_plot.plot(v_data, x_values, pen='gray')
            self.h_plot.plot(h_data, pen='gray')


class MeasureROI(pg.PolyLineROI):

    def __init__(self, view: pg.ViewBox):
        super().__init__([[0, 0], [0, 20], [10, 10]], closed=True)
        self.handleSize = 10
        self.sigRegionChanged.connect(self.update_labels)
        self.view = view

        self.n_px: int = 1
        self.px_in_mm: float = 1
        self.show_in_mm = False

        self.line_labels = []
        self.angle_labels = []

        self.view.addItem(self)
        self.setPen(color=(128, 128, 0, 100), width=3)
        self._visible = True
        self.toggle()

    def toggle(self) -> None:
        self._visible = not self._visible
        self.setVisible(self._visible)
        for label in self.line_labels:
            label.setVisible(self._visible)
        for label in self.angle_labels:
            label.setVisible(self._visible)
        if self._visible:
            self.update_labels()

    def update_labels(self) -> None:
        self.update_angles()
        self.update_lines()

    def update_angles(self) -> None:
        positions = self.getSceneHandlePositions()

        while len(self.angle_labels) > len(positions):
            label = self.angle_labels.pop()
            self.view.removeItem(label)

        for i, (_, pos) in enumerate(positions):
            _, prev_pos = positions[(i - 1) % len(positions)]
            _, next_pos = positions[(i + 1) % len(positions)]
            angle = pg.Point(pos - next_pos).angle(
                pg.Point(pos - prev_pos)
            )
            pos = self.view.mapToView(pos)
            if i < len(self.angle_labels):
                self.angle_labels[i].setPos(pos.x(), pos.y())
                self.angle_labels[i].setText(f"{angle:.2f}°")
            else:
                angle_label = pg.TextItem(f"{angle:.2f}°")
                angle_label.setPos(pos.x(), pos.y())
                self.view.addItem(angle_label)
                self.angle_labels.append(angle_label)

    def update_lines(self) -> None:
        positions = self.getSceneHandlePositions()

        while len(self.line_labels) > len(positions):
            label = self.line_labels.pop()
            self.view.removeItem(label)

        for i, (_, pos) in enumerate(positions):
            _, next_pos = positions[(i + 1) % len(positions)]
            pos = self.view.mapToView(pos)
            next_pos = self.view.mapToView(next_pos)
            length = pg.Point(pos - next_pos).length()
            mid = (pos + next_pos) / 2
            if self.show_in_mm:
                length = length * self.px_in_mm / self.n_px
            pos = self.view.mapToView(pos)
            if i < len(self.line_labels):
                self.line_labels[i].setPos(mid.x(), mid.y())
                self.line_labels[i].setText(f"{length:.2f}")
            else:
                angle_label = pg.TextItem(f"{length:.2f}")
                angle_label.setPos(mid.x(), mid.y())
                self.view.addItem(angle_label)
                self.line_labels.append(angle_label)
