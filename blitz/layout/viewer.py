from typing import Optional

import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QMouseEvent
from PyQt5.QtWidgets import QLabel
from pyqtgraph import RectROI, mkPen

from ..data.image import ImageData
from ..data.load import from_file
from ..tools import format_pixel_value, wrap_text


class ImageViewer(pg.ImageView):

    def __init__(
        self,
        h_plot: pg.PlotWidget,
        v_plot: pg.PlotWidget,
        info_label: QLabel,
        common_size: float,
    ) -> None:
        view = pg.PlotItem()
        view.showGrid(x=True, y=True, alpha=0.4)
        super().__init__(view=view)

        self.ui.graphicsView.setBackground(pg.mkBrush(20, 20, 20))

        self.ui.roiBtn.setChecked(True)
        self.roiClicked()
        self.ui.histogram.setMinimumWidth(220)
        self.ui.roiPlot.setFixedHeight(common_size)
        self.ui.roiPlot.plotItem.showGrid(  # type: ignore
            x=True, y=True, alpha=0.6,
        )
        self.ui.histogram.gradient.loadPreset('greyclip')

        self.roi_viewer = ImageViewerROI(
            [[0,0], [0,20], [10, 10]],
            self,
            closed=True,
        )
        self.view.addItem(self.roi_viewer)

        self.data = ImageData()
        self.mask: None | RectROI = None
        self.pixel_value: Optional[np.ndarray] = None

        self.scene.sigMouseMoved.connect(self.mouseMovedScene)
        self.timeLine.sigPositionChanged.connect(self.mouse_moved_timeline)
        self.ui.roiPlot.mousePressEvent = self.on_roi_plot_clicked

        self.info_label = info_label
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

    def load_data(self, filepath: Optional[str] = None, **kwargs) -> None:
        if filepath is None:
            self.data.set(*from_file())
        else:
            self.data.set(*from_file(filepath, **kwargs))

        self.show_image(self.data.image)
        self.ui.roiPlot.plotItem.vb.autoRange()
        self.init_roi_and_crosshair()

    def init_roi_and_crosshair(self) -> None:
        height = self.data.image.shape[2]
        width = self.data.image.shape[1]
        roi_width = max(int(0.02 * width),1)
        roi_height = max(int(0.02 * height),1)
        x_pos = (width - roi_width) / 2
        y_pos = (height - roi_height) / 2
        self.roi_viewer.setPos([x_pos, y_pos])
        self.roi_viewer.setSize([roi_width, roi_height])
        self.crosshair_vline.setPos(width / 2)
        self.crosshair_hline.setPos(height / 2)

    def manipulation(self, operation: str) -> None:
        match operation:
            case 'Min':
                self.setImage(self.data.min)
            case 'Max':
                self.setImage(self.data.max)
            case 'Mean':
                self.setImage(self.data.mean)
            case 'STD':
                self.setImage(self.data.std)
            case 'Org':
                self.setImage(self.data.image)
            case ('rotate', 'flip_x', 'flip_y'):
                getattr(self.data, operation)()
                self.setImage(
                    self.data.image,
                    autoRange=False,
                    autoLevels=False,
                    autoHistogramRange=False,
                )
            case _:
                print("Operation not implemented")

    def new_mask(self) -> None:
        img = self.getImageItem().image
        width, height = img.shape[0], img.shape[1]  # type: ignore
        self.mask = RectROI([0, 0], [width, height], pen=(0,9))
        self.mask.addScaleHandle([0, 0], [1, 1])
        self.mask.addScaleHandle([1, 1], [0, 0])
        self.mask.addScaleHandle([0, 1], [1, 0])
        self.mask.addScaleHandle([1, 0], [0, 1])
        self.view.addItem(self.mask)

    def reset(self) -> None:
        self.data.reset()
        if self.mask is not None:
            self.view.removeItem(self.mask)
        self.show_image(self.data.image, autoRange=True)
        self.autoRange()
        self.autoLevels()
        self.autoHistogramRange()
        self.init_roi_and_crosshair()

    def apply_mask(self) -> None:
        self.data.mask(self.mask)
        self.view.removeItem(self.mask)
        self.setImage(self.data.image, autoRange=True)
        self.init_roi_and_crosshair()

    def toggle_mask(self) -> None:
        if self.mask is None:
            img = self.getImageItem().image
            width, height = img.shape[0], img.shape[1]  # type: ignore
            self.mask = RectROI([0, 0], [width, height], pen=(0,9))
            self.mask.addScaleHandle([0, 0], [1, 1])
            self.mask.addScaleHandle([1, 1], [0, 0])
            self.mask.addScaleHandle([0, 1], [1, 0])
            self.mask.addScaleHandle([1, 0], [0, 1])
            self.view.addItem(self.mask)
        else:
            self.view.removeItem(self.mask)
            self.data.reset()
            self.setImage(self.data.image)

    def show_image(self, image: np.ndarray, **kwargs) -> None:
        self.image = image
        self.setImage(image, **kwargs)

    def mouseMovedScene(self, pos: tuple[float, float]) -> None:
        img_coords = self.view.vb.mapSceneToView(pos)
        self.x_, self.y_ = int(img_coords.x()), int(img_coords.y())

        if (0 <= self.x_ < self.data.image.shape[0]
                and 0 <= self.y_ < self.data.image.shape[1]):
            self.pixel_value = self.data.image[self.x_, self.y_]
        else:
            self.pixel_value = None
        self.update_position_label()

    def on_roi_plot_clicked(self, ev) -> None:
        if isinstance(ev, QMouseEvent):
            if ev.button() == Qt.MouseButton.MiddleButton:
                x_pos = self.ui.roiPlot.plotItem.vb.mapSceneToView(
                    ev.pos()
                ).x()
                index = int(x_pos)
                index = max(0, min(index, self.data.image.shape[0]-1))
                self.setCurrentIndex(index)
            else:
                pg.PlotWidget.mousePressEvent(self.ui.roiPlot, ev)

    def mouse_moved_timeline(self) -> None:
        self.update_position_label()

    def update_position_label(self) -> None:
        self.current_image = int(self.currentIndex)
        self.current_image_name = self.data.meta[self.current_image].get(
            'file_name', str(self.current_image)
        )
        pixel_text = format_pixel_value(self.pixel_value)
        current_image_name_wrapped = wrap_text(self.current_image_name, 40)

        text = (
            f"|X:{self.x_:4d} Y:{self.y_:4d}|\n"
            f"|{pixel_text}|\n"
            f"|Frame:{self.current_image:4d}"
            f"/{self.data.image.shape[0]-1:4d}|\n"
            f"|Name: {current_image_name_wrapped}|"
        )

        self.info_label.setText(text)

    def setup_connections(self) -> None:
        self.crosshair_vline.sigPositionChanged.connect(self.update_plots)
        self.crosshair_hline.sigPositionChanged.connect(self.update_plots)
        self.items_added = False

    def toggle_crosshair(self):
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

    def update_plots(self):
        x = int(self.crosshair_vline.getPos()[0])
        y = int(self.crosshair_hline.getPos()[1])
        frame_idx = int(self.timeLine.value())  # type: ignore

        self.v_plot.clear()
        self.h_plot.clear()
        self.plot_data(frame_idx, x, y)

    def plot_data(self, frame_idx: int, x: int, y: int) -> None:
        frame_max = self.data.image.shape[0] - 1
        x_max = self.data.image.shape[1] - 1
        y_max = self.data.image.shape[2] - 1
        if frame_idx > frame_max:
            print("Frame index out of bounds, skipping plotting.")
            return

        x_constrained = min(max(0, x), x_max)
        y_constrained = min(max(0, y), y_max)

        if x_constrained != x or y_constrained != y:
            print("Indices out of bounds, skipping plotting.")
            return

        x_values = np.arange(self.data.image.shape[2])

        if (len(self.data.image.shape) == 4
                and self.data.image.shape[3] == 3):
            # colored image
            r_data_v = self.data.image[frame_idx, x, :, 0]
            g_data_v = self.data.image[frame_idx, x, :, 1]
            b_data_v = self.data.image[frame_idx, x, :, 2]

            self.v_plot.plot(r_data_v, x_values, pen='r')
            self.v_plot.plot(g_data_v, x_values, pen='g')
            self.v_plot.plot(b_data_v, x_values, pen='b')

            r_data_h = self.data.image[frame_idx, :, y, 0]
            g_data_h = self.data.image[frame_idx, :, y, 1]
            b_data_h = self.data.image[frame_idx, :, y, 2]

            self.h_plot.plot(r_data_h, pen='r')
            self.h_plot.plot(g_data_h, pen='g')
            self.h_plot.plot(b_data_h, pen='b')
        else:
            # grayscale image
            v_data = self.data.image[frame_idx, x, :]
            h_data = self.data.image[frame_idx, :, y]
            self.v_plot.plot(v_data, x_values, pen='gray')
            self.h_plot.plot(h_data, pen='gray')


class ImageViewerROI(pg.PolyLineROI):

    def __init__(
        self,
        positions: list[list[int]],
        image_viewer: ImageViewer,
        closed: bool = False,
        pos=None,
        **kwargs,
    ) -> None:
        super().__init__(positions, closed, pos, **kwargs)
        self.image_viewer = image_viewer
        self.setPen(color=(128, 128, 0, 100), width=3)
        self.line_labels = []
        points = self.getHandles()
        for i in range(len(points)):
            self.create_label(i, points)
        self.angle_labels = []
        self._visible = False

    def toggle(self) -> None:
        self._visible = not self._visible
        self.setVisible(self._visible)
        for label in self.line_labels:
            label.setVisible(self._visible)
        for label in self.angle_labels:
            label.setVisible(self._visible)
        if self._visible:
            self.update_line_labels_and_angles()

    def convert_to_view_coords(self, point) -> pg.ViewBox:
        print(point)
        return self.mapToView(point)  # type: ignore

    def midpoint(self, p1, p2) -> tuple[float, float]:
        view_coords_p1 = self.convert_to_view_coords(p1)
        view_coords_p2 = self.convert_to_view_coords(p2)
        # return (
        #     (view_coords_p1.x() + view_coords_p2.x()) / 2,
        #     (view_coords_p1.y() + view_coords_p2.y()) / 2,
        # )
        return (0, 10)

    def create_label(self, i: int, points: list) -> None:
        p1 = points[i].pos()
        p2 = points[(i + 1) % len(points)].pos()
        mid = self.midpoint(p1, p2)

        # Calculate the length of the line segment
        length = np.sqrt((p2.x() - p1.x())**2 + (p2.y() - p1.y())**2)

        # Set the length as the text of the label
        label = pg.TextItem("{:.2f}".format(length))

        label.setPos(mid[0], mid[1])
        self.image_viewer.getView().addItem(label)
        self.line_labels.append(label)

    def angle_between_lines(self, p0, p1, p2) -> float:
        angle1 = np.arctan2(p1.y() - p0.y(), p1.x() - p0.x())
        angle2 = np.arctan2(p2.y() - p0.y(), p2.x() - p0.x())
        angle = np.degrees(angle1 - angle2)
        angle = abs(angle) % 360
        if angle > 180:
            angle = 360 - angle
        return angle

    def update_line_labels_and_angles(
        self,
        show_in_mm: bool = False,
        pixels: float | int = 1,
        in_mm: float | int = 1,
    ) -> None:
        points = self.getHandles()
        n = len(points)

        if not hasattr(self, 'angle_labels'):
            self.angle_labels = []
        if not hasattr(self, 'angles'):
            self.angles = []

        for i in range(len(points)):
            p1 = points[i].pos()
            p2 = points[(i + 1) % len(points)].pos()
            mid = self.midpoint(p1, p2)

            # recalculate the length of the line segment
            length = np.sqrt((p2.x() - p1.x())**2 + (p2.y() - p1.y())**2)
            if show_in_mm:
                length = length * in_mm / pixels
            else:
                length = length

            if i < len(self.line_labels):
                self.line_labels[i].setPos(mid[0], mid[1])
                self.line_labels[i].setText("{:.2f}".format(length))
            else:
                self.create_label(i, points)

        while len(self.line_labels) > len(points):
            label = self.line_labels.pop()
            self.image_viewer.getView().removeItem(label)

        for i in range(n):
            p0 = points[i].pos()
            p1 = points[(i - 1) % n].pos()
            p2 = points[(i + 1) % n].pos()

            angle = self.angle_between_lines(p0, p1, p2)

            if i < len(self.angle_labels):
                self.angle_labels[i].setPos(
                    self.convert_to_view_coords(p0).x(),
                    self.convert_to_view_coords(p0).y(),
                )
                self.angle_labels[i].setText(f"{angle:.2f}°")
            else:
                angle_label = pg.TextItem(f"{angle:.2f}°")
                angle_label.setPos(p0.x(), p0.y())
                self.image_viewer.getView().addItem(angle_label)
                self.angle_labels.append(angle_label)

            self.angles.append(angle)

        while len(self.angle_labels) > n:
            label = self.angle_labels.pop()
            self.image_viewer.getView().removeItem(label)
