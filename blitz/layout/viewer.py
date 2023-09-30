from typing import Optional

import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QMouseEvent
from PyQt5.QtWidgets import QLabel
from pyqtgraph import RectROI

from ..data.image import ImageData
from ..tools import format_pixel_value, wrap_text


class ImageViewer(pg.ImageView):

    def __init__(self, data: ImageData, info_label: QLabel) -> None:
        view = pg.PlotItem()
        view.showGrid(x=True, y=True, alpha=0.4)
        super().__init__(view=view)

        self.ui.graphicsView.setBackground(pg.mkBrush(20, 20, 20))

        self.ui.roiBtn.setChecked(True)
        self.roiClicked()
        self.ui.histogram.setMinimumWidth(220)
        self.ui.roiPlot.plotItem.showGrid(  # type: ignore
            x=True, y=True, alpha=0.6,
        )
        self.ui.histogram.gradient.loadPreset('greyclip')

        self.poly_roi = pg.PolyLineROI([[0,0], [0,20], [10, 10]], closed=True)
        self.poly_roi.setPen(color=(128, 128, 0, 100),width = 3)
        self.view.addItem(self.poly_roi)

        self.data = data
        self.mask: None | RectROI = None

        self._last_rotate = self._last_flip_x = self._last_flip_y = False
        self.pixel_value: Optional[np.ndarray] = None

        self.scene.sigMouseMoved.connect(self.mouseMovedScene)
        self.timeLine.sigPositionChanged.connect(self.mouse_moved_timeline)
        self.ui.roiPlot.mousePressEvent = self.on_roi_plot_clicked

        self.info_label = info_label

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

    def apply_mask(self) -> None:
        self.data.mask(self.mask)
        self.view.removeItem(self.mask)
        self.setImage(self.data.image, autoRange=True)

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

    def update_data(self, rotate: bool, flip_x: bool, flip_y: bool) -> None:
        if rotate:
            self.data.rotate()
        if flip_x:
            self.data.flip_x()
        if flip_y:
            self.data.flip_y()
        self.setImage(
            self.data.image,
            autoRange=False,
            autoLevels=False,
            autoHistogramRange=False,
        )

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
