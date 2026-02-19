from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import QPoint, pyqtSignal
from PyQt5.QtGui import QDropEvent
from pyqtgraph import RectROI
from pyqtgraph.graphicsItems.GradientEditorItem import Gradients

from .. import settings
from ..data.load import DataLoader, ImageData
from ..data.ops import ReduceOperation
from ..tools import fit_text, format_pixel_value, log


class ImageViewer(pg.ImageView):

    image: np.ndarray

    file_dropped = pyqtSignal(str)
    image_size_changed = pyqtSignal()
    image_changed = pyqtSignal()
    image_mask_changed = pyqtSignal()
    image_crop_changed = pyqtSignal()

    def __init__(self) -> None:
        view = pg.PlotItem()
        view.showGrid(x=True, y=True, alpha=0.4)
        self.poly_roi = pg.PolyLineROI(
            ((0, 0), (1, 0), (1, 1), (0, 1)),
            closed=True,
        )
        self.poly_roi.handleSize = 9
        self.square_roi = pg.ROI(pos=(0, 0), size=10)  # type: ignore
        self.square_roi.handleSize = 9
        self.square_roi.addScaleHandle([1, 1], [0, 0])
        self.square_roi.addRotateHandle([0, 0], [0.5, 0.5])
        super().__init__(view=view, roi=self.square_roi)
        self.poly_roi_state = self.poly_roi.getState()
        self.square_roi_state = self.square_roi.getState()
        self._timeline_aggregation = "mean"
        self._timeline_show_bands = False
        self.ui.graphicsView.setBackground(pg.mkBrush(20, 20, 20))

        self.ui.roiBtn.setChecked(True)
        self.roiClicked()
        self.ui.histogram.setMinimumWidth(220)
        self.ui.roiPlot.plotItem.showGrid(  # type: ignore
            x=True, y=True, alpha=0.6,
        )
        self.ui.histogram.gradient.loadPreset('plasma')

        self.mask: None | RectROI = None
        self.pixel_value: Optional[np.ndarray] = None

        self.square_roi.sigRegionChanged.connect(
            lambda: self.ui.roiPlot.plotItem.vb.autoRange()  # type: ignore
        )
        self.square_roi.sigRegionChangeFinished.connect(
            lambda: self.ui.roiPlot.plotItem.vb.autoRange()  # type: ignore
        )
        self.poly_roi.sigRegionChanged.connect(
            lambda: self.ui.roiPlot.plotItem.vb.autoRange()  # type: ignore
        )
        self.poly_roi.sigRegionChangeFinished.connect(
            lambda: self.ui.roiPlot.plotItem.vb.autoRange()  # type: ignore
        )
        self.setAcceptDrops(True)
        self._auto_colormap = True
        self._background_image: ImageData | None = None
        self.load_data()

    @property
    def now(self) -> np.ndarray:
        return self.image[self.currentIndex, ...]

    def set_roi(self, roi: pg.ROI) -> None:
        if self.roi is roi:
            return
        try:
            self.view.removeItem(self.roi)
            roi.sigRegionChanged.disconnect(self.roiChanged)
        except TypeError:
            pass
        self.view.addItem(roi)
        roi.sigRegionChanged.connect(self.roiChanged)
        self.roi = roi

    def change_roi(self) -> None:
        if self.roi is self.square_roi:
            self.square_roi_state = self.square_roi.state
            self.set_roi(self.poly_roi)
            self.roi.sigRegionChanged.disconnect(self.roiChanged)
            self.poly_roi.setState(self.poly_roi_state)
            self.roi.sigRegionChanged.connect(self.roiChanged)
        else:
            self.poly_roi_state = self.poly_roi.getState()
            self.square_roi.setState(self.square_roi_state)
            self.set_roi(self.square_roi)
        self.roiChanged()
        self.ui.roiPlot.plotItem.vb.autoRange()  # type: ignore

    def set_timeline_options(self, aggregation: str, show_bands: bool) -> None:
        self._timeline_aggregation = aggregation
        self._timeline_show_bands = show_bands
        self.roiChanged()

    def roiChanged(self) -> None:
        if self.image is None:
            return
        data_obj = getattr(self, "data", None)
        in_agg = (
            data_obj is not None
            and getattr(data_obj, "_redop", None) is not None
            and data_obj.image_timeline is not None
        )
        if in_agg:
            image = data_obj.image_timeline
        else:
            image = self.getProcessedImage()
        colmaj = self.imageItem.axisOrder == 'col-major'
        if colmaj:
            axes = (self.axes['x'], self.axes['y'])
        else:
            axes = (self.axes['y'], self.axes['x'])
        data, coords = self.roi.getArrayRegion(
            image.view(np.ndarray), img=self.imageItem, axes=axes,
            returnMappedCoords=True,
        )
        if data is None:
            return
        agg = np.nanmedian if self._timeline_aggregation == "median" else np.nanmean
        if self.axes['t'] is None and not in_agg:
            data_main = agg(data, axis=self.axes['y'])
            if colmaj:
                coords = coords[:, :, 0] - coords[:, 0:1, 0]
            else:
                coords = coords[:, 0, :] - coords[:, 0, 0:1]
            xvals = (coords**2).sum(axis=0) ** 0.5
            data_min = np.nanmin(data, axis=self.axes['y'])
            data_max = np.nanmax(data, axis=self.axes['y'])
        else:
            data_main = agg(data, axis=axes)
            data_min = np.nanmin(data, axis=axes) if self._timeline_show_bands else None
            data_max = np.nanmax(data, axis=axes) if self._timeline_show_bands else None
            xvals = (
                np.arange(image.shape[0]) if in_agg else self.tVals
            )
        if data_main.ndim == 1:
            plots = [(xvals, data_main, 'w')]
            if self._timeline_show_bands and (in_agg or self.axes['t'] is not None) and data_min is not None:
                plots.append((xvals, data_min, (100, 150, 100)))
                plots.append((xvals, data_max, (150, 200, 150)))
        elif data_main.ndim == 2:
            colors = 'w' if data_main.shape[1] == 1 else 'rgbw'
            plots = []
            for i in range(data_main.shape[1]):
                d = data_main[:, i]
                plots.append((xvals, d, colors[i]))
            if self._timeline_show_bands and (in_agg or self.axes['t'] is not None) and data_min is not None:
                for i in range(data_min.shape[1]):
                    plots.append((xvals, data_min[:, i], (100, 150, 100)))
                    plots.append((xvals, data_max[:, i], (150, 200, 150)))
        else:
            plots = [(xvals, data_main.squeeze(), 'w')]
        while len(plots) < len(self.roiCurves):
            c = self.roiCurves.pop()
            c.scene().removeItem(c)
        while len(plots) > len(self.roiCurves):
            self.roiCurves.append(self.ui.roiPlot.plot())
        for i in range(len(plots)):
            x, y, p = plots[i]
            self.roiCurves[i].setData(x, y, pen=p)
        if in_agg and len(xvals) > 0:
            xmin = 0.0
            xmax = float(xvals.max())
            self.timeLine.setBounds([xmin, xmax])
        self.ui.roiPlot.plotItem.vb.autoRange()  # type: ignore
        if in_agg and len(xvals) > 0:
            self.ui.roiPlot.setXRange(0.0, float(xvals.max()), padding=0)

    def image_mask(self, file_path: Optional[Path] = None, **kwargs) -> None:
        mask = DataLoader(**kwargs).load(file_path)
        self.data.image_mask(mask)
        self.setImage(
            self.data.image,
            keep_timestep=True,
            autoRange=False,
            autoLevels=False,
        )

    def dragEnterEvent(self, e: QDropEvent):
        if e.mimeData().hasUrls():
            e.accept()
        else:
            e.ignore()

    def dropEvent(self, e: QDropEvent) -> None:
        file_path = e.mimeData().urls()[0].toLocalFile()
        self.file_dropped.emit(file_path)

    def setImage(self, *args, keep_timestep: bool = False, **kwargs) -> None:
        if keep_timestep:
            pos = self.timeLine.pos()
        super().setImage(*args, **kwargs)
        if keep_timestep:
            self.timeLine.setPos(pos)  # type: ignore
        self.init_roi()
        self.image_changed.emit()

    def updateImage(self, autoHistogramRange: bool = False) -> None:
        super().updateImage(autoHistogramRange)

    def toggle_auto_colormap(self) -> None:
        self._auto_colormap = not self._auto_colormap

    def autoLevels(self) -> None:
        if self.data.is_greyscale() and self._auto_colormap:
            self.auto_colormap()
        else:
            super().autoLevels()

    def auto_colormap(self) -> None:
        if (min_ := self.image.min()) < 0 < (max_ := self.image.max()):
            max_ = max(abs(min_), max_)
            min_ = - max_
            self.ui.histogram.gradient.restoreState(Gradients['bipolar'])
            self.setLevels(min=min_, max=max_)
            self.ui.histogram.setHistogramRange(min_, max_)
        else:
            self.ui.histogram.gradient.restoreState(Gradients['plasma'])
            super().autoLevels()

    def load_data(
        self,
        path: Optional[Path] = None,
        progress_callback: Optional[Callable[[int], None]] = None,
        message_callback: Optional[Callable[[str], None]] = None,
        **kwargs,
    ) -> None:
        load_keys = {"frame_range", "step"}
        load_kwargs = {k: kwargs.pop(k) for k in list(kwargs) if k in load_keys}

        self.data = DataLoader(**kwargs).load(
            path,
            progress_callback=progress_callback,
            message_callback=message_callback,
            **load_kwargs,
        )
        self.setImage(self.data.image, autoLevels=True)
        self.autoRange()
        self.image_size_changed.emit()

    def set_image(self, img: ImageData) -> None:
        self.data = img
        self.update_image()

    def update_image(self) -> None:
        self.setImage(self.data.image)
        self.autoRange()
        self.image_size_changed.emit()

    def load_background_file(self, path: Path) -> bool:
        self._background_image = DataLoader().load(path)
        if not self._background_image.is_single_image():
            log("Error: Background is not a single image", color="red")
            self._background_image = None
            return False
        return True

    def unload_background_file(self) -> None:
        self._background_image = None

    def init_roi(self) -> None:
        height = self.image.shape[2]
        width = self.image.shape[1]
        if self.roi is self.square_roi:
            self.square_roi.setAngle(0)
            self.square_roi.setSize((.1*width, .1*height))
            self.square_roi.setPos((width*9/20, height*9/20))
            self.poly_roi_state["pos"] = pg.Point(0, 0)
            self.poly_roi_state["size"] = pg.Point(1, 1)
            self.poly_roi_state["points"] = (
                pg.Point(width*9/20, height*9/20),
                pg.Point(width*9/20+.1*width, height*9/20),
                pg.Point(width*9/20+.1*width, height*9/20+.1*height),
                pg.Point(width*9/20, height*9/20+.1*height),
            )
        else:
            self.roi.sigRegionChanged.disconnect(self.roiChanged)
            self.poly_roi.setSize((1, 1))
            self.poly_roi.setPos((0, 0))
            self.poly_roi.setPoints((
                (width*9/20, height*9/20),
                (width*9/20+.1*width, height*9/20),
                (width*9/20+.1*width, height*9/20+.1*height),
                (width*9/20, height*9/20+.1*height),
            ))
            self.roi.sigRegionChanged.connect(self.roiChanged)
            self.square_roi_state["angle"] = 0.0
            self.square_roi_state["pos"] = pg.Point(width*9/20, height*9/20)
            self.square_roi_state["size"] = pg.Point(.1*width, .1*height)
        on_drop_roi_update = (
            self.data.n_images * np.prod(self.roi.size())
            > settings.get("viewer/ROI_on_drop_threshold")
        )
        self.toggle_roi_update_frequency(on_drop_roi_update)

    def crop(self, left: int, right: int, keep: bool = False) -> None:
        self.data.crop(left, right, keep=keep)
        self.setImage(
            self.data.image,
            keep_timestep=(left < self.currentIndex < right),
            autoLevels=False,
        )
        self.ui.roiPlot.plotItem.vb.autoRange()  # type: ignore
        self.image_crop_changed.emit()

    def undo_crop(self) -> bool:
        success = self.data.undo_crop()
        if success:
            self.setImage(
                self.data.image,
                autoLevels=False,
            )
            self.ui.roiPlot.plotItem.vb.autoRange()  # type: ignore
        return success

    def unravel(self) -> None:
        self.data.unravel()
        self.update_image()
        self.setImage(
            self.data.image,
            autoRange=False,
            autoLevels=False,
        )

    def reduce(
        self,
        operation: ReduceOperation | str,
        bounds: Optional[tuple[int, int]] = None,
    ) -> None:
        self.data.reduce(operation, bounds=bounds)
        img = self.data.image
        if img.ndim == 2:
            img = img[np.newaxis, ...]
        elif img.ndim == 3 and img.shape[2] in (1, 3):
            img = img[np.newaxis, ...]
        self.setImage(
            img,
            autoRange=False,
            autoLevels=False,
        )

    def manipulate(self, operation: str) -> None:
        if operation in ['transpose', 'flip_x', 'flip_y']:
            getattr(self.data, operation)()
        else:
            raise RuntimeError(f"Operation {operation!r} not implemented")
        self.setImage(
            self.data.image,
            keep_timestep=True,
            autoRange=False,
            autoLevels=False,
        )
        if operation == 'transpose':
            self.image_size_changed.emit()

    def apply_mask(self) -> None:
        if self.mask is None:
            return
        if self.data.mask(self.mask):
            self.setImage(
                self.data.image,
                keep_timestep=True,
                autoLevels=False,
            )
            self.image_size_changed.emit()
            self.image_mask_changed.emit()
        self.toggle_mask()

    def reset_mask(self) -> None:
        if self.data.reset_mask():
            self.setImage(
                self.data.image,
                autoLevels=False,
            )
            self.image_size_changed.emit()
            self.image_mask_changed.emit()

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
        pos: Optional[tuple[int, int]] = None,
    ) -> tuple[float, float, str | None]:
        if pos is None:
            if self.ui.graphicsView.lastMousePos is not None:
                pos = self.ui.graphicsView.lastMousePos
            else:
                pos = QPoint(0, 0)
        img_coords = self.view.vb.mapSceneToView(pos)
        x, y = int(img_coords.x()), int(img_coords.y())
        if (0 <= x < self.image.shape[1] and 0 <= y < self.image.shape[2]):
            pixel_value = self.image[self.currentIndex, x, y]
        else:
            pixel_value = None
        return x, y, format_pixel_value(pixel_value)

    def get_frame_info(self) -> tuple[int, int, str]:
        current_image = int(self.currentIndex)
        name = fit_text(
            self.data.meta[self.currentIndex].file_name,
            max_length=settings.get("viewer/max_file_name_length"),
        )
        return current_image, self.image.shape[0]-1, name

    def load_lut_config(self, lut: dict[str, Any]) -> None:
        self.ui.histogram.restoreState(lut)

    def get_lut_config(self) -> dict[str, Any]:
        return self.ui.histogram.saveState()
