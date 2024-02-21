from pathlib import Path
from typing import Any, Optional

import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtGui import QDropEvent
from pyqtgraph import RectROI

from .. import settings
from ..data.load import DataLoader, ImageData
from ..tools import format_pixel_value, log, wrap_text


class ImageViewer(pg.ImageView):

    image: np.ndarray

    AVAILABLE_OPERATIONS = {
        "-": "org",
        "Minimum": "min",
        "Maximum": "max",
        "Mean": "mean",
        "Standard Deviation": "std",
    }

    file_dropped = pyqtSignal(str)
    image_changed = pyqtSignal()

    def __init__(self) -> None:
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

        self.mask: None | RectROI = None
        self.pixel_value: Optional[np.ndarray] = None

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

    @property
    def now(self) -> np.ndarray:
        return self.image[self.currentIndex, ...]

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
            self.timeLine.setPos(pos)
        self.init_roi()
        self.image_changed.emit()

    def toggle_fit_levels(self) -> None:
        self._fit_levels = not self._fit_levels
        if self._fit_levels:
            self.autoLevels()

    def load_data(self, path: Optional[Path] = None, **kwargs) -> None:
        self.data = DataLoader(**kwargs).load(path)
        self.setImage(self.data.image)
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

    def init_roi(self) -> None:
        height = self.image.shape[2]
        width = self.image.shape[1]
        self.roi.setSize((.1*width, .1*height))
        self.roi.setPos((width*9/20, height*9/20))
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
        force_calculation: bool = False,
    ) -> None:
        self.data.normalize(
            operation=operation,  # type: ignore
            beta=beta,
            left=left,
            right=right,
            reference=self._background_image if background else None,
            force_calculation=force_calculation,
        )
        self.setImage(
            self.data.image,
            keep_timestep=True,
            autoRange=False,
            autoLevels=self._fit_levels,
        )
        self.ui.roiPlot.plotItem.vb.autoRange()  # type: ignore

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

    def manipulate(self, operation: str) -> None:
        if operation in ['transpose', 'flip_x', 'flip_y']:
            getattr(self.data, operation)()
        else:
            log("Operation not implemented")
            return
        self.setImage(
            self.data.image,
            keep_timestep=True,
            autoRange=False,
            autoLevels=self._fit_levels,
        )

    def reset(self) -> None:
        self.data.reset()
        self.setImage(
            self.data.image,
            autoLevels=self._fit_levels,
        )

    def apply_mask(self) -> None:
        if self.mask is None:
            return
        self.data.mask(self.mask)
        self.toggle_mask()
        self.setImage(
            self.data.image,
            keep_timestep=True,
            autoLevels=self._fit_levels,
        )

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
        if (0 <= x < self.image.shape[1] and 0 <= y < self.image.shape[2]):
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

    def load_lut_config(self, lut: dict[str, Any]) -> None:
        self.ui.histogram.restoreState(lut)

    def get_lut_config(self) -> dict[str, Any]:
        return self.ui.histogram.saveState()
