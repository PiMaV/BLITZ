from pathlib import Path
from typing import Any, Optional

import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox,
                             QDoubleSpinBox, QFormLayout, QFrame, QHBoxLayout,
                             QLabel, QPushButton, QSpinBox, QVBoxLayout, QWidget)

from pyqtgraph.graphicsItems.GradientEditorItem import Gradients

from ..data.load import (get_image_metadata, get_image_preview,
                         get_sample_bytes_per_pixel, get_sample_format,
                         get_sample_format_display, get_video_preview)
from ..data.image import VideoMetaData
from ..tools import get_available_ram


def _plasma_lut() -> np.ndarray:
    """256x4 RGBA LUT from plasma preset."""
    ticks = Gradients["plasma"]["ticks"]
    positions = np.array([t[0] for t in ticks])
    colors = np.array([t[1] for t in ticks], dtype=np.float64)
    indices = np.linspace(0, 1, 256)
    lut = np.column_stack([
        np.interp(indices, positions, colors[:, i]) for i in range(4)
    ]).astype(np.uint8)
    return lut


class _PreviewLoader(QThread):
    finished = pyqtSignal(object)

    def __init__(self, path: Path, size_ratio: float, grayscale: bool,
                 n_frames: int = 10, mode: str = "max", normalize: bool = True):
        super().__init__()
        self._path = path
        self._size_ratio = size_ratio
        self._grayscale = grayscale
        self._n_frames = n_frames
        self._mode = mode
        self._normalize = normalize

    def run(self):
        img = get_video_preview(
            self._path,
            n_frames=self._n_frames,
            size_ratio=self._size_ratio,
            grayscale=self._grayscale,
            mode=self._mode,
            normalize=self._normalize,
        )
        self.finished.emit(img)


class VideoLoadOptionsDialog(QDialog):
    def __init__(
        self,
        path: Path,
        metadata: VideoMetaData,
        parent: Optional[QWidget] = None,
        initial_params: Optional[dict] = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Video Loading Options")
        self._path = path
        self.metadata = metadata
        self._preview: Optional[np.ndarray] = None
        self._roi: Optional[pg.RectROI] = None
        self._preview_loader: Optional[_PreviewLoader] = None
        self._setup_ui()
        self._initial_mask_rel = initial_params.get("mask_rel") if initial_params else None
        if initial_params:
            self.spin_resize.setValue(
                int(initial_params.get("size_ratio", 1.0) * 100),
            )
            self.spin_step.setValue(initial_params.get("step", 1))
            self.chk_grayscale.setChecked(
                initial_params.get("grayscale", False),
            )
            self.chk_8bit.setChecked(
                initial_params.get("convert_to_8_bit", False),
            )
        else:
            is_gray, is_uint8 = get_sample_format(path)
            self.chk_grayscale.setChecked(is_gray)
            self.chk_8bit.setChecked(is_uint8)
        self._sample_bytes = get_sample_bytes_per_pixel(self._path)
        self._update_estimates()
        self._start_preview_load()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # Info Section
        info_layout = QFormLayout()
        self.lbl_file = QLabel(self.metadata.file_name)
        self.lbl_dims = QLabel(f"{self.metadata.size[0]} x {self.metadata.size[1]}")
        self.lbl_frames = QLabel(f"{self.metadata.frame_count}")
        self.lbl_format = QLabel(get_sample_format_display(self._path))
        self.lbl_format.setToolTip("Detected source format")
        info_layout.addRow("File:", self.lbl_file)
        info_layout.addRow("Dimensions:", self.lbl_dims)
        info_layout.addRow("Total Frames:", self.lbl_frames)
        info_layout.addRow("Source format:", self.lbl_format)
        layout.addLayout(info_layout)

        layout.addWidget(QLabel("<b>Loading Options</b>"))

        # Controls
        controls_layout = QFormLayout()

        # Frame Range
        range_layout = QHBoxLayout()
        self.spin_start = QSpinBox()
        self.spin_start.setRange(0, self.metadata.frame_count - 1)
        self.spin_start.setValue(0)
        self.spin_end = QSpinBox()
        self.spin_end.setRange(0, self.metadata.frame_count - 1)
        self.spin_end.setValue(self.metadata.frame_count - 1)
        range_layout.addWidget(QLabel("Start:"))
        range_layout.addWidget(self.spin_start)
        range_layout.addWidget(QLabel("End:"))
        range_layout.addWidget(self.spin_end)
        controls_layout.addRow("Frame Range:", range_layout)

        # Step (Skip)
        self.spin_step = QSpinBox()
        self.spin_step.setRange(1, 1000)
        self.spin_step.setValue(1)
        controls_layout.addRow("Step (skip frames):", self.spin_step)

        # Resize
        self.spin_resize = QSpinBox()
        self.spin_resize.setRange(1, 100)
        self.spin_resize.setValue(100)
        self.spin_resize.setSuffix(" %")
        controls_layout.addRow("Resize:", self.spin_resize)

        # Grayscale & 8-bit
        self.chk_grayscale = QCheckBox("Load as Grayscale")
        self.chk_grayscale.setChecked(False)
        self.chk_8bit = QCheckBox("Convert to 8 bit")
        self.chk_8bit.setChecked(False)
        controls_layout.addRow("", self.chk_grayscale)
        controls_layout.addRow("", self.chk_8bit)

        layout.addLayout(controls_layout)

        # Preview with Crop ROI
        layout.addWidget(QLabel("<b>Preview / Crop</b>"))
        preview_opts = QHBoxLayout()
        self.cmb_preview_mode = QComboBox()
        self.cmb_preview_mode.addItems(["MAX (across frames)", "Single frame (center)"])
        self.cmb_preview_mode.setToolTip("MAX: maximum value per pixel across frames")
        preview_opts.addWidget(QLabel("Mode:"))
        preview_opts.addWidget(self.cmb_preview_mode)
        self.chk_preview_norm = QCheckBox("Normalize")
        self.chk_preview_norm.setChecked(True)
        self.chk_preview_norm.setToolTip("Min-max stretch for better visibility")
        preview_opts.addWidget(self.chk_preview_norm)
        self.btn_preview_reload = QPushButton("Refresh")
        self.btn_preview_reload.pressed.connect(self._start_preview_load)
        self.btn_roi_reset = QPushButton("Reset ROI")
        self.btn_roi_reset.setToolTip("Reset crop to full frame")
        self.btn_roi_reset.pressed.connect(self._reset_roi)
        preview_opts.addWidget(self.btn_preview_reload)
        preview_opts.addWidget(self.btn_roi_reset)
        preview_opts.addStretch()
        layout.addLayout(preview_opts)
        preview_frame = QFrame()
        preview_frame.setFrameStyle(QFrame.StyledPanel)
        preview_layout = QVBoxLayout(preview_frame)
        self.lbl_preview_status = QLabel("Loading preview...")
        preview_layout.addWidget(self.lbl_preview_status)
        self._plot_widget = pg.PlotWidget(background=(80, 80, 80))
        self._plot_widget.setMinimumSize(400, 360)
        self._plot_widget.setAspectLocked(False)
        self._plot_widget.hideAxis("left")
        self._plot_widget.hideAxis("bottom")
        self._img_item = pg.ImageItem()
        self._plot_widget.addItem(self._img_item)
        preview_layout.addWidget(self._plot_widget)
        layout.addWidget(preview_frame)

        # Estimates
        self.lbl_ram_usage = QLabel("Estimated RAM: Calculating...")
        self.lbl_ram_available = QLabel(f"Available RAM: {get_available_ram():.2f} GB")
        layout.addWidget(self.lbl_ram_usage)
        layout.addWidget(self.lbl_ram_available)

        # Buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        # Connections for live updates
        self.spin_start.valueChanged.connect(self._update_estimates)
        self.spin_end.valueChanged.connect(self._update_estimates)
        self.spin_step.valueChanged.connect(self._update_estimates)
        self.spin_resize.valueChanged.connect(self._update_estimates)
        self.chk_grayscale.toggled.connect(self._update_estimates)
        self.chk_grayscale.toggled.connect(self._on_grayscale_changed)
        self.chk_8bit.toggled.connect(self._update_estimates)

    def _on_grayscale_changed(self):
        if self._preview_loader is None:
            self._start_preview_load()

    def _start_preview_load(self):
        if self._preview_loader is not None:
            return
        size_ratio = self.spin_resize.value() / 100.0
        grayscale = self.chk_grayscale.isChecked()
        n_frames = min(10, max(1, self.metadata.frame_count))
        mode = "max" if self.cmb_preview_mode.currentIndex() == 0 else "single"
        normalize = self.chk_preview_norm.isChecked()
        self.lbl_preview_status.setText("Loading preview...")
        self.btn_preview_reload.setEnabled(False)
        self._preview_loader = _PreviewLoader(
            self._path, size_ratio, grayscale,
            n_frames=n_frames, mode=mode, normalize=normalize
        )
        self._preview_loader.finished.connect(self._on_preview_loaded)
        self._preview_loader.start()

    def _on_preview_loaded(self, img: np.ndarray | None):
        self._preview_loader = None
        self.btn_preview_reload.setEnabled(True)
        if img is None:
            self.lbl_preview_status.setText("Preview failed")
            return
        self._preview = img
        self.lbl_preview_status.setText(
            "Drag ROI to select region (optional)"
        )
        h, w = img.shape[0], img.shape[1]
        grayscale = img.ndim == 2
        display_img = np.swapaxes(img, 0, 1) if img.ndim == 3 else img.T
        self._img_item.setImage(display_img)
        self._img_item.setRect(pg.QtCore.QRectF(0, 0, w, h))
        if grayscale:
            self._img_item.setLookupTable(_plasma_lut())
            self._img_item.setLevels([0, 255])
        else:
            self._img_item.setLookupTable(None)
        vb = self._plot_widget.getViewBox()
        vb.invertY(True)
        vb.setAspectLocked(lock=True, ratio=1.0)
        pad = 1.1
        margin_x = w * (pad - 1) / 2
        margin_y = h * (pad - 1) / 2
        self._plot_widget.setRange(
            xRange=(-margin_x, w + margin_x),
            yRange=(-margin_y, h + margin_y),
            padding=0,
        )
        if self._roi is not None:
            self._plot_widget.removeItem(self._roi)
        if self._initial_mask_rel is not None:
            r = self._initial_mask_rel
            x0 = max(0, int(r[0] * w))
            y0 = max(0, int(r[1] * h))
            x1 = min(w, int(r[2] * w))
            y1 = min(h, int(r[3] * h))
            if x1 > x0 and y1 > y0 and (x1 - x0 < w or y1 - y0 < h):
                self._roi = pg.RectROI((x0, y0), (x1 - x0, y1 - y0), pen=pg.mkPen("lime", width=2))
            else:
                self._roi = pg.RectROI((0, 0), (w, h), pen=pg.mkPen("lime", width=2))
        else:
            self._roi = pg.RectROI((0, 0), (w, h), pen=pg.mkPen("lime", width=2))
        self._roi.handleSize = 10
        self._roi.addScaleHandle([1, 1], [0, 0])
        self._roi.addScaleHandle([0, 0], [1, 1])
        self._roi.sigRegionChanged.connect(self._update_estimates)
        self._plot_widget.addItem(self._roi)

    def _reset_roi(self) -> None:
        """Reset crop ROI to full frame."""
        if self._preview is not None and self._roi is not None:
            h, w = self._preview.shape[0], self._preview.shape[1]
            self._roi.setPos((0, 0))
            self._roi.setSize((w, h))
            self._update_estimates()

    def _update_estimates(self):
        start = self.spin_start.value()
        end = self.spin_end.value()
        step = self.spin_step.value()
        resize = self.spin_resize.value() / 100.0
        grayscale = self.chk_grayscale.isChecked()

        # Validate range
        if start > end:
            # Normalize so that start <= end by swapping the values
            self.spin_start.setValue(end)
            self.spin_end.setValue(start)
            start, end = end, start
        # Calculate number of frames: len(range(start, end + 1, step))
        if end >= start:
            num_frames = (end - start) // step + 1
        else:
            num_frames = 0

        # metadata.size is (height, width); preview uses same dimensions
        height_full = int(self.metadata.size[0] * resize)
        width_full = int(self.metadata.size[1] * resize)
        width, height = width_full, height_full
        if self._preview is not None and self._roi is not None:
            pos = self._roi.pos()
            size = self._roi.size()
            h, w = self._preview.shape[0], self._preview.shape[1]
            x0 = max(0, int(pos.x()))
            y0 = max(0, int(pos.y()))
            x1 = min(w, int(pos.x() + size.x()))
            y1 = min(h, int(pos.y() + size.y()))
            if x1 > x0 and y1 > y0 and (x1 - x0 < w or y1 - y0 < h):
                width = x1 - x0
                height = y1 - y0
        channels = 1 if grayscale else 3
        dtype_size = 1 if self.chk_8bit.isChecked() else getattr(self, "_sample_bytes", 1)
        total_bytes = width * height * channels * num_frames * dtype_size
        gb = total_bytes / (1024**3)

        color = "green"
        available = get_available_ram()
        if gb > available * 0.9:
            color = "red"
        elif gb > available * 0.7:
            color = "orange"

        crop_note = " (with crop)" if (width, height) != (width_full, height_full) else ""
        self.lbl_ram_usage.setText(
            f"Estimated RAM{crop_note}: <font color='{color}'><b>{gb:.2f} GB</b></font>"
        )

    def get_params(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "frame_range": (self.spin_start.value(), self.spin_end.value()),
            "step": self.spin_step.value(),
            "size_ratio": self.spin_resize.value() / 100.0,
            "grayscale": self.chk_grayscale.isChecked(),
            "convert_to_8_bit": self.chk_8bit.isChecked(),
        }
        if self._preview is not None and self._roi is not None:
            pos = self._roi.pos()
            size = self._roi.size()
            h, w = self._preview.shape[0], self._preview.shape[1]
            x0 = max(0, int(pos.x()))
            y0 = max(0, int(pos.y()))
            x1 = min(w, int(pos.x() + size.x()))
            y1 = min(h, int(pos.y() + size.y()))
            if x1 > x0 and y1 > y0 and (x1 - x0 < w or y1 - y0 < h):
                out["mask"] = (slice(x0, x1), slice(y0, y1))
                out["mask_rel"] = (x0 / w, y0 / h, x1 / w, y1 / h)
        return out


class _ImagePreviewLoader(QThread):
    finished = pyqtSignal(object)

    def __init__(self, path: Path, size_ratio: float, grayscale: bool,
                 mode: str = "max", normalize: bool = True):
        super().__init__()
        self._path = path
        self._size_ratio = size_ratio
        self._grayscale = grayscale
        self._mode = mode
        self._normalize = normalize

    def run(self):
        img = get_image_preview(
            self._path,
            size_ratio=self._size_ratio,
            grayscale=self._grayscale,
            mode=self._mode,
            normalize=self._normalize,
        )
        self.finished.emit(img)


class ImageLoadOptionsDialog(QDialog):
    def __init__(
        self,
        path: Path,
        metadata: dict,
        parent: Optional[QWidget] = None,
        initial_params: Optional[dict] = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Image Loading Options")
        self._path = path
        self.metadata = metadata
        self._is_folder = metadata["file_count"] > 1
        self._preview: Optional[np.ndarray] = None
        self._roi: Optional[pg.RectROI] = None
        self._preview_loader: Optional[_ImagePreviewLoader] = None
        self._setup_ui()
        self._initial_mask_rel = initial_params.get("mask_rel") if initial_params else None
        if initial_params:
            self.spin_resize.setValue(
                int(initial_params.get("size_ratio", 1.0) * 100),
            )
            self.chk_grayscale.setChecked(
                initial_params.get("grayscale", False),
            )
            self.chk_8bit.setChecked(
                initial_params.get("convert_to_8_bit", False),
            )
            if self._is_folder and "subset_ratio" in initial_params:
                self.spin_subset.setValue(initial_params["subset_ratio"])
        else:
            is_gray, is_uint8 = get_sample_format(path)
            self.chk_grayscale.setChecked(is_gray)
            self.chk_8bit.setChecked(is_uint8)
        self._sample_bytes = get_sample_bytes_per_pixel(path)
        self._update_estimates()
        self._start_preview_load()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        info_layout = QFormLayout()
        self.lbl_file = QLabel(self.metadata["file_name"])
        h, w = self.metadata["size"]
        self.lbl_dims = QLabel(f"{h} x {w}")
        self.lbl_count = QLabel(str(self.metadata["file_count"]))
        self.lbl_format = QLabel(get_sample_format_display(self._path))
        self.lbl_format.setToolTip("Detected source format")
        info_layout.addRow("File:", self.lbl_file)
        info_layout.addRow("Dimensions:", self.lbl_dims)
        if self._is_folder:
            info_layout.addRow("Images:", self.lbl_count)
        info_layout.addRow("Source format:", self.lbl_format)
        layout.addLayout(info_layout)

        layout.addWidget(QLabel("<b>Loading Options</b>"))
        controls_layout = QFormLayout()
        self.spin_resize = QSpinBox()
        self.spin_resize.setRange(1, 100)
        self.spin_resize.setValue(100)
        self.spin_resize.setSuffix(" %")
        controls_layout.addRow("Resize:", self.spin_resize)
        if self._is_folder:
            self.spin_subset = QDoubleSpinBox()
            self.spin_subset.setRange(0.01, 1.0)
            self.spin_subset.setValue(1.0)
            self.spin_subset.setSingleStep(0.1)
            self.spin_subset.setPrefix("Subset: ")
            self.spin_subset.setToolTip("Ratio of images to load (1.0 = all)")
            controls_layout.addRow("", self.spin_subset)
        self.chk_grayscale = QCheckBox("Load as Grayscale")
        self.chk_grayscale.setChecked(False)
        self.chk_8bit = QCheckBox("Convert to 8 bit")
        self.chk_8bit.setChecked(False)
        controls_layout.addRow("", self.chk_grayscale)
        controls_layout.addRow("", self.chk_8bit)
        layout.addLayout(controls_layout)

        layout.addWidget(QLabel("<b>Preview / Crop</b>"))
        preview_opts = QHBoxLayout()
        self.cmb_preview_mode = QComboBox()
        self.cmb_preview_mode.addItems(["MAX (across samples)", "Single image"])
        self.cmb_preview_mode.setToolTip("MAX: max value per pixel across sampled images")
        preview_opts.addWidget(QLabel("Mode:"))
        preview_opts.addWidget(self.cmb_preview_mode)
        self.chk_preview_norm = QCheckBox("Normalize")
        self.chk_preview_norm.setChecked(True)
        self.chk_preview_norm.setToolTip("Min-max stretch for better visibility")
        preview_opts.addWidget(self.chk_preview_norm)
        self.btn_preview_reload = QPushButton("Refresh")
        self.btn_preview_reload.pressed.connect(self._start_preview_load)
        self.btn_roi_reset = QPushButton("Reset ROI")
        self.btn_roi_reset.setToolTip("Reset crop to full frame")
        self.btn_roi_reset.pressed.connect(self._reset_roi)
        preview_opts.addWidget(self.btn_preview_reload)
        preview_opts.addWidget(self.btn_roi_reset)
        preview_opts.addStretch()
        layout.addLayout(preview_opts)
        preview_frame = QFrame()
        preview_frame.setFrameStyle(QFrame.StyledPanel)
        preview_layout = QVBoxLayout(preview_frame)
        self.lbl_preview_status = QLabel("Loading preview...")
        preview_layout.addWidget(self.lbl_preview_status)
        self._plot_widget = pg.PlotWidget(background=(80, 80, 80))
        self._plot_widget.setMinimumSize(400, 360)
        self._plot_widget.setAspectLocked(False)
        self._plot_widget.hideAxis("left")
        self._plot_widget.hideAxis("bottom")
        self._img_item = pg.ImageItem()
        self._plot_widget.addItem(self._img_item)
        preview_layout.addWidget(self._plot_widget)
        layout.addWidget(preview_frame)

        self.lbl_ram_usage = QLabel("Estimated RAM: Calculating...")
        self.lbl_ram_available = QLabel(f"Available RAM: {get_available_ram():.2f} GB")
        layout.addWidget(self.lbl_ram_usage)
        layout.addWidget(self.lbl_ram_available)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.spin_resize.valueChanged.connect(self._update_estimates)
        self.chk_grayscale.toggled.connect(self._update_estimates)
        self.chk_grayscale.toggled.connect(self._on_grayscale_changed)
        self.chk_8bit.toggled.connect(self._update_estimates)
        if self._is_folder:
            self.spin_subset.valueChanged.connect(self._update_estimates)

    def _on_grayscale_changed(self):
        if self._preview_loader is None:
            self._start_preview_load()

    def _start_preview_load(self):
        if self._preview_loader is not None:
            return
        size_ratio = self.spin_resize.value() / 100.0
        grayscale = self.chk_grayscale.isChecked()
        mode = "max" if self.cmb_preview_mode.currentIndex() == 0 else "single"
        normalize = self.chk_preview_norm.isChecked()
        self.lbl_preview_status.setText("Loading preview...")
        self.btn_preview_reload.setEnabled(False)
        self._preview_loader = _ImagePreviewLoader(
            self._path, size_ratio, grayscale, mode=mode, normalize=normalize
        )
        self._preview_loader.finished.connect(self._on_preview_loaded)
        self._preview_loader.start()

    def _on_preview_loaded(self, img: np.ndarray | None):
        self._preview_loader = None
        self.btn_preview_reload.setEnabled(True)
        if img is None:
            self.lbl_preview_status.setText("Preview failed")
            return
        self._preview = img
        self.lbl_preview_status.setText("Drag ROI to select region (optional)")
        h, w = img.shape[0], img.shape[1]
        grayscale = img.ndim == 2
        display_img = np.swapaxes(img, 0, 1) if img.ndim == 3 else img.T
        self._img_item.setImage(display_img)
        self._img_item.setRect(pg.QtCore.QRectF(0, 0, w, h))
        if grayscale:
            self._img_item.setLookupTable(_plasma_lut())
            self._img_item.setLevels([0, 255])
        else:
            self._img_item.setLookupTable(None)
        vb = self._plot_widget.getViewBox()
        vb.invertY(True)
        vb.setAspectLocked(lock=True, ratio=1.0)
        pad = 1.1
        margin_x = w * (pad - 1) / 2
        margin_y = h * (pad - 1) / 2
        self._plot_widget.setRange(
            xRange=(-margin_x, w + margin_x),
            yRange=(-margin_y, h + margin_y),
            padding=0,
        )
        if self._roi is not None:
            self._plot_widget.removeItem(self._roi)
        if self._initial_mask_rel is not None:
            r = self._initial_mask_rel
            x0 = max(0, int(r[0] * w))
            y0 = max(0, int(r[1] * h))
            x1 = min(w, int(r[2] * w))
            y1 = min(h, int(r[3] * h))
            if x1 > x0 and y1 > y0 and (x1 - x0 < w or y1 - y0 < h):
                self._roi = pg.RectROI((x0, y0), (x1 - x0, y1 - y0), pen=pg.mkPen("lime", width=2))
            else:
                self._roi = pg.RectROI((0, 0), (w, h), pen=pg.mkPen("lime", width=2))
        else:
            self._roi = pg.RectROI((0, 0), (w, h), pen=pg.mkPen("lime", width=2))
        self._roi.handleSize = 10
        self._roi.addScaleHandle([1, 1], [0, 0])
        self._roi.addScaleHandle([0, 0], [1, 1])
        self._roi.sigRegionChanged.connect(self._update_estimates)
        self._plot_widget.addItem(self._roi)

    def _reset_roi(self) -> None:
        """Reset crop ROI to full frame."""
        if self._preview is not None and self._roi is not None:
            h, w = self._preview.shape[0], self._preview.shape[1]
            self._roi.setPos((0, 0))
            self._roi.setSize((w, h))
            self._update_estimates()

    def _update_estimates(self):
        resize = self.spin_resize.value() / 100.0
        grayscale = self.chk_grayscale.isChecked()
        subset = self.spin_subset.value() if self._is_folder else 1.0
        h_full, w_full = self.metadata["size"]
        height_full = int(h_full * resize)
        width_full = int(w_full * resize)
        width, height = width_full, height_full
        if self._preview is not None and self._roi is not None:
            pos = self._roi.pos()
            size = self._roi.size()
            h, w = self._preview.shape[0], self._preview.shape[1]
            x0 = max(0, int(pos.x()))
            y0 = max(0, int(pos.y()))
            x1 = min(w, int(pos.x() + size.x()))
            y1 = min(h, int(pos.y() + size.y()))
            if x1 > x0 and y1 > y0 and (x1 - x0 < w or y1 - y0 < h):
                width = x1 - x0
                height = y1 - y0
        num_images = max(1, int(self.metadata["file_count"] * subset))
        channels = 1 if grayscale else 3
        dtype_size = 1 if self.chk_8bit.isChecked() else getattr(self, "_sample_bytes", 1)
        total_bytes = width * height * channels * num_images * dtype_size
        gb = total_bytes / (1024**3)
        color = "green"
        available = get_available_ram()
        if gb > available * 0.9:
            color = "red"
        elif gb > available * 0.7:
            color = "orange"
        crop_note = " (with crop)" if (width, height) != (width_full, height_full) else ""
        self.lbl_ram_usage.setText(
            f"Estimated RAM{crop_note}: <font color='{color}'><b>{gb:.2f} GB</b></font>"
        )

    def get_params(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "size_ratio": self.spin_resize.value() / 100.0,
            "grayscale": self.chk_grayscale.isChecked(),
            "convert_to_8_bit": self.chk_8bit.isChecked(),
        }
        if self._is_folder:
            out["subset_ratio"] = self.spin_subset.value()
        if self._preview is not None and self._roi is not None:
            pos = self._roi.pos()
            size = self._roi.size()
            h, w = self._preview.shape[0], self._preview.shape[1]
            x0 = max(0, int(pos.x()))
            y0 = max(0, int(pos.y()))
            x1 = min(w, int(pos.x() + size.x()))
            y1 = min(h, int(pos.y() + size.y()))
            if x1 > x0 and y1 > y0 and (x1 - x0 < w or y1 - y0 < h):
                out["mask"] = (slice(x0, x1), slice(y0, y1))
                out["mask_rel"] = (x0 / w, y0 / h, x1 / w, y1 / h)
        return out

class LiveViewOptionsDialog(QDialog):
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("Camera Options")
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        form = QFormLayout()

        self.spin_cam = QSpinBox()
        self.spin_cam.setRange(0, 99)
        self.spin_cam.setValue(0)
        form.addRow("Camera Index:", self.spin_cam)

        self.spin_buffer = QSpinBox()
        self.spin_buffer.setRange(1, 10000)
        self.spin_buffer.setValue(100)
        self.spin_buffer.setSuffix(" frames")
        form.addRow("Buffer Size:", self.spin_buffer)

        self.spin_interval = QSpinBox()
        self.spin_interval.setRange(1, 1000) # 1ms to 1s
        self.spin_interval.setValue(33) # ~30fps
        self.spin_interval.setSuffix(" ms")
        form.addRow("Frame Interval:", self.spin_interval)

        self.spin_downsample = QDoubleSpinBox()
        self.spin_downsample.setRange(0.1, 1.0)
        self.spin_downsample.setValue(1.0)
        self.spin_downsample.setSingleStep(0.1)
        form.addRow("Downsample:", self.spin_downsample)

        self.chk_grayscale = QCheckBox("Grayscale")
        self.chk_grayscale.setChecked(False)
        form.addRow("", self.chk_grayscale)

        layout.addLayout(form)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_params(self) -> dict[str, Any]:
        return {
            "cam_id": self.spin_cam.value(),
            "buffer_size": self.spin_buffer.value(),
            "frame_interval_ms": self.spin_interval.value(),
            "downsample": self.spin_downsample.value(),
            "grayscale": self.chk_grayscale.isChecked(),
        }
