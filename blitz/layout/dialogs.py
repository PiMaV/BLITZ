from pathlib import Path
from typing import Any, Optional

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import (QButtonGroup, QCheckBox, QComboBox, QDialog,
                             QDialogButtonBox, QDoubleSpinBox, QFormLayout, QFrame,
                             QHBoxLayout, QLabel, QListWidget, QListWidgetItem,
                             QProgressBar, QPushButton,
                             QRadioButton, QSlider, QSpinBox, QVBoxLayout, QWidget)

from pyqtgraph.graphicsItems.GradientEditorItem import Gradients

from ..data.converters.ascii import (
    DELIMITERS,
    estimate_ascii_datatype,
    first_col_looks_like_row_number,
    get_ascii_files,
    get_ascii_preview,
    parse_ascii_raw,
)
from ..data.load import (get_image_metadata, get_image_preview,
                         get_paths_image_preview, get_sample_bytes_per_pixel,
                         get_sample_format, get_sample_format_display,
                         get_video_preview)
from ..data.image import VideoMetaData
from ..theme import get_dialog_preview_bg, set_checkbox_visibly_enabled
from ..tools import get_available_ram
from .roi_mixin import ROIMixin

RAW_PREVIEW_MAX_CHARS = 80
RAW_PREVIEW_MAX_LINES = 5

# Keep refs to preview workers that outlive a dialog (close while still running).
_ORPHAN_PREVIEW_THREADS: list[QThread] = []


class _PreviewThreadGuard:
    """Keep QThread refs alive until QThread.finished — avoids destroy-while-running."""

    def __init__(self) -> None:
        self.current: Optional[QThread] = None
        self._zombies: list[QThread] = []

    def _reap(self, loader: QThread) -> None:
        if getattr(loader, "_preview_reaped", False):
            return
        loader._preview_reaped = True  # type: ignore[attr-defined]
        if loader in self._zombies:
            self._zombies.remove(loader)
        if loader in _ORPHAN_PREVIEW_THREADS:
            _ORPHAN_PREVIEW_THREADS.remove(loader)
        if self.current is loader:
            self.current = None
        try:
            loader.finished.disconnect()
        except TypeError:
            pass
        loader.deleteLater()

    def stop(self, finished_slot=None, signal_name: str = "finished_preview") -> None:
        loader = self.current
        self.current = None
        if loader is None:
            return
        if finished_slot is not None:
            try:
                getattr(loader, signal_name).disconnect(finished_slot)
            except TypeError:
                pass
        # Always park in zombies until the real QThread.finished fires (or already done).
        if loader not in self._zombies:
            self._zombies.append(loader)
        if not loader.isRunning():
            self._reap(loader)
        else:
            # Brief yield so a nearly-done worker can finish without stacking forever.
            loader.wait(50)
            if not loader.isRunning():
                self._reap(loader)

    def adopt(self, loader: QThread) -> QThread:
        self.current = loader
        # Use QThread.finished (not a shadowed custom signal) for lifetime.
        loader.finished.connect(lambda _l=loader: self._reap(_l))
        return loader

    def clear_current_if(self, loader: Optional[QThread]) -> None:
        if loader is not None and self.current is loader:
            self.current = None
        # Keep ref in zombies until QThread.finished → _reap (set by adopt).
        if loader is not None and loader not in self._zombies:
            self._zombies.append(loader)

    def stop_all(self, finished_slot=None, signal_name: str = "finished_preview") -> None:
        self.stop(finished_slot=finished_slot, signal_name=signal_name)
        for z in list(self._zombies):
            if z.isRunning() and not z.wait(8000):
                # Dialog is closing; keep a module-level ref so GC cannot destroy
                # the wrapper while the C++ thread is still running.
                if z not in _ORPHAN_PREVIEW_THREADS:
                    _ORPHAN_PREVIEW_THREADS.append(z)
                if z in self._zombies:
                    self._zombies.remove(z)
                continue
            self._reap(z)


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
    finished_preview = pyqtSignal(object, int)  # img, generation

    def __init__(self, path: Path, size_ratio: float, grayscale: bool,
                 n_frames: int = 10, mode: str = "max", normalize: bool = True,
                 generation: int = 0):
        super().__init__()
        self._path = path
        self._size_ratio = size_ratio
        self._grayscale = grayscale
        self._n_frames = n_frames
        self._mode = mode
        self._normalize = normalize
        self._generation = generation

    def run(self):
        img = get_video_preview(
            self._path,
            n_frames=self._n_frames,
            size_ratio=self._size_ratio,
            grayscale=self._grayscale,
            mode=self._mode,
            normalize=self._normalize,
        )
        self.finished_preview.emit(img, self._generation)


class VideoLoadOptionsDialog(QDialog, ROIMixin):
    def __init__(
        self,
        path: Path,
        metadata: VideoMetaData,
        parent: Optional[QWidget] = None,
        initial_params: Optional[dict] = None,
    ):
        super().__init__(parent)
        self.setWindowIcon(QIcon(":/icon/blitz.ico"))
        self.setWindowTitle("Video Loading Options")
        self._path = path
        self.metadata = metadata
        self._preview: Optional[np.ndarray] = None
        self._roi: Optional[pg.RectROI] = None
        self._preview_loader: Optional[_PreviewLoader] = None
        self._preview_threads = _PreviewThreadGuard()
        self._preview_generation = 0
        self._preview_debounce: Optional[QTimer] = None
        self._initial_params = initial_params or {}
        self._setup_ui()
        self._initial_mask_rel = self._initial_params.get("mask_rel")
        self._initial_roi_state = self._initial_params.get("roi_state")
        is_gray, is_uint8 = get_sample_format(path)
        if initial_params:
            self.spin_resize.setValue(
                int(initial_params.get("size_ratio", 1.0) * 100),
            )
            self.spin_step.setValue(initial_params.get("step", 1))
            self.chk_grayscale.setChecked(is_gray or initial_params.get("grayscale", False))
            self.chk_8bit.setChecked(is_uint8 or initial_params.get("convert_to_8_bit", False))
            self.chk_normalize.setChecked(initial_params.get("normalize", False))
            # Set transform checkboxes
            self._apply_initial_transforms(initial_params)
        else:
            self.chk_grayscale.setChecked(is_gray)
            self.chk_8bit.setChecked(is_uint8)
        set_checkbox_visibly_enabled(self.chk_8bit, not is_uint8)
        if is_uint8:
            self.chk_8bit.setToolTip("Source is already 8 bit")
        set_checkbox_visibly_enabled(self.chk_grayscale, not is_gray)
        if is_gray:
            self.chk_grayscale.setToolTip(
                "Source is grayscale; loading as RGB would waste RAM"
            )
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
        self.chk_normalize = QCheckBox("Normalize each frame")
        self.chk_normalize.setChecked(False)
        self.chk_normalize.setToolTip(
            "On load: stretch each frame to full range (per-frame). "
            "Works for 8/16 bit and float. Off = comparable brightness."
        )
        controls_layout.addRow("", self.chk_grayscale)
        controls_layout.addRow("", self.chk_8bit)
        controls_layout.addRow("", self.chk_normalize)

        layout.addLayout(controls_layout)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setFrameShadow(QFrame.Shadow.Sunken)
        layout.addWidget(sep)

        # Preview with Crop ROI
        layout.addWidget(QLabel("<b>Preview / Crop</b>"))
        preview_opts = QHBoxLayout()
        self.cmb_preview_mode = QComboBox()
        self.cmb_preview_mode.addItems(["MAX (across frames)", "Single frame (center)"])
        self.cmb_preview_mode.setCurrentIndex(0)  # MAX default
        self.cmb_preview_mode.setToolTip(
            "MAX: maximum value per pixel across frames. Single: one representative frame."
        )
        preview_opts.addWidget(QLabel("Mode:"))
        preview_opts.addWidget(self.cmb_preview_mode)
        self.chk_preview_norm = QCheckBox("Preview normalize")
        self.chk_preview_norm.setChecked(True)
        self.chk_preview_norm.setToolTip(
            "Display only in this dialog — min-max stretch for visibility. "
            "Does not change what is loaded."
        )
        preview_opts.addWidget(self.chk_preview_norm)
        self.btn_roi_reset = QPushButton("Reset ROI")
        self.btn_roi_reset.setToolTip("Reset crop to full frame")
        self.btn_roi_reset.pressed.connect(self._reset_roi)
        preview_opts.addWidget(self.btn_roi_reset)
        preview_opts.addWidget(self._make_auto_crop_button())
        preview_opts.addStretch()
        layout.addLayout(preview_opts)

        preview_frame = QFrame()
        preview_frame.setFrameStyle(QFrame.Shape.StyledPanel | QFrame.Shadow.Sunken)
        preview_layout = QVBoxLayout(preview_frame)
        self.lbl_preview_status = QLabel("Loading preview...")
        preview_layout.addWidget(self.lbl_preview_status)
        self._plot_widget = pg.PlotWidget(background=get_dialog_preview_bg())
        self._plot_widget.setMinimumSize(400, 360)
        self._plot_widget.setAspectLocked(False)
        self._plot_widget.hideAxis("left")
        self._plot_widget.hideAxis("bottom")
        self._img_item = pg.ImageItem()
        self._plot_widget.addItem(self._img_item)
        preview_layout.addWidget(self._plot_widget)
        layout.addWidget(preview_frame)

        # ROI Mixin Controls
        self._setup_roi_controls(layout)

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
        self.cmb_preview_mode.currentTextChanged.connect(self._on_preview_option_changed)
        self.chk_preview_norm.toggled.connect(self._on_preview_option_changed)

    def _on_grayscale_changed(self):
        self._on_preview_option_changed()

    def _stop_preview_loader(self) -> None:
        self._preview_threads.stop(
            finished_slot=self._on_preview_loaded, signal_name="finished_preview"
        )
        self._preview_loader = None

    def _on_preview_option_changed(self):
        self._preview_generation += 1
        self.lbl_preview_status.setText("Loading preview...")
        # Drop the in-flight worker immediately; debounce only restarts.
        self._stop_preview_loader()
        if self._preview_debounce is None:
            self._preview_debounce = QTimer(self)
            self._preview_debounce.setSingleShot(True)
            self._preview_debounce.timeout.connect(self._restart_preview_debounced)
        self._preview_debounce.start(80)

    def _restart_preview_debounced(self) -> None:
        self._start_preview_load()

    def _start_preview_load(self):
        self._stop_preview_loader()
        size_ratio = self.spin_resize.value() / 100.0
        grayscale = self.chk_grayscale.isChecked()
        n_frames = min(10, max(1, self.metadata.frame_count))
        # Index 0 = MAX (default), 1 = Single
        mode = "max" if self.cmb_preview_mode.currentIndex() == 0 else "single"
        normalize = self.chk_preview_norm.isChecked()
        self.lbl_preview_status.setText("Loading preview...")
        self._preview_generation += 1
        generation = self._preview_generation
        loader = _PreviewLoader(
            self._path, size_ratio, grayscale,
            n_frames=n_frames, mode=mode, normalize=normalize,
            generation=generation,
        )
        self._preview_loader = self._preview_threads.adopt(loader)
        loader.finished_preview.connect(self._on_preview_loaded)
        loader.start()

    def _on_preview_loaded(self, img: np.ndarray | None, generation: int = 0):
        sender = self.sender()
        if generation != self._preview_generation:
            if sender is not None and sender is self._preview_loader:
                self._preview_threads.clear_current_if(self._preview_loader)
                self._preview_loader = None
            return
        if sender is not None and sender is not self._preview_loader:
            return
        self._preview_threads.clear_current_if(self._preview_loader)
        self._preview_loader = None
        if img is None:
            self.lbl_preview_status.setText("Preview failed")
            return
        self._preview = img
        self.lbl_preview_status.setText(
            "Drag ROI to select region (optional)"
        )
        self._apply_auto_transpose()

        # Apply transforms from checkboxes if they were set in init
        img = self._get_transformed_preview()

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

        # ROI Logic:
        # 1. Prefer absolute roi_state if present and valid?
        #    The sanity check should be done in MainWindow.
        #    If roi_state is passed, we assume it's valid/desired.
        # 2. Else use mask_rel (legacy/fallback).
        # 3. Else Full Frame.

        self._roi = pg.RectROI((0, 0), (w, h), pen=pg.mkPen("lime", width=2))

        if self._initial_roi_state:
            s = self._initial_roi_state
            self._roi.setPos(s['pos'])
            self._roi.setSize(s['size'])
            self._roi.setAngle(0)
        elif self._initial_mask_rel is not None:
            r = self._initial_mask_rel
            x0 = max(0, int(r[0] * w))
            y0 = max(0, int(r[1] * h))
            x1 = min(w, int(r[2] * w))
            y1 = min(h, int(r[3] * h))
            if x1 > x0 and y1 > y0 and (x1 - x0 < w or y1 - y0 < h):
                self._roi.setPos((x0, y0))
                self._roi.setSize((x1 - x0, y1 - y0))

        self._roi.handleSize = 10
        self._plot_widget.addItem(self._roi)

        # Connect Mixin Signals
        self._connect_roi_signals()
        self._update_estimates()

    def _reset_roi(self) -> None:
        """Reset crop ROI to full frame and reset transforms."""
        self._reset_transform_checkboxes()
        if self._preview is not None and self._roi is not None:
            h, w = self._preview.shape[0], self._preview.shape[1]
            self._roi.setPos((0, 0))
            self._roi.setSize((w, h))
            self._roi.setAngle(0)
            self._update_estimates()

    def _update_estimates(self):
        start = self.spin_start.value()
        end = self.spin_end.value()
        step = self.spin_step.value()
        resize = self.spin_resize.value() / 100.0
        grayscale = self.chk_grayscale.isChecked()

        if start > end:
            self.spin_start.setValue(end)
            self.spin_end.setValue(start)
            start, end = end, start
        if end >= start:
            num_frames = (end - start) // step + 1
        else:
            num_frames = 0

        # Calculate mask using Mixin
        mask, _ = self._get_roi_source_mask()

        # Original dims
        h_orig, w_orig = self.metadata.size
        # But we need the dims of the *masked* region
        if mask:
            # mask is (slice_y, slice_x)
            sl_x = mask[1]
            sl_y = mask[0]
            width = sl_x.stop - sl_x.start
            height = sl_y.stop - sl_y.start
        else:
            # Full frame (fallback)
            width, height = w_orig, h_orig

        # Apply resize
        width = int(width * resize)
        height = int(height * resize)

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

        self.lbl_ram_usage.setText(
            f"Estimated RAM: <font color='{color}'><b>{gb:.2f} GB</b></font>"
        )

    def closeEvent(self, event) -> None:
        if self._preview_debounce is not None:
            self._preview_debounce.stop()
        self._stop_preview_loader()
        self._preview_threads.stop_all(
            finished_slot=self._on_preview_loaded, signal_name="finished_preview"
        )
        super().closeEvent(event)

    def reject(self) -> None:
        if self._preview_debounce is not None:
            self._preview_debounce.stop()
        self._stop_preview_loader()
        self._preview_threads.stop_all(
            finished_slot=self._on_preview_loaded, signal_name="finished_preview"
        )
        super().reject()

    def accept(self) -> None:
        if self._preview_debounce is not None:
            self._preview_debounce.stop()
        self._stop_preview_loader()
        self._preview_threads.stop_all(
            finished_slot=self._on_preview_loaded, signal_name="finished_preview"
        )
        super().accept()

    def get_params(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "frame_range": (self.spin_start.value(), self.spin_end.value()),
            "step": self.spin_step.value(),
            "size_ratio": self.spin_resize.value() / 100.0,
            "grayscale": self.chk_grayscale.isChecked(),
            "convert_to_8_bit": self.chk_8bit.isChecked(),
            "normalize": self.chk_normalize.isChecked(),
            **self._transform_params(),
        }

        mask, mask_rel = self._get_roi_source_mask()
        if mask and mask_rel:
            out["mask"] = mask
            out["mask_rel"] = mask_rel

        if self._roi is not None:
            state = self._roi.getState()
            img = self._get_transformed_preview()
            h_src, w_src = img.shape[0], img.shape[1] if img is not None else (0, 0)
            out["roi_state"] = {
                "pos": (state['pos'].x(), state['pos'].y()),
                "size": (state['size'].x(), state['size'].y()),
                "source_size": (h_src, w_src),
            }

        return out


class _ImagePreviewLoader(QThread):
    finished_preview = pyqtSignal(object, int)  # img, generation

    def __init__(
        self,
        path: Path,
        size_ratio: float,
        grayscale: bool,
        mode: str = "max",
        normalize: bool = True,
        file_list: Optional[list] = None,
        generation: int = 0,
    ):
        super().__init__()
        self._path = path
        self._size_ratio = size_ratio
        self._grayscale = grayscale
        self._mode = mode
        self._normalize = normalize
        self._file_list = file_list
        self._generation = generation

    def run(self):
        if self._file_list:
            img = get_paths_image_preview(
                self._file_list,
                size_ratio=self._size_ratio,
                n_samples=10,
                mode=self._mode,
                normalize=self._normalize,
                grayscale=self._grayscale,
            )
        else:
            img = get_image_preview(
                self._path,
                size_ratio=self._size_ratio,
                grayscale=self._grayscale,
                mode=self._mode,
                normalize=self._normalize,
            )
        self.finished_preview.emit(img, self._generation)


class ImageLoadOptionsDialog(QDialog, ROIMixin):
    def __init__(
        self,
        path: Path,
        metadata: dict,
        parent: Optional[QWidget] = None,
        initial_params: Optional[dict] = None,
        file_list: Optional[list] = None,
    ):
        super().__init__(parent)
        self.setWindowIcon(QIcon(":/icon/blitz.ico"))
        self.setWindowTitle("Image Loading Options")
        self._path = path
        self._file_list = file_list
        self.metadata = metadata
        self._is_folder = metadata["file_count"] > 1
        self._preview: Optional[np.ndarray] = None
        self._roi: Optional[pg.RectROI] = None
        self._preview_loader: Optional[_ImagePreviewLoader] = None
        self._preview_threads = _PreviewThreadGuard()
        self._preview_generation = 0
        self._preview_debounce: Optional[QTimer] = None
        self._initial_params = initial_params or {}
        self._setup_ui()
        self._initial_mask_rel = self._initial_params.get("mask_rel")
        self._initial_roi_state = self._initial_params.get("roi_state")
        probe = (file_list[0] if file_list else path)
        is_gray, is_uint8 = get_sample_format(probe)
        if initial_params:
            self.spin_resize.setValue(
                int(initial_params.get("size_ratio", 1.0) * 100),
            )
            self.chk_grayscale.setChecked(is_gray or initial_params.get("grayscale", False))
            self.chk_8bit.setChecked(is_uint8 or initial_params.get("convert_to_8_bit", False))
            self.chk_normalize.setChecked(initial_params.get("normalize", False))
            if self._is_folder and "subset_ratio" in initial_params:
                self.spin_subset.setValue(initial_params["subset_ratio"])
            self._apply_initial_transforms(initial_params)
        else:
            self.chk_grayscale.setChecked(is_gray)
            self.chk_8bit.setChecked(is_uint8)
        set_checkbox_visibly_enabled(self.chk_grayscale, not is_gray)
        if is_gray:
            self.chk_grayscale.setToolTip("Source is grayscale; loading as RGB would waste RAM")
        set_checkbox_visibly_enabled(self.chk_8bit, not is_uint8)
        if is_uint8:
            self.chk_8bit.setToolTip("Source is already 8 bit")
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
        resize_row = QHBoxLayout()
        resize_row.addWidget(self.spin_resize)
        self.lbl_size_px = QLabel("")
        resize_row.addWidget(self.lbl_size_px)
        resize_row.addStretch()
        controls_layout.addRow("Resize:", resize_row)
        if self._is_folder:
            self.spin_subset = QDoubleSpinBox()
            self.spin_subset.setRange(0.01, 1.0)
            self.spin_subset.setValue(1.0)
            self.spin_subset.setSingleStep(0.1)
            self.spin_subset.setPrefix("Subset: ")
            self.spin_subset.setToolTip("Ratio of images to load (1.0 = all)")
            subset_row = QHBoxLayout()
            subset_row.addWidget(self.spin_subset)
            self.lbl_num_frames = QLabel("")
            subset_row.addWidget(self.lbl_num_frames)
            subset_row.addStretch()
            controls_layout.addRow("", subset_row)
        self.chk_grayscale = QCheckBox("Load as Grayscale")
        self.chk_grayscale.setChecked(False)
        self.chk_8bit = QCheckBox("Convert to 8 bit")
        self.chk_8bit.setChecked(False)
        self.chk_normalize = QCheckBox("Normalize each image")
        self.chk_normalize.setChecked(False)
        self.chk_normalize.setToolTip(
            "On load: stretch each image to full range (per-image). "
            "Works for 8/16 bit and float. Off = comparable brightness."
        )
        controls_layout.addRow("", self.chk_grayscale)
        controls_layout.addRow("", self.chk_8bit)
        controls_layout.addRow("", self.chk_normalize)
        layout.addLayout(controls_layout)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setFrameShadow(QFrame.Shadow.Sunken)
        layout.addWidget(sep)

        layout.addWidget(QLabel("<b>Preview / Crop</b>"))
        preview_opts = QHBoxLayout()
        self.cmb_preview_mode = QComboBox()
        self.cmb_preview_mode.addItems(["MAX (across samples)", "Single image"])
        self.cmb_preview_mode.setCurrentIndex(0)  # MAX default
        self.cmb_preview_mode.setToolTip(
            "MAX: max value per pixel across samples. Single: one representative image."
        )
        preview_opts.addWidget(QLabel("Mode:"))
        preview_opts.addWidget(self.cmb_preview_mode)
        self.chk_preview_norm = QCheckBox("Preview normalize")
        self.chk_preview_norm.setChecked(True)
        self.chk_preview_norm.setToolTip(
            "Display only in this dialog — min-max stretch for visibility. "
            "Does not change what is loaded."
        )
        preview_opts.addWidget(self.chk_preview_norm)
        self.btn_roi_reset = QPushButton("Reset ROI")
        self.btn_roi_reset.setToolTip("Reset crop to full frame")
        self.btn_roi_reset.pressed.connect(self._reset_roi)
        preview_opts.addWidget(self.btn_roi_reset)
        preview_opts.addWidget(self._make_auto_crop_button())
        preview_opts.addStretch()
        layout.addLayout(preview_opts)

        preview_frame = QFrame()
        preview_frame.setFrameStyle(QFrame.Shape.StyledPanel | QFrame.Shadow.Sunken)
        preview_layout = QVBoxLayout(preview_frame)
        self.lbl_preview_status = QLabel("Loading preview...")
        preview_layout.addWidget(self.lbl_preview_status)
        self._plot_widget = pg.PlotWidget(background=get_dialog_preview_bg())
        self._plot_widget.setMinimumSize(400, 360)
        self._plot_widget.setAspectLocked(False)
        self._plot_widget.hideAxis("left")
        self._plot_widget.hideAxis("bottom")
        self._img_item = pg.ImageItem()
        self._plot_widget.addItem(self._img_item)
        preview_layout.addWidget(self._plot_widget)
        layout.addWidget(preview_frame)

        # ROI Mixin Controls
        self._setup_roi_controls(layout)

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
        self.cmb_preview_mode.currentTextChanged.connect(self._on_preview_option_changed)
        self.chk_preview_norm.toggled.connect(self._on_preview_option_changed)
        if self._is_folder:
            self.spin_subset.valueChanged.connect(self._update_estimates)

    def _on_grayscale_changed(self):
        self._on_preview_option_changed()

    def _stop_preview_loader(self) -> None:
        self._preview_threads.stop(
            finished_slot=self._on_preview_loaded, signal_name="finished_preview"
        )
        self._preview_loader = None

    def _on_preview_option_changed(self):
        self._preview_generation += 1
        self.lbl_preview_status.setText("Loading preview...")
        # Drop the in-flight worker immediately; debounce only restarts.
        self._stop_preview_loader()
        if self._preview_debounce is None:
            self._preview_debounce = QTimer(self)
            self._preview_debounce.setSingleShot(True)
            self._preview_debounce.timeout.connect(self._restart_preview_debounced)
        self._preview_debounce.start(80)

    def _restart_preview_debounced(self) -> None:
        self._start_preview_load()

    def _start_preview_load(self):
        self._stop_preview_loader()
        size_ratio = self.spin_resize.value() / 100.0
        grayscale = self.chk_grayscale.isChecked()
        # Index 0 = MAX (default), 1 = Single
        mode = "max" if self.cmb_preview_mode.currentIndex() == 0 else "single"
        normalize = self.chk_preview_norm.isChecked()
        self.lbl_preview_status.setText("Loading preview...")
        self._preview_generation += 1
        generation = self._preview_generation
        loader = _ImagePreviewLoader(
            self._path,
            size_ratio,
            grayscale,
            mode=mode,
            normalize=normalize,
            file_list=self._file_list,
            generation=generation,
        )
        self._preview_loader = self._preview_threads.adopt(loader)
        loader.finished_preview.connect(self._on_preview_loaded)
        loader.start()

    def _on_preview_loaded(self, img: np.ndarray | None, generation: int = 0):
        sender = self.sender()
        if generation != self._preview_generation:
            if sender is not None and sender is self._preview_loader:
                self._preview_threads.clear_current_if(self._preview_loader)
                self._preview_loader = None
            return
        if sender is not None and sender is not self._preview_loader:
            return
        self._preview_threads.clear_current_if(self._preview_loader)
        self._preview_loader = None
        if img is None:
            self.lbl_preview_status.setText("Preview failed")
            return
        self._preview = img
        self.lbl_preview_status.setText("Drag ROI to select region (optional)")
        self._apply_auto_transpose()

        # Apply transforms
        img = self._get_transformed_preview()

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

        self._roi = pg.RectROI((0, 0), (w, h), pen=pg.mkPen("lime", width=2))

        if self._initial_roi_state:
            s = self._initial_roi_state
            self._roi.setPos(s['pos'])
            self._roi.setSize(s['size'])
            self._roi.setAngle(0)
        elif self._initial_mask_rel is not None:
            r = self._initial_mask_rel
            x0 = max(0, int(r[0] * w))
            y0 = max(0, int(r[1] * h))
            x1 = min(w, int(r[2] * w))
            y1 = min(h, int(r[3] * h))
            if x1 > x0 and y1 > y0 and (x1 - x0 < w or y1 - y0 < h):
                self._roi.setPos((x0, y0))
                self._roi.setSize((x1 - x0, y1 - y0))

        self._roi.handleSize = 10
        self._plot_widget.addItem(self._roi)

        self._connect_roi_signals()
        self._update_estimates()

    def closeEvent(self, event) -> None:
        if self._preview_debounce is not None:
            self._preview_debounce.stop()
        self._stop_preview_loader()
        self._preview_threads.stop_all(
            finished_slot=self._on_preview_loaded, signal_name="finished_preview"
        )
        super().closeEvent(event)

    def reject(self) -> None:
        if self._preview_debounce is not None:
            self._preview_debounce.stop()
        self._stop_preview_loader()
        self._preview_threads.stop_all(
            finished_slot=self._on_preview_loaded, signal_name="finished_preview"
        )
        super().reject()

    def accept(self) -> None:
        if self._preview_debounce is not None:
            self._preview_debounce.stop()
        self._stop_preview_loader()
        self._preview_threads.stop_all(
            finished_slot=self._on_preview_loaded, signal_name="finished_preview"
        )
        super().accept()

    def _reset_roi(self) -> None:
        """Reset crop ROI to full frame."""
        self._reset_transform_checkboxes()
        if self._preview is not None and self._roi is not None:
            h, w = self._preview.shape[0], self._preview.shape[1]
            self._roi.setPos((0, 0))
            self._roi.setSize((w, h))
            self._roi.setAngle(0)
            self._update_estimates()

    def _update_estimates(self):
        resize = self.spin_resize.value() / 100.0
        grayscale = self.chk_grayscale.isChecked()
        subset = self.spin_subset.value() if self._is_folder else 1.0

        # Original dims (file on disk)
        h_orig, w_orig = self.metadata["size"]

        mask, _ = self._get_roi_source_mask()
        if mask:
            sl_x = mask[1]
            sl_y = mask[0]
            w_orig = sl_x.stop - sl_x.start
            h_orig = sl_y.stop - sl_y.start

        h_out = int(h_orig * resize)
        w_out = int(w_orig * resize)

        self.lbl_size_px.setText(f"-> {h_out} x {w_out} px")

        num_images = max(1, int(self.metadata["file_count"] * subset))

        if self._is_folder:
            color_result = "green"
            tooltip = ""
            if num_images > 1000:
                color_result = "red"
                tooltip = "Kann man machen, aber starte evtl. erstmal mit weniger."
            elif num_images > 500:
                color_result = "orange"
                tooltip = "Kann man machen, aber starte evtl. erstmal mit weniger."
            self.lbl_num_frames.setText(
                f"-> <font color='{color_result}'><b>{num_images}</b></font> images"
            )
            self.lbl_num_frames.setToolTip(tooltip)

        channels = 1 if grayscale else 3
        dtype_size = 1 if self.chk_8bit.isChecked() else getattr(self, "_sample_bytes", 1)
        total_bytes = h_out * w_out * channels * num_images * dtype_size
        total_gb = total_bytes / (1024**3)
        color = "green"
        if total_gb > get_available_ram() * 0.9:
            color = "red"
        elif total_gb > get_available_ram() * 0.7:
            color = "orange"

        self.lbl_ram_usage.setText(
            f"Estimated RAM: <font color='{color}'><b>{total_gb:.2f} GB</b></font>"
        )

    def get_params(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "size_ratio": self.spin_resize.value() / 100.0,
            "grayscale": self.chk_grayscale.isChecked(),
            "convert_to_8_bit": self.chk_8bit.isChecked(),
            "normalize": self.chk_normalize.isChecked(),
            **self._transform_params(),
        }
        if self._is_folder:
            out["subset_ratio"] = self.spin_subset.value()

        mask, mask_rel = self._get_roi_source_mask()
        if mask and mask_rel:
            out["mask"] = mask
            out["mask_rel"] = mask_rel

        if self._roi is not None:
            state = self._roi.getState()
            img = self._get_transformed_preview()
            h_src, w_src = img.shape[0], img.shape[1] if img is not None else (0, 0)
            out["roi_state"] = {
                "pos": (state['pos'].x(), state['pos'].y()),
                "size": (state['size'].x(), state['size'].y()),
                "source_size": (h_src, w_src),
            }
        return out


class AsciiLoadOptionsDialog(QDialog, ROIMixin):
    """Load options for ASCII files (.asc, .dat). Structure mirrors ImageLoadOptionsDialog."""

    def __init__(
        self,
        path: Path,
        metadata: dict,
        parent: Optional[QWidget] = None,
        initial_params: Optional[dict] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowIcon(QIcon(":/icon/blitz.ico"))
        self.setWindowTitle("ASCII Loading Options")
        self._path = path
        self.metadata = metadata
        self._is_folder = metadata["file_count"] > 1
        self._preview: Optional[np.ndarray] = None
        self._roi: Optional[pg.RectROI] = None
        self._last_auto_detect_path: Optional[Path] = None
        self._initial_params = initial_params or {}
        self._used_initial_params = bool(initial_params)
        self._setup_ui()
        self._initial_mask_rel = self._initial_params.get("mask_rel")
        self._initial_roi_state = self._initial_params.get("roi_state")
        datatype_est = self.metadata.get("datatype_est", "")
        is_uint8_source = datatype_est == "uint8"
        if initial_params:
            self.spin_resize.setValue(int(initial_params.get("size_ratio", 1.0) * 100))
            self.chk_8bit.setChecked(
                is_uint8_source or initial_params.get("convert_to_8_bit", False)
            )
            self.chk_normalize.setChecked(
                initial_params.get("normalize", False),
            )
            self.chk_row_number.blockSignals(True)
            self.chk_row_number.setChecked(
                initial_params.get("first_col_is_row_number", True)
            )
            delim = initial_params.get("delimiter", "\t")
            name = next((k for k, v in DELIMITERS.items() if v == delim), "Tab")
            idx = self.cmb_delimiter.findText(name)
            if idx >= 0:
                self.cmb_delimiter.setCurrentIndex(idx)
            if self._is_folder and "subset_ratio" in initial_params:
                self.spin_subset.setValue(initial_params["subset_ratio"])
            self.chk_row_number.blockSignals(False)
            self._apply_initial_transforms(initial_params)
        else:
            self.chk_8bit.setChecked(
                is_uint8_source or self.metadata.get("convert_to_8_bit_suggest", False)
            )
        set_checkbox_visibly_enabled(self.chk_8bit, not is_uint8_source)
        if is_uint8_source:
            self.chk_8bit.setToolTip("Source is already 8 bit")
        self.cmb_preview_mode.currentTextChanged.connect(self._refresh_preview)
        self.chk_preview_norm.toggled.connect(self._refresh_preview)
        self._update_estimates()
        self._refresh_preview()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)

        info_layout = QFormLayout()
        self.lbl_file = QLabel(self.metadata["file_name"])
        h, w = self.metadata["size"]
        self.lbl_dims = QLabel(f"{h} x {w}")
        self.lbl_count = QLabel(str(self.metadata["file_count"]))
        self.lbl_format = QLabel(self.metadata.get("format_display", "ASCII"))
        info_layout.addRow("File:", self.lbl_file)
        info_layout.addRow("Dimensions:", self.lbl_dims)
        if self._is_folder:
            info_layout.addRow("Files:", self.lbl_count)
        info_layout.addRow("Source format:", self.lbl_format)
        self.lbl_datatype = QLabel("")
        self.lbl_datatype.setToolTip("Estimated from first file")
        info_layout.addRow("Data type (est.):", self.lbl_datatype)
        layout.addLayout(info_layout)

        layout.addWidget(QLabel("<b>Loading Options</b>"))
        controls_layout = QFormLayout()

        self.chk_row_number = QCheckBox("First column = row number (ASC format)")
        self.chk_row_number.setChecked(True)
        self.chk_row_number.setToolTip("Uncheck for .dat without row index")
        controls_layout.addRow("", self.chk_row_number)

        self.cmb_delimiter = QComboBox()
        self.cmb_delimiter.addItems(list(DELIMITERS.keys()))
        self.cmb_delimiter.setCurrentText("Tab")
        controls_layout.addRow("Delimiter:", self.cmb_delimiter)

        self.spin_resize = QSpinBox()
        self.spin_resize.setRange(1, 100)
        self.spin_resize.setValue(100)
        self.spin_resize.setSuffix(" %")
        resize_row = QHBoxLayout()
        resize_row.addWidget(self.spin_resize)
        self.lbl_size_px = QLabel("")
        resize_row.addWidget(self.lbl_size_px)
        resize_row.addStretch()
        controls_layout.addRow("Resize:", resize_row)

        if self._is_folder:
            self.spin_subset = QDoubleSpinBox()
            self.spin_subset.setRange(0.01, 1.0)
            self.spin_subset.setValue(1.0)
            self.spin_subset.setSingleStep(0.1)
            self.spin_subset.setPrefix("Subset: ")
            self.spin_subset.setToolTip("Ratio of files to load (1.0 = all)")
            subset_row = QHBoxLayout()
            subset_row.addWidget(self.spin_subset)
            self.lbl_num_frames = QLabel("")
            subset_row.addWidget(self.lbl_num_frames)
            subset_row.addStretch()
            controls_layout.addRow("", subset_row)

        self.chk_8bit = QCheckBox("Convert to 8 bit")
        self.chk_8bit.setChecked(False)
        self.chk_normalize = QCheckBox("Normalize each image")
        self.chk_normalize.setChecked(False)
        self.chk_normalize.setToolTip(
            "On load: stretch each file to full range (per-file). "
            "If off: fixed scale 0-1 -> 0-255 (comparable brightness)."
        )
        controls_layout.addRow("", self.chk_8bit)
        controls_layout.addRow("", self.chk_normalize)
        layout.addLayout(controls_layout)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setFrameShadow(QFrame.Shadow.Sunken)
        layout.addWidget(sep)

        layout.addWidget(QLabel("<b>Preview / Crop</b>"))
        preview_opts = QHBoxLayout()
        self.cmb_preview_mode = QComboBox()
        self.cmb_preview_mode.addItems(["MAX (across samples)", "Single file"])
        self.cmb_preview_mode.setCurrentIndex(0)  # MAX default
        preview_opts.addWidget(QLabel("Mode:"))
        preview_opts.addWidget(self.cmb_preview_mode)
        self.chk_preview_norm = QCheckBox("Preview normalize")
        self.chk_preview_norm.setChecked(True)
        self.chk_preview_norm.setToolTip(
            "Display only in this dialog — min-max stretch for visibility. "
            "Does not change what is loaded."
        )
        preview_opts.addWidget(self.chk_preview_norm)
        self.btn_roi_reset = QPushButton("Reset ROI")
        self.btn_roi_reset.pressed.connect(self._reset_roi)
        preview_opts.addWidget(self.btn_roi_reset)
        preview_opts.addWidget(self._make_auto_crop_button())
        preview_opts.addStretch()
        layout.addLayout(preview_opts)

        preview_frame = QFrame()
        preview_frame.setFrameStyle(QFrame.Shape.StyledPanel | QFrame.Shadow.Sunken)
        preview_layout = QVBoxLayout(preview_frame)
        self.lbl_raw = QLabel("")
        self.lbl_raw.setWordWrap(False)
        self.lbl_raw.setStyleSheet("font-family: monospace; font-size: 10px;")
        self.lbl_raw.setMaximumHeight(70)
        preview_layout.addWidget(self.lbl_raw)
        self._plot_widget = pg.PlotWidget(background=get_dialog_preview_bg())
        self._plot_widget.setMinimumSize(400, 300)
        self._plot_widget.setAspectLocked(False)
        self._plot_widget.hideAxis("left")
        self._plot_widget.hideAxis("bottom")
        self._img_item = pg.ImageItem()
        self._plot_widget.addItem(self._img_item)
        preview_layout.addWidget(self._plot_widget)
        layout.addWidget(preview_frame)

        # ROI Mixin Controls
        self._setup_roi_controls(layout)

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
        self.chk_row_number.stateChanged.connect(self._refresh_preview)
        self.cmb_delimiter.currentTextChanged.connect(self._refresh_preview)
        self.chk_8bit.stateChanged.connect(self._update_estimates)
        if self._is_folder:
            self.spin_subset.valueChanged.connect(self._update_estimates)

    def _get_delimiter(self) -> str:
        return DELIMITERS.get(self.cmb_delimiter.currentText(), "\t")

    def _refresh_preview(self) -> None:
        files = get_ascii_files(self._path)
        if not files:
            self.lbl_raw.setText("(no files)")
            self.lbl_datatype.setText("—")
            self._img_item.setImage(np.zeros((1, 1), dtype=np.uint8))
            return

        preview_path = files[0]
        delimiter = self._get_delimiter()
        first_col = self.chk_row_number.isChecked()

        if preview_path != self._last_auto_detect_path:
            self._last_auto_detect_path = preview_path
            if not self._used_initial_params:
                raw_arr = parse_ascii_raw(preview_path, delimiter)
                if raw_arr is not None and raw_arr.shape[1] > 1:
                    self.chk_row_number.blockSignals(True)
                    self.chk_row_number.setChecked(first_col_looks_like_row_number(raw_arr))
                    first_col = self.chk_row_number.isChecked()
                    self.chk_row_number.blockSignals(False)

        mode = "max" if self.cmb_preview_mode.currentIndex() == 0 else "single"
        size_ratio = self.spin_resize.value() / 100.0
        normalize = self.chk_preview_norm.isChecked()
        est = estimate_ascii_datatype(self._path, delimiter, first_col)
        stats = est
        fmt = lambda x: f"{x:.4g}" if isinstance(x, (int, float)) and not (x != x) else str(x)
        stats_txt = f"min {fmt(stats['min'])}, max {fmt(stats['max'])}, med {fmt(stats['median'])}, mean {fmt(stats['mean'])}"
        self.lbl_datatype.setText(f"{est['dtype']} | {stats_txt}")
        img = get_ascii_preview(
            self._path, delimiter, first_col,
            size_ratio=size_ratio, mode=mode, normalize=normalize,
        )
        if img is None:
            self.lbl_raw.setText("(parse failed)")
            self.lbl_datatype.setText("—")
            self._img_item.setImage(np.zeros((1, 1), dtype=np.uint8))
            return

        self._preview = img
        self._apply_auto_transpose()

        lines = []
        with open(preview_path, encoding="utf-8", errors="replace") as f:
            for i, line in enumerate(f):
                if i >= RAW_PREVIEW_MAX_LINES:
                    break
                line = line.rstrip("\n\r")
                if len(line) > RAW_PREVIEW_MAX_CHARS:
                    line = line[:RAW_PREVIEW_MAX_CHARS] + "..."
                lines.append(line)
        self.lbl_raw.setText("\n".join(lines) if lines else "(empty)")

        # Apply transforms
        img = self._get_transformed_preview()

        cols, rows = img.shape[1], img.shape[0]
        display_img = np.swapaxes(img, 0, 1)
        self._img_item.setImage(display_img)
        self._img_item.setRect(pg.QtCore.QRectF(0, 0, cols, rows))
        self._img_item.setLookupTable(_plasma_lut())
        self._img_item.setLevels([0, 255])
        vb = self._plot_widget.getViewBox()
        vb.invertY(True)
        vb.setAspectLocked(lock=True, ratio=1.0)

        if self._roi is not None:
            self._plot_widget.removeItem(self._roi)

        self._roi = pg.RectROI((0, 0), (cols, rows), pen=pg.mkPen("lime", width=2))

        if self._initial_roi_state:
            s = self._initial_roi_state
            self._roi.setPos(s['pos'])
            self._roi.setSize(s['size'])
            self._roi.setAngle(0)
        elif self._initial_mask_rel is not None:
            r = self._initial_mask_rel
            x0 = max(0, int(r[0] * cols))
            y0 = max(0, int(r[1] * rows))
            x1 = min(cols, int(r[2] * cols))
            y1 = min(rows, int(r[3] * rows))
            if x1 > x0 and y1 > y0 and (x1 - x0 < cols or y1 - y0 < rows):
                self._roi.setPos((x0, y0))
                self._roi.setSize((x1 - x0, y1 - y0))

        self._roi.handleSize = 10
        self._plot_widget.addItem(self._roi)
        self._initial_mask_rel = None

        self._connect_roi_signals()
        self._update_estimates()

    def _reset_roi(self) -> None:
        self._reset_transform_checkboxes()
        if self._preview is not None and self._roi is not None:
            cols, rows = self._preview.shape[1], self._preview.shape[0]
            self._roi.setPos((0, 0))
            self._roi.setSize((cols, rows))
            self._roi.setAngle(0)
            self._update_estimates()

    def _update_estimates(self) -> None:
        if self._preview is None:
            return

        files = get_ascii_files(self._path)
        n_files = len(files)
        subset = self.spin_subset.value() if self._is_folder else 1.0
        n_load = max(1, int(n_files * subset)) if n_files > 0 else 1
        size_ratio = self.spin_resize.value() / 100.0

        orig_h, orig_w = self._preview.shape[0], self._preview.shape[1]

        mask, _ = self._get_roi_source_mask()
        if mask:
            # mask is (slice_x, slice_y) from mixin; for ASCII arr[row,col]: rows=mask[1], cols=mask[0]
            sl_row, sl_col = mask[1], mask[0]
            orig_h = sl_row.stop - sl_row.start
            orig_w = sl_col.stop - sl_col.start

        h_out = int(orig_h * size_ratio)
        w_out = int(orig_w * size_ratio)

        self.lbl_size_px.setText(f"-> {h_out} x {w_out} px")
        if self._is_folder:
            self.lbl_num_frames.setText(f"-> <b>{n_load}</b> files")
        channels = 1
        dtype_size = 1 if self.chk_8bit.isChecked() else 8
        total_bytes = h_out * w_out * channels * n_load * dtype_size
        total_gb = total_bytes / (1024**3)
        color = "green"
        if total_gb > get_available_ram() * 0.9:
            color = "red"
        elif total_gb > get_available_ram() * 0.7:
            color = "orange"

        self.lbl_ram_usage.setText(
            f"Estimated RAM: <font color='{color}'><b>{total_gb:.2f} GB</b></font>"
        )

    def get_params(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "size_ratio": self.spin_resize.value() / 100.0,
            "convert_to_8_bit": self.chk_8bit.isChecked(),
            "normalize": self.chk_normalize.isChecked(),
            "delimiter": self._get_delimiter(),
            "first_col_is_row_number": self.chk_row_number.isChecked(),
            **self._transform_params(),
        }
        if self._is_folder:
            out["subset_ratio"] = self.spin_subset.value()

        mask, mask_rel = self._get_roi_source_mask()
        if mask and mask_rel:
            # ROI mixin returns (slice_x, slice_y); ASCII arr[row,col] needs (slice_y, slice_x)
            out["mask"] = (mask[1], mask[0])
            out["mask_rel"] = mask_rel

        if self._roi is not None:
            state = self._roi.getState()
            img = self._get_transformed_preview()
            h_src, w_src = img.shape[0], img.shape[1] if img is not None else (0, 0)
            out["roi_state"] = {
                "pos": (state['pos'].x(), state['pos'].y()),
                "size": (state['size'].x(), state['size'].y()),
                "source_size": (h_src, w_src),
            }
        return out


# Display/capture fixed at 10 FPS until pipeline supports configurable FPS
_CAMERA_DISPLAY_FPS = 10


class CropTimelineDialog(QDialog):
    """Confirm timeline crop to the selected range. Non-destructive by default (mask)."""

    def __init__(
        self,
        start: int,
        end: int,
        n_total: int,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowIcon(QIcon(":/icon/blitz.ico"))
        self.setWindowTitle("Crop Timeline")
        self._start = start
        self._end = end
        self._n_total = n_total
        self._keep = True
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.addWidget(
            QLabel("Crop the timeline to the range set above.")
        )
        form = QFormLayout()
        self.lbl_start = QLabel(str(self._start))
        self.lbl_end = QLabel(str(self._end))
        n_frames = max(1, self._end - self._start + 1)
        self.lbl_frames = QLabel(f"{n_frames} (of {self._n_total})")
        form.addRow("Start index:", self.lbl_start)
        form.addRow("End index:", self.lbl_end)
        form.addRow("Frames:", self.lbl_frames)
        layout.addLayout(form)
        self.chk_keep = QCheckBox("Keep in RAM (reversible)")
        self.chk_keep.setChecked(True)
        self.chk_keep.setToolTip(
            "If checked: non-destructive mask, data stays in RAM, Undo restores all. "
            "If unchecked: frames outside range are discarded to save memory."
        )
        layout.addWidget(self.chk_keep)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_keep(self) -> bool:
        """Whether to use non-destructive mode (mask, reversible)."""
        return self.chk_keep.isChecked()


class RealCameraDialog(QDialog):
    """Webcam dialog with Buffer, Resolution. FPS fixed at 10 for now."""

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        on_frame: Optional[Any] = None,
        on_stop: Optional[Any] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowIcon(QIcon(":/icon/blitz.ico"))
        self.setWindowTitle("Webcam")
        self._on_frame = on_frame
        self._on_stop = on_stop
        self._handler: Any = None
        self._buffer_current = 0
        self._buffer_max = 1
        self._buffer_actual_sec: float | None = None
        self._last_capture_fps: float | None = None  # measured; used for buffer time
        self._setup_ui()
        self._update_estimates()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("<b>Webcam (cv2.VideoCapture)</b>"))

        form = QFormLayout()

        # Device (only indices that actually capture)
        self.cmb_device = QComboBox()
        self.cmb_device.setToolTip(
            "Working capture devices only. On Linux, some /dev/videoN "
            "nodes are metadata-only and are omitted."
        )
        form.addRow("Device:", self.cmb_device)
        self._refresh_devices()

        # Resolution
        self.cmb_res = QComboBox()
        self.cmb_res.addItems(["640x480", "800x600", "1024x768", "1280x720", "1920x1080"])
        self.cmb_res.setCurrentText("640x480")
        form.addRow("Resolution:", self.cmb_res)

        # FPS (capture rate; 10 fixed until measured)
        self.lbl_fps = QLabel("10 fps (fixed)")
        self.lbl_fps.setToolTip(
            "Capture rate (measured). 10 fps = init until first measurement. "
            "Used to assign buffer length to correct time span."
        )
        form.addRow("FPS:", self.lbl_fps)

        # Grayscale
        self.chk_grayscale = QCheckBox("Grayscale (1 Byte/px)")
        self.chk_grayscale.setChecked(True)
        form.addRow("", self.chk_grayscale)

        layout.addLayout(form)

        # Buffer Settings Group
        grp_buffer = QFrame()
        grp_buffer.setFrameStyle(QFrame.Shape.StyledPanel | QFrame.Shadow.Raised)
        buf_layout = QVBoxLayout(grp_buffer)
        buf_layout.addWidget(QLabel("<b>Buffer Settings</b>"))

        # Mode: Frames vs Seconds
        mode_layout = QHBoxLayout()
        self.radio_frames = QRadioButton("Frames")
        self.radio_frames.setChecked(True)
        self.radio_seconds = QRadioButton("Seconds")
        mode_layout.addWidget(self.radio_frames)
        mode_layout.addWidget(self.radio_seconds)
        mode_layout.addStretch()
        buf_layout.addLayout(mode_layout)

        # Value
        self.spin_buffer_val = QSpinBox()
        self.spin_buffer_val.setRange(1, 10000)
        self.spin_buffer_val.setValue(64)
        buf_layout.addWidget(self.spin_buffer_val)

        # RAM Estimate
        self.lbl_ram_est = QLabel("Est. RAM: ...")
        self.lbl_ram_est.setStyleSheet("color: #888; font-size: 9pt;")
        buf_layout.addWidget(self.lbl_ram_est)

        layout.addWidget(grp_buffer)

        # Buffer Status
        layout.addWidget(QLabel("Buffer Status:"))
        self.progress_buffer = QProgressBar()
        self.progress_buffer.setRange(0, 100)
        self.progress_buffer.setValue(0)
        self.progress_buffer.setFormat("%v / %m Frames")
        layout.addWidget(self.progress_buffer)

        # Buttons
        btn_row = QHBoxLayout()
        self.btn_toggle = QPushButton("Start")
        self.btn_toggle.setToolTip("Start or stop camera stream")
        self.btn_toggle.clicked.connect(self._on_toggle_clicked)
        btn_row.addWidget(self.btn_toggle)
        self.btn_close = QPushButton("Close")
        self.btn_close.setToolTip("Stop stream (if running) and close dialog")
        self.btn_close.clicked.connect(self.close)
        btn_row.addWidget(self.btn_close)
        layout.addLayout(btn_row)

        # Connections
        self.cmb_res.currentTextChanged.connect(self._update_estimates)
        self.chk_grayscale.toggled.connect(self._update_estimates)
        self.spin_buffer_val.valueChanged.connect(self._update_estimates)
        self.radio_frames.toggled.connect(self._on_mode_changed)
        self.radio_seconds.toggled.connect(self._on_mode_changed)

    def _refresh_devices(self) -> None:
        """Fill device combo with cameras that open + read a frame."""
        from ..data.live_camera import list_working_cameras
        from ..tools import log

        self.cmb_device.clear()
        devices = list_working_cameras()
        if not devices:
            self.cmb_device.addItem("No camera found", -1)
            self.cmb_device.setEnabled(False)
            log(
                "[CAM] No working camera devices found (checked indices 0–9).",
                color="orange",
            )
            return
        self.cmb_device.setEnabled(True)
        for device_id, label in devices:
            self.cmb_device.addItem(label, device_id)
        self.cmb_device.setCurrentIndex(0)

    def _on_mode_changed(self, checked: bool) -> None:
        """Handle switching between frames and seconds mode with value conversion."""
        if not checked:
            return

        val = self.spin_buffer_val.value()
        fps = self._last_capture_fps if self._last_capture_fps is not None else _CAMERA_DISPLAY_FPS

        # Block signals to prevent recursive update during conversion
        self.spin_buffer_val.blockSignals(True)

        if self.radio_seconds.isChecked():
            # Switched to Seconds (was Frames): frames / fps -> seconds
            new_val = max(1, int(val / fps))
        else:
            # Switched to Frames (was Seconds): seconds * fps -> frames
            new_val = int(val * fps)

        self.spin_buffer_val.setValue(new_val)
        self.spin_buffer_val.blockSignals(False)

        # Update estimates and suffixes
        self._update_estimates()

    def _update_estimates(self) -> None:
        """Calculate and display estimated RAM usage and buffer time span."""
        # 1. Resolution
        res_txt = self.cmb_res.currentText()
        if "x" in res_txt:
            w_s, h_s = res_txt.split("x")
            w, h = int(w_s), int(h_s)
        else:
            w, h = 640, 480

        # 2. Frames; use last capture FPS when available, else 10 fixed
        val = self.spin_buffer_val.value()
        fps = self._last_capture_fps if self._last_capture_fps is not None else _CAMERA_DISPLAY_FPS
        if self.radio_seconds.isChecked():
            frames = int(val * fps)
            suffix = " s"
        else:
            frames = val
            suffix = " frames"

        # Update spinbox suffix to match mode
        if self.spin_buffer_val.suffix() != suffix:
            self.spin_buffer_val.setSuffix(suffix)

        # 3. Time span: frames / fps = seconds
        time_sec = frames / fps

        # 4. Channels
        channels = 1 if self.chk_grayscale.isChecked() else 3

        # 5. RAM
        total_bytes = w * h * channels * frames
        mb = total_bytes / (1024**2)
        gb = total_bytes / (1024**3)

        color = "green"
        avail = get_available_ram()
        if gb > avail * 0.9:
            color = "red"
        elif gb > avail * 0.7:
            color = "orange"

        time_str = f"{time_sec:.1f} s" if time_sec >= 1 else f"{time_sec * 1000:.0f} ms"
        fps_note = f"@ {fps:.1f} fps" if self._last_capture_fps is not None else f"@ {fps} fps (fixed)"
        self.lbl_ram_est.setText(
            f"Frames: {frames} | {time_str} {fps_note} | Est. RAM: "
            f"<font color='{color}'><b>{mb:.0f} MB ({gb:.2f} GB)</b></font>"
        )

    def _on_toggle_clicked(self) -> None:
        from ..data.live_camera import RealCameraHandler
        if self._handler is None:
            self._start(RealCameraHandler)
        else:
            self._stop()

    def _start(self, HandlerClass) -> None:
        if self._handler is not None:
            return

        # Parse params
        res_txt = self.cmb_res.currentText()
        if "x" in res_txt:
            w_s, h_s = res_txt.split("x")
            w, h = int(w_s), int(h_s)
        else:
            w, h = 640, 480

        val = self.spin_buffer_val.value()
        fps = self._last_capture_fps if self._last_capture_fps is not None else _CAMERA_DISPLAY_FPS
        if self.radio_seconds.isChecked():
            frames = int(val * fps)
        else:
            frames = val
        frames = max(1, frames)

        device = self.cmb_device.currentData()
        if device is None or int(device) < 0:
            from ..tools import log
            log("[CAM] No working camera selected.", color="red")
            return
        device = int(device)
        gray = self.chk_grayscale.isChecked()

        self._handler = HandlerClass(
            parent=self,
            device_id=device,
            width=w,
            height=h,
            fps=float(fps),
            buffer_size=frames,
            grayscale=gray,
            exposure=0.5,
            gain=0.5,
            brightness=0.5,
            contrast=0.5,
            auto_exposure=True,
            send_live_only=True,
        )
        if self._on_frame:
            self._handler.frame_ready.connect(self._on_frame)

        self._handler.buffer_status.connect(self._on_buffer_status)
        self._handler.buffer_time_span_sec.connect(self._on_buffer_time_span)
        self._handler.stopped.connect(self._on_handler_stopped_ui)

        self._handler.start()
        self.btn_toggle.setText("Stop")
        self._set_stream_controls_enabled(False)

    def _on_handler_stopped_ui(self) -> None:
        """Worker exited (open failure or finished stop) — reset Start UI if still armed."""
        if self._handler is None:
            return
        handler = self._handler
        self._handler = None
        try:
            if self._on_frame:
                handler.frame_ready.disconnect(self._on_frame)
            handler.buffer_status.disconnect(self._on_buffer_status)
            handler.buffer_time_span_sec.disconnect(self._on_buffer_time_span)
            handler.stopped.disconnect(self._on_handler_stopped_ui)
        except (TypeError, RuntimeError):
            pass
        self._buffer_actual_sec = None
        self._update_fps_label()
        if self._on_stop:
            self._on_stop()
        self.btn_toggle.setText("Start")
        self._set_stream_controls_enabled(True)

    def _stop(self) -> None:
        handler = self._handler
        self._handler = None  # clear immediately to avoid re-entrancy (e.g. closeEvent)
        if handler is None:
            return
        try:
            handler.stopped.disconnect(self._on_handler_stopped_ui)
        except (TypeError, RuntimeError):
            pass
        handler.stop()
        if not handler.wait_stopped(4000):
            from ..tools import log
            log("[CAM] Timeout beim Stoppen", color="orange")
        try:
            if self._on_frame:
                handler.frame_ready.disconnect(self._on_frame)
            handler.buffer_status.disconnect(self._on_buffer_status)
            handler.buffer_time_span_sec.disconnect(self._on_buffer_time_span)
        except (TypeError, RuntimeError):
            pass
        self._buffer_actual_sec = None
        self._update_fps_label()
        if self._on_stop:
            self._on_stop()
        self.btn_toggle.setText("Start")
        self._set_stream_controls_enabled(True)

    def _on_buffer_status(self, current: int, max_val: int) -> None:
        self.progress_buffer.setMaximum(max_val)
        self.progress_buffer.setValue(current)
        self._buffer_current = current
        self._buffer_max = max_val
        self._update_streaming_display()

    def _on_buffer_time_span(self, sec: float) -> None:
        self._buffer_actual_sec = sec
        self._update_streaming_display()

    def _update_fps_label(self) -> None:
        """Show last capture FPS (for buffer time); 10 fps only as init."""
        if self._last_capture_fps is not None:
            self.lbl_fps.setText(f"~{self._last_capture_fps:.1f} fps (capture)")
        else:
            self.lbl_fps.setText("10 fps (fixed)")

    def _update_streaming_display(self) -> None:
        """Update RAM/time estimate during streaming with measured values."""
        if not self._handler or self._buffer_max < 1:
            return
        res_txt = self.cmb_res.currentText()
        if "x" in res_txt:
            w_s, h_s = res_txt.split("x")
            w, h = int(w_s), int(h_s)
        else:
            w, h = 640, 480
        channels = 1 if self.chk_grayscale.isChecked() else 3
        frames = self._buffer_current
        total_bytes = w * h * channels * frames
        mb = total_bytes / (1024**2)
        color = "green"
        avail = get_available_ram()
        if total_bytes / (1024**3) > avail * 0.9:
            color = "red"
        elif total_bytes / (1024**3) > avail * 0.7:
            color = "orange"
        if self._buffer_actual_sec is not None and self._buffer_actual_sec > 0:
            time_str = (
                f"{self._buffer_actual_sec:.2f} s (measured)"
                if self._buffer_actual_sec >= 0.01
                else f"{self._buffer_actual_sec * 1000:.0f} ms (measured)"
            )
            self._last_capture_fps = frames / self._buffer_actual_sec
        else:
            time_str = f"{frames / _CAMERA_DISPLAY_FPS:.1f} s (est.)"
        self._update_fps_label()
        self.lbl_ram_est.setText(
            f"Frames: {frames} / {self._buffer_max} | {time_str} | Est. RAM: "
            f"<font color='{color}'><b>{mb:.0f} MB</b></font>"
        )

    def _set_stream_controls_enabled(self, enabled: bool) -> None:
        self.btn_toggle.setEnabled(True)
        self.cmb_device.setEnabled(enabled)
        self.cmb_res.setEnabled(enabled)
        self.spin_buffer_val.setEnabled(enabled)
        self.radio_frames.setEnabled(enabled)
        self.radio_seconds.setEnabled(enabled)
        self.chk_grayscale.setEnabled(enabled)

    def stop_stream(self) -> None:
        self._stop()

    def closeEvent(self, event) -> None:
        self._stop()
        super().closeEvent(event)


class _FolderChooserPreviewLoader(QThread):
    """Background preview for one file (or group fallback) in FolderLoadChooserDialog."""

    finished_preview = pyqtSignal(object, str, int)  # img, badge, generation

    def __init__(self, group, paths: list, generation: int, preview_index: int = 0):
        super().__init__()
        self._group = group
        self._paths = list(paths)
        self._generation = generation
        self._preview_index = preview_index

    def run(self):
        img = None
        badge = ""
        g = self._group
        paths = self._paths
        if not paths:
            self.finished_preview.emit(None, "", self._generation)
            return
        idx = max(0, min(self._preview_index, len(paths) - 1))
        path = paths[idx]
        try:
            if g.kind == "hikmicro_celsius":
                from ..data.converters.hikmicro import celsius_preview

                img = celsius_preview(path, size_ratio=0.35)
                badge = f"HIKMICRO °C — {path.name} ({idx + 1}/{len(paths)})"
            elif g.kind == "ascii":
                from ..data.converters.ascii import (
                    first_col_looks_like_row_number,
                    get_ascii_preview,
                    parse_ascii_raw,
                )

                delim = "\t"
                raw = parse_ascii_raw(path, delim)
                if raw is None:
                    for d in (",", " "):
                        raw = parse_ascii_raw(path, d)
                        if raw is not None:
                            delim = d
                            break
                first_col = False
                if raw is not None and raw.shape[1] > 1:
                    first_col = first_col_looks_like_row_number(raw)
                img = get_ascii_preview(
                    path,
                    delimiter=delim,
                    first_col_is_row_number=first_col,
                    size_ratio=0.5,
                    mode="single",
                    normalize=True,
                )
                if raw is not None:
                    badge = f"{path.name} — ASCII {raw.shape[0]}x{raw.shape[1]} ({idx + 1}/{len(paths)})"
                else:
                    badge = f"{path.name} ({idx + 1}/{len(paths)})"
            elif g.kind == "array":
                arr = np.load(path, mmap_mode="r")
                badge = f"{path.name} — shape={arr.shape} dtype={arr.dtype} ({idx + 1}/{len(paths)})"
                sample = arr[0] if arr.ndim >= 3 else arr
                sample = np.asarray(sample, dtype=np.float32)
                if sample.ndim == 2:
                    lo, hi = float(np.nanmin(sample)), float(np.nanmax(sample))
                    if hi > lo:
                        img = ((sample - lo) / (hi - lo) * 255).clip(0, 255).astype(np.uint8)
                elif sample.ndim == 3 and sample.shape[-1] in (3, 4):
                    img = sample[..., :3]
                    lo, hi = float(np.nanmin(img)), float(np.nanmax(img))
                    if hi > lo:
                        img = ((img - lo) / (hi - lo) * 255).clip(0, 255).astype(np.uint8)
                    else:
                        img = np.clip(img, 0, 255).astype(np.uint8)
            elif g.kind == "video":
                img = get_video_preview(
                    path, n_frames=6, size_ratio=0.25, mode="single", normalize=True
                )
                badge = f"{path.name} ({idx + 1}/{len(paths)})"
            elif g.kind == "image":
                img = get_paths_image_preview(
                    [path], size_ratio=0.3, n_samples=1, mode="single", normalize=True
                )
                badge = f"{path.name} ({idx + 1}/{len(paths)})"
        except Exception as exc:
            badge = f"Preview failed ({path.name}): {exc}"
            img = None
        self.finished_preview.emit(img, badge, self._generation)


class FolderLoadChooserDialog(QDialog):
    """Pick which file group to load from a mixed folder (with file list + preview)."""

    def __init__(
        self,
        folder: Path,
        groups: list,
        parent: Optional[QWidget] = None,
        initial_group_id: Optional[str] = None,
    ):
        super().__init__(parent)
        self.setWindowIcon(QIcon(":/icon/blitz.ico"))
        self.setWindowTitle("Choose what to load")
        self._folder = folder
        self._groups = list(groups)
        self._selected: Any = None
        self._preview_loader: Optional[_FolderChooserPreviewLoader] = None
        self._preview_threads = _PreviewThreadGuard()
        self._preview_generation = 0
        self._preview_debounce: Optional[QTimer] = None
        self._current_paths: list = []
        self._setup_ui()
        start_row = 0
        if initial_group_id:
            for i, g in enumerate(self._groups):
                if g.id == initial_group_id:
                    start_row = i
                    break
        else:
            for i, g in enumerate(self._groups):
                if g.kind != "other":
                    start_row = i
                    break
        self.list.setCurrentRow(start_row)
        self._on_group_changed()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(f"<b>{self._folder.name}</b> — select a group, browse files:"))
        body = QHBoxLayout()

        left = QVBoxLayout()
        left.addWidget(QLabel("Groups"))
        self.list = QListWidget()
        self.list.setMinimumWidth(220)
        for g in self._groups:
            item = QListWidgetItem(g.label)
            item.setData(Qt.ItemDataRole.UserRole, g.id)
            if g.kind == "other":
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEnabled)
            self.list.addItem(item)
        self.list.currentRowChanged.connect(lambda _r: self._on_group_changed())
        left.addWidget(self.list, stretch=1)
        body.addLayout(left)

        mid = QVBoxLayout()
        mid.addWidget(QLabel("Files in group"))
        self.file_list = QListWidget()
        self.file_list.setMinimumWidth(240)
        self.file_list.currentRowChanged.connect(lambda _r: self._on_file_changed())
        mid.addWidget(self.file_list, stretch=1)
        nav = QHBoxLayout()
        self.btn_prev = QPushButton("◀")
        self.btn_prev.setFixedWidth(36)
        self.btn_prev.setToolTip("Previous file")
        self.btn_next = QPushButton("▶")
        self.btn_next.setFixedWidth(36)
        self.btn_next.setToolTip("Next file")
        self.lbl_file_pos = QLabel("")
        self.btn_prev.clicked.connect(self._prev_file)
        self.btn_next.clicked.connect(self._next_file)
        nav.addWidget(self.btn_prev)
        nav.addWidget(self.lbl_file_pos, stretch=1)
        nav.addWidget(self.btn_next)
        mid.addLayout(nav)
        body.addLayout(mid)

        preview_col = QVBoxLayout()
        self.lbl_info = QLabel("")
        self.lbl_info.setWordWrap(True)
        preview_col.addWidget(self.lbl_info)
        self._plot_widget = pg.PlotWidget(background=get_dialog_preview_bg())
        self._plot_widget.setMinimumSize(320, 240)
        self._plot_widget.hideAxis("left")
        self._plot_widget.hideAxis("bottom")
        self._plot_widget.setAspectLocked(lock=True, ratio=1.0)
        self._img_item = pg.ImageItem()
        self._plot_widget.addItem(self._img_item)
        preview_col.addWidget(self._plot_widget, stretch=1)
        self.lbl_badge = QLabel("")
        self.lbl_badge.setStyleSheet("color: gray;")
        self.lbl_badge.setWordWrap(True)
        preview_col.addWidget(self.lbl_badge)
        body.addLayout(preview_col, stretch=1)
        layout.addLayout(body)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        self._ok_btn = buttons.button(QDialogButtonBox.StandardButton.Ok)
        layout.addWidget(buttons)
        self.resize(960, 480)

    def _current_group(self):
        row = self.list.currentRow()
        if row < 0 or row >= len(self._groups):
            return None
        g = self._groups[row]
        if g.kind == "other":
            return None
        return g

    def _clear_preview(self) -> None:
        self._img_item.clear()
        self.lbl_badge.setText("")

    def _stop_preview_loader(self) -> None:
        self._preview_threads.stop(
            finished_slot=self._on_preview, signal_name="finished_preview"
        )
        self._preview_loader = None

    def _on_group_changed(self) -> None:
        g = self._current_group()
        self._ok_btn.setEnabled(g is not None)
        self.file_list.blockSignals(True)
        self.file_list.clear()
        self._current_paths = []
        if g is None:
            self.file_list.blockSignals(False)
            self.lbl_info.setText("Select a loadable group.")
            self.lbl_file_pos.setText("")
            self._clear_preview()
            self._preview_generation += 1
            self._stop_preview_loader()
            return
        extra = ""
        if g.kind == "hikmicro_celsius":
            extra = " Approximate °C from radiometric JPEG."
        elif g.naming_cluster:
            extra = f" Naming cluster: {g.naming_cluster}."
        self.lbl_info.setText(
            f"<b>{g.label}</b><br/>{g.count} file(s), kind={g.kind}.{extra}"
        )
        self._current_paths = list(g.paths)
        for p in self._current_paths:
            self.file_list.addItem(p.name)
        self.file_list.blockSignals(False)
        if self._current_paths:
            self.file_list.setCurrentRow(0)
        else:
            self.lbl_file_pos.setText("")
            self._clear_preview()

    def _on_file_changed(self) -> None:
        self._update_file_pos_label()
        g = self._current_group()
        if g is None or not self._current_paths:
            return
        self._preview_generation += 1
        self.lbl_badge.setText("Loading preview...")
        self._stop_preview_loader()
        if self._preview_debounce is None:
            self._preview_debounce = QTimer(self)
            self._preview_debounce.setSingleShot(True)
            self._preview_debounce.timeout.connect(self._restart_preview_debounced)
        self._preview_debounce.start(60)

    def _restart_preview_debounced(self) -> None:
        g = self._current_group()
        if g is None:
            return
        self._start_preview(g, self.file_list.currentRow())

    def _update_file_pos_label(self) -> None:
        n = len(self._current_paths)
        i = self.file_list.currentRow()
        if n <= 0 or i < 0:
            self.lbl_file_pos.setText("")
            self.btn_prev.setEnabled(False)
            self.btn_next.setEnabled(False)
            return
        self.lbl_file_pos.setText(f"{i + 1} / {n}")
        self.btn_prev.setEnabled(i > 0)
        self.btn_next.setEnabled(i < n - 1)

    def _prev_file(self) -> None:
        row = self.file_list.currentRow()
        if row > 0:
            self.file_list.setCurrentRow(row - 1)

    def _next_file(self) -> None:
        row = self.file_list.currentRow()
        if 0 <= row < self.file_list.count() - 1:
            self.file_list.setCurrentRow(row + 1)

    def _start_preview(self, g, file_index: int = 0) -> None:
        self._stop_preview_loader()
        self._preview_generation += 1
        generation = self._preview_generation
        self.lbl_badge.setText("Loading preview...")
        loader = _FolderChooserPreviewLoader(
            g, self._current_paths, generation, preview_index=file_index
        )
        self._preview_loader = self._preview_threads.adopt(loader)
        loader.finished_preview.connect(self._on_preview)
        loader.start()

    def _on_preview(self, img, badge: str, generation: int) -> None:
        sender = self.sender()
        if generation != self._preview_generation:
            if sender is not None and sender is self._preview_loader:
                self._preview_threads.clear_current_if(self._preview_loader)
                self._preview_loader = None
            return
        if sender is not None and sender is not self._preview_loader:
            return
        self._preview_threads.clear_current_if(self._preview_loader)
        self._preview_loader = None
        self.lbl_badge.setText(badge)
        if img is None:
            self._img_item.clear()
            return
        if img.ndim == 2:
            display = img.T
            self._img_item.setLookupTable(_plasma_lut())
            self._img_item.setLevels([0, 255])
        else:
            display = np.transpose(img, (1, 0, 2))
            self._img_item.setLookupTable(None)
        self._img_item.setImage(display)
        h, w = display.shape[0], display.shape[1]
        self._img_item.setRect(pg.QtCore.QRectF(0, 0, w, h))
        vb = self._plot_widget.getViewBox()
        vb.invertY(True)
        pad = 1.05
        mx = w * (pad - 1) / 2
        my = h * (pad - 1) / 2
        self._plot_widget.setRange(
            xRange=(-mx, w + mx),
            yRange=(-my, h + my),
            padding=0,
        )

    def selected_group(self):
        return self._current_group()

    def accept(self) -> None:
        self._selected = self._current_group()
        if self._selected is None:
            return
        if self._preview_debounce is not None:
            self._preview_debounce.stop()
        self._stop_preview_loader()
        self._preview_threads.stop_all(
            finished_slot=self._on_preview, signal_name="finished_preview"
        )
        super().accept()

    def reject(self) -> None:
        if self._preview_debounce is not None:
            self._preview_debounce.stop()
        self._stop_preview_loader()
        self._preview_threads.stop_all(
            finished_slot=self._on_preview, signal_name="finished_preview"
        )
        super().reject()

    def closeEvent(self, event) -> None:
        if self._preview_debounce is not None:
            self._preview_debounce.stop()
        self._stop_preview_loader()
        self._preview_threads.stop_all(
            finished_slot=self._on_preview, signal_name="finished_preview"
        )
        super().closeEvent(event)


class MixedImageSizesDialog(QDialog):
    """Ask how to handle unequal HxW in a folder load."""

    def __init__(
        self,
        shapes: set,
        n_files: int,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.setWindowIcon(QIcon(":/icon/blitz.ico"))
        self.setWindowTitle("Mixed image sizes")
        self.policy: Optional[str] = None
        layout = QVBoxLayout(self)
        shape_txt = ", ".join(f"{h}x{w}" for h, w in sorted(shapes)[:8])
        if len(shapes) > 8:
            shape_txt += ", ..."
        layout.addWidget(
            QLabel(
                f"Images in this set have different sizes ({n_files} files).\n"
                f"Detected: {shape_txt}\n\n"
                "How should BLITZ proceed?"
            )
        )
        self.radio_crop = QRadioButton("Crop all to common minimum size (lossy edges)")
        self.radio_abort = QRadioButton("Cancel load")
        self.radio_crop.setChecked(True)
        layout.addWidget(self.radio_crop)
        layout.addWidget(self.radio_abort)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._on_ok)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _on_ok(self) -> None:
        if self.radio_abort.isChecked():
            self.policy = None
            self.reject()
            return
        self.policy = "crop_min"
        self.accept()
