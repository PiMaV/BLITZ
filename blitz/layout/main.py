import json
import time

import numpy as np
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import QCoreApplication, Qt, QTimer, QUrl
from PyQt6.QtGui import QDesktopServices, QKeySequence, QShortcut
from PyQt6.QtWidgets import (QApplication, QDialog, QFileDialog, QMainWindow,
                             QTableWidgetItem)
import pyqtgraph as pg
from pyqtgraph.graphicsItems.GradientEditorItem import Gradients

from .. import settings
from ..theme import get_style
from ..data import optimized
from ..data.image import ImageData
from ..data.load import DataLoader, get_image_metadata, get_sample_format
from ..data.web import WebDataLoader
from ..data.ops import ReduceOperation
from ..tools import (LoadingManager, format_size_mb, get_available_ram,
                     get_cpu_percent, get_disk_io_mbs, get_used_ram, log)


def _pca_sync_vb2(plot_widget) -> None:
    """Sync second ViewBox geometry for dual-axis PCA plot."""
    vb2 = getattr(plot_widget, "_pca_vb2", None)
    if vb2 is None:
        return
    pi = plot_widget.getPlotItem()
    vb2.setGeometry(pi.getViewBox().sceneBoundingRect())
    vb2.linkedViewChanged(pi.getViewBox(), vb2.XAxis)


def _numba_active(data) -> bool:
    """True if pipeline (subtract/divide) or aggregation (MEAN/MAX/MIN/STD) uses Numba."""
    use_numba = getattr(data, "use_numba", True)
    if not use_numba or not optimized.HAS_NUMBA:
        return False
    pipeline = getattr(data, "_ops_pipeline", None)
    if pipeline:
        for step in (pipeline.get("subtract"), pipeline.get("divide")):
            if step and step.get("amount", 0) > 0:
                return True
    redop = getattr(data, "_redop", None)
    if redop is not None:
        name = redop.name if isinstance(redop, ReduceOperation) else str(redop)
        if name.upper() in ("MEAN", "MAX", "MIN", "STD"):
            return True
    return False


def _set_numba_status(label, data) -> None:
    """Set Numba status label for Bench tab."""
    if not optimized.HAS_NUMBA:
        label.setText("Numba: unavailable")
    elif data is None:
        label.setText("Numba: —")
    else:
        on = _numba_active(data)
        label.setText("Numba: on" if on else "Numba: off")
from ..data.converters import get_ascii_metadata, load_ascii
from .dialogs import (AsciiLoadOptionsDialog, CropTimelineDialog,
                     ImageLoadOptionsDialog, RealCameraDialog,
                     VideoLoadOptionsDialog)
from .rosee import ROSEEAdapter
from .simulated_live import SimulatedLiveWidget
from .tof import TOFAdapter
from .pca import PCAAdapter
from .ui import UI_MainWindow

URL_GITHUB = QUrl("https://github.com/CodeSchmiedeHGW/BLITZ")
URL_INP = QUrl("https://www.inp-greifswald.de/")
URL_MESS = QUrl("https://mess.engineering/")


def restart() -> None:
    QCoreApplication.exit(settings.get("app/restart_exit_code"))


class MainWindow(QMainWindow):

    def __init__(self) -> None:
        super().__init__()

        self.ui = UI_MainWindow()
        self.ui.setup_UI(self)

        self.last_file_dir = Path.cwd()
        self.last_file: str = ""
        self._video_session_defaults: dict = {}
        self._image_session_defaults: dict = {}
        self._ascii_session_defaults: dict = {}
        self._simulated_live: SimulatedLiveWidget | None = None
        self._real_camera_dialog: RealCameraDialog | None = None
        self._aggregate_first_open: bool = True

        self.pca_adapter = PCAAdapter(self.ui.image_viewer)
        self.tof_adapter = TOFAdapter(self.ui.roi_plot)
        self.rosee_adapter = ROSEEAdapter(
            self.ui.image_viewer,
            self.ui.h_plot,
            self.ui.v_plot,
            (self.ui.textbox_rosee_interval_h,
             self.ui.textbox_rosee_interval_v),
            (self.ui.textbox_rosee_slope_h,
             self.ui.textbox_rosee_slope_v),
        )
        self.setup_connections()
        self.reset_options()
        self.setup_sync()

        self.ui.checkbox_roi.setChecked(False)
        self.ui.checkbox_roi.setChecked(True)
        log("Welcome to BLITZ", color=(122, 162, 247))

    def _set_theme_and_restart(self, theme: str) -> None:
        settings.set("app/theme", theme)
        restart()

    def closeEvent(self, event):
        if self._simulated_live:
            self._simulated_live.stop_stream()
        if self._real_camera_dialog:
            self._real_camera_dialog.stop_stream()
        self.save_settings()
        event.accept()

    def setup_connections(self) -> None:
        # MainWindow connections
        self.shortcut_copy = QShortcut(QKeySequence("Ctrl+C"), self)
        self.shortcut_copy.activated.connect(self.on_strgC)

        # menu connections
        self.ui.action_open_file.triggered.connect(self.browse_file)
        self.ui.action_open_folder.triggered.connect(self.browse_folder)
        self.ui.button_load_file.clicked.connect(self.browse_file)
        self.ui.button_load_folder.clicked.connect(self.browse_folder)
        self.ui.action_load_tof.triggered.connect(self.browse_tof)
        self.ui.action_export.triggered.connect(self.export)
        self.ui.action_restart.triggered.connect(restart)
        self.ui.action_theme_dark.triggered.connect(
            lambda: self._set_theme_and_restart("dark")
        )
        self.ui.action_theme_light.triggered.connect(
            lambda: self._set_theme_and_restart("light")
        )
        self.ui.action_link_inp.triggered.connect(
            lambda: QDesktopServices.openUrl(URL_INP)  # type: ignore
        )
        self.ui.action_link_github.triggered.connect(
            lambda: QDesktopServices.openUrl(URL_GITHUB)  # type: ignore
        )

        # image_viewer connections
        self.ui.image_viewer.file_dropped.connect(self.load)
        self.ui.roi_plot.crop_range.sigRegionChanged.connect(
            self.update_crop_range_labels
        )
        self.ui.roi_plot.crop_range.sigRegionChanged.connect(
            self._crop_region_to_frame
        )
        self.ui.roi_plot.crop_range.sigRegionChangeFinished.connect(
            self._on_selection_changed
        )
        self.ui.roi_plot.crop_range.sigRegionChangeFinished.connect(
            self._on_crop_range_for_ops
        )
        self.ui.image_viewer.scene.sigMouseMoved.connect(
            self.update_statusbar_position
        )
        self.ui.image_viewer.timeLine.sigPositionChanged.connect(
            self.update_statusbar
        )
        self.ui.image_viewer.timeLine.sigPositionChanged.connect(
            self._sync_current_frame_spinbox
        )
        self.ui.image_viewer.timeLine.sigPositionChanged.connect(
            self._on_timeline_for_sliding_preview
        )
        self.ui.spinbox_current_frame.valueChanged.connect(
            self._on_current_frame_spinbox_changed
        )

        # lut connections
        self.ui.button_autofit.clicked.connect(self.ui.image_viewer.autoLevels)
        self.ui.checkbox_auto_fit.stateChanged.connect(
            lambda: self.ui.image_viewer.set_auto_fit(
                self.ui.checkbox_auto_fit.isChecked()
            )
        )
        self.ui.checkbox_auto_colormap.stateChanged.connect(
            self.ui.image_viewer.toggle_auto_colormap
        )
        self.ui.button_load_lut.pressed.connect(self.browse_lut)
        self.ui.button_export_lut.pressed.connect(self.save_lut)
        self.ui.spin_lut_min.valueChanged.connect(self._on_lut_spin_changed)
        self.ui.spin_lut_max.valueChanged.connect(self._on_lut_spin_changed)
        self._lut_sync_timer = QTimer(self)
        self._lut_sync_timer.timeout.connect(self._sync_lut_spinners)
        self._lut_sync_timer.start(80)
        self.ui.image_viewer.image_changed.connect(
            lambda: QTimer.singleShot(50, self._sync_lut_spinners)
        )
        self.ui.image_viewer.image_changed.connect(
            lambda: QTimer.singleShot(60, self._apply_lut_log_state)
        )
        self.ui.checkbox_lut_log.stateChanged.connect(self._on_lut_log_changed)
        self._sync_lut_spinners()

        # option connections
        self.ui.button_connect.pressed.connect(self.start_web_connection)
        self.ui.button_disconnect.pressed.connect(
            lambda: self.end_web_connection(None)
        )
        self.ui.button_simulated_live.pressed.connect(self.show_simulated_live)
        self.ui.button_real_camera.pressed.connect(self.show_real_camera_dialog)
        self.ui.checkbox_flipx.clicked.connect(
            lambda: self.ui.image_viewer.manipulate("flip_x")
        )
        self.ui.checkbox_flipy.clicked.connect(
            lambda: self.ui.image_viewer.manipulate("flip_y")
        )
        self.ui.checkbox_rotate_90.clicked.connect(
            lambda: self.ui.image_viewer.manipulate("rotate_90")
        )
        self.ui.checkbox_mask.clicked.connect(self.ui.image_viewer.toggle_mask)
        self.ui.button_apply_mask.pressed.connect(self.apply_mask)
        self.ui.button_apply_mask.pressed.connect(
            lambda: self.ui.checkbox_mask.setChecked(False)
        )
        self.ui.button_reset_mask.pressed.connect(self.reset_mask)
        self.ui.button_image_mask.pressed.connect(self.image_mask)
        self.ui.checkbox_crosshair.stateChanged.connect(
            self.ui.h_plot.toggle_line
        )
        self.ui.checkbox_crosshair.stateChanged.connect(
            self.ui.v_plot.toggle_line
        )
        self.ui.checkbox_crosshair_marking.stateChanged.connect(
            self.toggle_hvplot_markings
        )
        self.ui.spinbox_width_h.valueChanged.connect(
            self.ui.h_plot.change_width
        )
        self.ui.spinbox_width_v.valueChanged.connect(
            self.ui.v_plot.change_width
        )
        self.ui.checkbox_roi.stateChanged.connect(
            self.ui.image_viewer.roiClicked
        )
        self.ui.combobox_roi.currentIndexChanged.connect(self.change_roi)
        self.ui.checkbox_tof.stateChanged.connect(
            self.tof_adapter.toggle_plot
        )
        self.ui.checkbox_roi_drop.stateChanged.connect(
            lambda: self.ui.image_viewer.toggle_roi_update_frequency(
                self.ui.checkbox_roi_drop.isChecked()
            )
        )
        self.ui.spinbox_crop_range_start.editingFinished.connect(
            self.update_crop_range
        )
        self.ui.spinbox_crop_range_end.editingFinished.connect(
            self.update_crop_range
        )
        self.ui.spinbox_selection_window.editingFinished.connect(
            self._on_selection_changed
        )
        self.ui.spinbox_crop_range_start.valueChanged.connect(
            lambda: self._sync_selection_window_from_range(from_start=True)
        )
        self.ui.spinbox_crop_range_end.valueChanged.connect(
            lambda: self._sync_selection_window_from_range(from_start=False)
        )
        self.ui.checkbox_window_const.stateChanged.connect(
            lambda: self._sync_selection_from_window_const()
        )
        self.ui.spinbox_crop_range_start.valueChanged.connect(
            self._crop_index_to_frame
        )
        self.ui.spinbox_crop_range_end.valueChanged.connect(
            self._crop_index_to_frame
        )
        self.ui.spinbox_crop_range_start.valueChanged.connect(
            self._update_full_range_button_style
        )
        self.ui.spinbox_crop_range_end.valueChanged.connect(
            self._update_full_range_button_style
        )
        self.ui.spinbox_crop_range_start.editingFinished.connect(
            self._on_selection_changed
        )
        self.ui.spinbox_crop_range_end.editingFinished.connect(
            self._on_selection_changed
        )
        self.ui.spinbox_selection_window.valueChanged.connect(
            self._sync_selection_range_from_window
        )
        self.ui.button_reset_range.clicked.connect(self.reset_selection_range)
        self.ui.button_crop.clicked.connect(self.crop)
        self.ui.button_ops_crop.clicked.connect(self._open_crop_dialog)
        self.ui.button_ops_undo_crop.clicked.connect(self._undo_crop)
        self.ui.combobox_ops_subtract_src.currentIndexChanged.connect(
            self._update_ops_file_visibility
        )
        self.ui.combobox_ops_subtract_src.currentIndexChanged.connect(
            self._update_ops_norm_visibility
        )
        self.ui.combobox_ops_divide_src.currentIndexChanged.connect(
            self._update_ops_file_visibility
        )
        self.ui.combobox_ops_divide_src.currentIndexChanged.connect(
            self._update_ops_norm_visibility
        )
        self.ui.spinbox_ops_norm_window.valueChanged.connect(self.apply_ops)
        self.ui.spinbox_ops_norm_lag.valueChanged.connect(self.apply_ops)
        self.ui.checkbox_ops_sliding_apply_full.stateChanged.connect(self.apply_ops)
        for cb in (self.ui.combobox_ops_subtract_src, self.ui.combobox_ops_divide_src):
            cb.currentIndexChanged.connect(self.apply_ops)
        self.ui.combobox_ops_range_method.currentIndexChanged.connect(self.apply_ops)
        self.ui.slider_ops_subtract.valueChanged.connect(
            self._update_ops_slider_labels
        )
        self.ui.slider_ops_subtract.valueChanged.connect(self.apply_ops)
        self.ui.slider_ops_divide.valueChanged.connect(
            self._update_ops_slider_labels
        )
        self.ui.slider_ops_divide.valueChanged.connect(self.apply_ops)
        self.ui.button_ops_load_file.clicked.connect(self.load_ops_file)
        self.ui.spinbox_crop_range_start.editingFinished.connect(self.apply_ops)
        self.ui.spinbox_crop_range_end.editingFinished.connect(self.apply_ops)
        self.ui.combobox_reduce.currentIndexChanged.connect(self.apply_ops)
        self.ui.roi_plot.clicked_to_set_frame.connect(
            self._on_timeline_clicked_to_frame
        )
        self.ui.checkbox_timeline_bands.stateChanged.connect(
            self._on_timeline_options_changed
        )
        self.ui.combobox_timeline_aggregation.currentIndexChanged.connect(
            self._on_timeline_options_changed
        )
        self.ui.checkbox_agg_update_on_drag.stateChanged.connect(
            self._on_agg_update_on_drag_changed
        )
        self._on_agg_update_on_drag_changed()
        self.ui.combobox_reduce.currentIndexChanged.connect(
            self._on_reduce_changed
        )
        self.ui.spinbox_selection_window.valueChanged.connect(
            self._sync_selection_range_from_window
        )
        self.ui.checkbox_measure_roi.stateChanged.connect(
            self.ui.measure_roi.toggle
        )
        self.ui.checkbox_show_bounding_rect.stateChanged.connect(
            self.ui.measure_roi.toggle_bounding_rect
        )
        self.ui.checkbox_mm.stateChanged.connect(self.update_roi_settings)
        self.ui.spinbox_pixel.valueChanged.connect(self.update_roi_settings)
        self.ui.spinbox_mm.valueChanged.connect(self.update_roi_settings)
        self.ui.checkbox_minmax_per_image.stateChanged.connect(
            self._update_envelope_options
        )
        self.ui.checkbox_envelope_per_crosshair.stateChanged.connect(
            self._update_envelope_options
        )
        self.ui.checkbox_envelope_per_dataset.stateChanged.connect(
            self._update_envelope_options
        )
        self.ui.spinbox_envelope_pct.valueChanged.connect(
            self._update_envelope_options
        )
        self.ui.checkbox_rosee_active.stateChanged.connect(self.toggle_rosee)
        self.ui.checkbox_rosee_active.stateChanged.connect(
            self.update_isocurves
        )
        self.ui.checkbox_rosee_h.stateChanged.connect(self.toggle_rosee)
        self.ui.checkbox_rosee_v.stateChanged.connect(self.toggle_rosee)
        self.ui.checkbox_rosee_local_extrema.stateChanged.connect(
            self.toggle_rosee
        )
        self.ui.spinbox_rosee_smoothing.valueChanged.connect(self.toggle_rosee)
        self.ui.checkbox_rosee_normalize.stateChanged.connect(
            self.toggle_rosee
        )
        self.ui.checkbox_rosee_show_indices.stateChanged.connect(
            self.toggle_rosee
        )
        self.ui.checkbox_rosee_in_image_h.stateChanged.connect(
            self.toggle_rosee
        )
        self.ui.checkbox_rosee_in_image_v.stateChanged.connect(
            self.toggle_rosee
        )
        self.ui.h_plot._extractionline.sigPositionChanged.connect(
            self.toggle_rosee
        )
        self.ui.v_plot._extractionline.sigPositionChanged.connect(
            self.toggle_rosee
        )
        self.ui.image_viewer.timeLine.sigPositionChanged.connect(
            self.toggle_rosee
        )
        self.ui.checkbox_rosee_show_lines.stateChanged.connect(
            self.toggle_rosee
        )
        self.ui.checkbox_show_isocurve.stateChanged.connect(
            self.update_isocurves
        )
        self.ui.button_pca_calc.clicked.connect(self.pca_calculate)
        self.ui.button_pca_show.clicked.connect(self.pca_toggle_show)
        self.ui.checkbox_pca_exact.stateChanged.connect(self._pca_on_exact_changed)
        self.ui.spinbox_pcacomp.valueChanged.connect(self.pca_update_view)
        self.ui.combobox_pca.currentIndexChanged.connect(self._pca_on_mode_changed)
        self.ui.checkbox_pca_include_mean.stateChanged.connect(self.pca_update_view)
        self.pca_adapter.started.connect(self.pca_on_started)
        self.pca_adapter.finished.connect(self.pca_on_finished)
        self.pca_adapter.error.connect(self.pca_on_error)
        self.ui.spinbox_isocurves.editingFinished.connect(
            self.update_isocurves
        )
        self.ui.spinbox_iso_smoothing.editingFinished.connect(
            self.update_isocurves
        )
        self.ui.image_viewer.image_changed.connect(self.update_bench)
        self.ui.image_viewer.image_changed.connect(self._update_selection_visibility)
        self.ui.image_viewer.image_size_changed.connect(self.update_bench)
        self._bench_timer = QTimer(self)
        self._bench_timer.timeout.connect(self._bench_tick)
        self.ui.checkbox_bench_show_stats.stateChanged.connect(
            self._on_bench_show_stats_changed
        )
        self._on_bench_show_stats_changed()  # Apply initial state
        self.ui.option_tabwidget.currentChanged.connect(
            self._on_option_tab_changed
        )
        self._update_envelope_options()
        self._update_selection_visibility()
        self.update_bench()

    def _is_roi_valid(self, roi: dict, meta_size: tuple[int, int], params: dict) -> bool:
        """Sanity check: Does the stored ROI fit in the new image (roughly)?"""
        if not roi:
            return False

        h, w = meta_size

        # Apply transforms from params (defaults) to simulate preview dimensions
        if params.get("flip_xy"):
            h, w = w, h
        if params.get("rotate_90"):
            h, w = w, h # Rotate 90 swaps dims

        x, y = roi['pos']
        rw, rh = roi['size']

        # Check against new dimensions with 1px tolerance
        if x < -1 or y < -1:
            return False
        if x + rw > w + 1 or y + rh > h + 1:
            return False
        return True

    def _load_ascii(self, path: Path) -> None:
        """Load ASCII (.asc, .dat) via options dialog. Path from Open File/Folder or drop."""
        if path.exists():
            path = path.resolve()
        meta = get_ascii_metadata(path)
        if meta is None:
            log("Cannot read ASCII metadata", color="red")
            return

        # Sanity Check for ROI
        if "roi_state" in self._ascii_session_defaults:
            if not self._is_roi_valid(
                self._ascii_session_defaults["roi_state"],
                meta["size"],
                self._ascii_session_defaults
            ):
                self._ascii_session_defaults.pop("roi_state", None)
                self._ascii_session_defaults.pop("mask", None)
                self._ascii_session_defaults.pop("mask_rel", None)

        dlg = AsciiLoadOptionsDialog(
            path, meta, parent=self,
            initial_params=self._ascii_session_defaults,
        )
        if not dlg.exec():
            return
        user_params = dlg.get_params()
        params = {k: v for k, v in user_params.items()
                  if k not in ("mask_rel", "roi_state", "flip_xy", "rotate_90")}

        # Update session defaults
        self._ascii_session_defaults = {
            "size_ratio": user_params["size_ratio"],
            "convert_to_8_bit": user_params["convert_to_8_bit"],
            "delimiter": user_params["delimiter"],
            "first_col_is_row_number": user_params["first_col_is_row_number"],
            "flip_xy": user_params.get("flip_xy", False),
            "rotate_90": user_params.get("rotate_90", False),
        }
        if "subset_ratio" in user_params:
            self._ascii_session_defaults["subset_ratio"] = user_params["subset_ratio"]
        if "mask_rel" in user_params:
            self._ascii_session_defaults["mask_rel"] = user_params["mask_rel"]
        else:
            self._ascii_session_defaults.pop("mask_rel", None)
        if "roi_state" in user_params:
            self._ascii_session_defaults["roi_state"] = user_params["roi_state"]
        else:
            self._ascii_session_defaults.pop("roi_state", None)

        with LoadingManager(self, f"Loading {path}", blocking_label=self.ui.blocking_status) as lm:
            img = load_ascii(
                path,
                progress_callback=lm.set_progress,
                message_callback=lm.set_message,
                **params,
            )
        log(f"Loaded in {lm.duration:.2f}s")
        self.ui.image_viewer.set_image(img)

        # Apply transforms and set ROI
        if user_params.get("flip_xy"):
            self.ui.image_viewer.manipulate("transpose")
        if user_params.get("rotate_90"):
            self.ui.image_viewer.manipulate("rotate_90")

        if "roi_state" in user_params:
            s = user_params["roi_state"]
            # Need to update viewer's ROI to match user selection
            # Note: image_viewer uses square_roi (ROI) and poly_roi
            # Default is square_roi
            # roi_state is from RectROI (Dialog).
            # Viewer's ROI might be PolyLineROI if that was selected last time?
            # But normally we want the rect.
            # We update square_roi.
            self.ui.image_viewer.square_roi.setPos(s['pos'])
            self.ui.image_viewer.square_roi.setSize(s['size'])
            self.ui.image_viewer.square_roi.setAngle(s['angle'])
            # Ensure square_roi is active?
            # If poly was active, maybe switch? Or just set square.
            # If user selected Rect in dialog, we should probably switch to Rect in Viewer.
            if self.ui.image_viewer.roi is not self.ui.image_viewer.square_roi:
                self.ui.image_viewer.change_roi()

        self.last_file_dir = path.parent
        self.last_file = path.name
        self.update_statusbar()
        self.reset_options()

    def browse_tof(self) -> None:
        path, _ = QFileDialog.getOpenFileName()
        if not path:
            return
        with LoadingManager(self, "Loading TOF data...", blocking_label=self.ui.blocking_status, blocking_delay_ms=0):
            self.tof_adapter.set_data(path, self.ui.image_viewer.data.meta)
        self.ui.checkbox_tof.setEnabled(True)
        self.ui.checkbox_tof.setChecked(True)

    def browse_file(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            directory=str(self.last_file_dir),
        )
        if file_path:
            self.load(Path(file_path))

    def browse_folder(self) -> None:
        folder_path = QFileDialog.getExistingDirectory(
            directory=str(self.last_file_dir),
        )
        if folder_path:
            self.load(Path(folder_path))

    def export(self) -> None:
        with LoadingManager(self, "Exporting...", blocking_label=self.ui.blocking_status, blocking_delay_ms=0):
            self.ui.image_viewer.exportClicked()

    def browse_lut(self) -> None:
        file, _ = QFileDialog.getOpenFileName(
            caption="Choose LUT File",
            directory=str(self.last_file_dir),
            filter="JSON (*.json)",
        )
        if file:
            try:
                with open(file, "r", encoding="utf-8") as f:
                    lut_config = json.load(f)
                self.ui.image_viewer.load_lut_config(lut_config)
            except Exception:
                log("LUT could not be loaded. Make sure it is an "
                    "appropriately structured '.json' file.", color="red")

    def save_lut(self) -> None:
        path = QFileDialog.getExistingDirectory(
            caption="Choose LUT File Location",
            directory=str(self.last_file_dir),
        )
        if path:
            lut_config = self.ui.image_viewer.get_lut_config()
            file = Path(path) / "lut_config.json"
            with open(file, "w", encoding="utf-8") as f:
                lut_config = json.dump(
                    lut_config,
                    f,
                    ensure_ascii=False,
                    indent=4,
                )

    def _on_lut_spin_changed(self) -> None:
        """Apply LUT min/max from spinners to histogram."""
        mn = self.ui.spin_lut_min.value()
        mx = self.ui.spin_lut_max.value()
        if mn > mx:
            mn, mx = mx, mn
        self.ui.image_viewer.ui.histogram.setLevels(min=mn, max=mx)

    def _on_lut_log_changed(self) -> None:
        """Toggle logarithmic scale on histogram counts via data transform (no setLogMode)."""
        self._apply_lut_log_state()

    def _apply_lut_log_state(self) -> None:
        """Apply log/linear histogram display. Uses data transform instead of setLogMode to keep
        Y-axis anchored at bottom (avoids pyqtgraph ViewBox log-mode jump)."""
        log_on = self.ui.checkbox_lut_log.isChecked()
        hist = self.ui.image_viewer.ui.histogram
        vb = hist.vb
        plot = hist.item.plot
        img_item = self.ui.image_viewer.getImageItem()
        if img_item is None:
            vb.setLogMode("y", False)
            return
        h = img_item.getHistogram()
        if h[0] is None:
            vb.setLogMode("y", False)
            return
        xdata, ydata = np.asarray(h[0]), np.asarray(h[1], dtype=np.float64)
        if len(ydata) == 0:
            vb.setLogMode("y", False)
            return
        with np.errstate(invalid="ignore"):
            ymax_raw = float(np.nanmax(ydata))
        if not np.isfinite(ymax_raw) or ymax_raw <= 0:
            ymax_raw = 1.0
        vb.enableAutoRange(vb.YAxis, False)
        vb.setLogMode("y", False)
        if log_on:
            y_transformed = np.log10(ydata + 1.0)
            ymax_log = float(np.log10(ymax_raw + 1.0))
            plot.setData(xdata, y_transformed)
            vb.setYRange(0.0, ymax_log, padding=0)
        else:
            plot.setData(xdata, ydata)
            vb.setYRange(0.0, ymax_raw, padding=0)
        vb.updateViewRange()

    def _sync_lut_spinners(self) -> None:
        """Sync spinner values from imageItem (authoritative) or histogram."""
        img_item = self.ui.image_viewer.getImageItem()
        levels = img_item.getLevels()
        if levels is None:
            levels = self.ui.image_viewer.ui.histogram.getLevels()
        if levels is None or (hasattr(levels, "__len__") and len(levels) != 2):
            return
        arr = np.asarray(levels)
        if arr.ndim != 1 or arr.size != 2:
            return
        mn_f, mx_f = float(arr[0]), float(arr[1])

        # Skip if values already match (avoid redundant updates)
        if (abs(self.ui.spin_lut_min.value() - mn_f) < 1e-12
                and abs(self.ui.spin_lut_max.value() - mx_f) < 1e-12):
            return

        decimals = 2

        for s in (self.ui.spin_lut_min, self.ui.spin_lut_max):
            s.blockSignals(True)
        self.ui.spin_lut_min.setDecimals(decimals)
        self.ui.spin_lut_max.setDecimals(decimals)
        self.ui.spin_lut_min.setValue(mn_f)
        self.ui.spin_lut_max.setValue(mx_f)
        for s in (self.ui.spin_lut_min, self.ui.spin_lut_max):
            s.blockSignals(False)

    def start_web_connection(self) -> None:
        address = self.ui.address_edit.text()
        token = self.ui.token_edit.text()
        if not address or not token:
            return

        self._web_connection = WebDataLoader(address, token)
        self._web_connection.image_received.connect(self.end_web_connection)
        self._web_connection.start()
        self.ui.button_connect.setEnabled(False)
        self.ui.button_connect.setText("Listening...")
        self.ui.button_disconnect.setEnabled(True)
        self.ui.address_edit.setEnabled(False)
        self.ui.token_edit.setEnabled(False)

    def end_web_connection(self, img: ImageData | None) -> None:
        if img is not None:
            self.ui.image_viewer.set_image(img)
            self.reset_options()
        else:
            self._web_connection.stop()
            self.ui.address_edit.setEnabled(True)
            self.ui.token_edit.setEnabled(True)
            self.ui.button_connect.setEnabled(True)
            self.ui.button_connect.setText("Connect")
            self.ui.button_disconnect.setEnabled(False)
            self._web_connection.deleteLater()

    def show_simulated_live(self) -> None:
        """Open Simulated Live: generates Lissajous/Lightning viz, streams to viewer."""
        from PyQt6.QtCore import Qt
        self._simulated_first_frame = True
        if self._simulated_live is None:
            self._simulated_live = SimulatedLiveWidget(self)
            self._simulated_live.setWindowFlags(
                Qt.WindowType.Tool | Qt.WindowType.WindowStaysOnTopHint
            )
            def _simulated_frame_cb(img):
                self.ui.image_viewer.set_image(img, live_update=True)
                self.ui.image_viewer.setCurrentIndex(img.n_images - 1)
                if self._simulated_first_frame:
                    self._simulated_first_frame = False
                    self.reset_options()
            self._simulated_live.set_frame_callback(_simulated_frame_cb)
            self._simulated_live.setWindowIcon(self.windowIcon())
        self._simulated_live.show()
        self._simulated_live.raise_()
        self._simulated_live.activateWindow()

    def show_real_camera_dialog(self) -> None:
        """Open Webcam dialog: sliders, Start/Stop, streams to viewer."""
        if self._real_camera_dialog is None:
            self._camera_pending: ImageData | None = None
            self._camera_has_applied = False
            self._camera_apply_timer = QTimer(self)
            self._camera_apply_timer.setSingleShot(True)
            _CAMERA_APPLY_MS = 100  # Fixed 10 FPS display; FPS from dialog ignored for now

            def _apply_camera_frame() -> None:
                if self._camera_pending is not None:
                    img = self._camera_pending
                    self._camera_pending = None
                    self.ui.image_viewer.set_image(img, live_update=True)
                    self.ui.image_viewer.setCurrentIndex(img.n_images - 1)
                    if not self._camera_has_applied:
                        self._camera_has_applied = True
                        self.reset_options()
                if self._camera_pending is not None:
                    self._camera_apply_timer.start(_CAMERA_APPLY_MS)

            def _on_camera_frame(img: ImageData) -> None:
                self._camera_pending = img
                if not self._camera_apply_timer.isActive():
                    self._camera_apply_timer.start(1)

            self._camera_apply_timer.timeout.connect(_apply_camera_frame)

            def _camera_stop_cleanup() -> None:
                self._camera_apply_timer.stop()
                # Apply final buffer if pending (worker sends full buffer on stop)
                if self._camera_pending is not None:
                    img = self._camera_pending
                    self._camera_pending = None
                    self.ui.image_viewer.set_image(img, live_update=False)
                    self.ui.image_viewer.setCurrentIndex(max(0, img.n_images - 1))
                    self.reset_options()

            self._real_camera_dialog = RealCameraDialog(
                self,
                on_frame=_on_camera_frame,
                on_stop=_camera_stop_cleanup,
            )
            self._real_camera_dialog.setWindowFlags(
                Qt.WindowType.Tool | Qt.WindowType.WindowStaysOnTopHint
            )
            self._real_camera_dialog.setWindowIcon(self.windowIcon())
            self._real_camera_dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
            self._real_camera_dialog.destroyed.connect(
                lambda: setattr(self, "_real_camera_dialog", None)
            )
        self._real_camera_dialog.show()
        self._real_camera_dialog.raise_()
        self._real_camera_dialog.activateWindow()

    def load(self, path: Optional[Path | str] = None) -> None:
        if path is None:
            return self.ui.image_viewer.load_data()

        if isinstance(path, str):
            path = path.strip()
            if path.startswith("file://"):
                path = QUrl(path).toLocalFile()
            path = Path(path)
        # Sofortiges Feedback bei Drop/Open (Scan von Ordnern kann laenger dauern)
        lbl = self.ui.blocking_status
        lbl.setText("SCAN")
        lbl.setStyleSheet(get_style("scan"))
        QApplication.processEvents()

        try:
            if path.suffix == ".blitz":
                self.load_project(path)
            elif (
                path.suffix.lower() in (".asc", ".dat")
                or (path.is_dir() and any(
                    f.suffix.lower() in (".asc", ".dat")
                    for f in path.iterdir() if f.is_file()
                ))
            ):
                self._load_ascii(path)
            else:
                self.load_images(path)
        finally:
            # IDLE falls LoadingManager nicht aktiv (z.B. Dialog abgebrochen)
            if lbl.text() == "SCAN":
                lbl.setText("IDLE")
                lbl.setStyleSheet(get_style("idle"))

    def load_project(self, path: Path) -> None:
        log(f"Loading '{path.name}' configuration file...",
            color="green")
        self.last_file_dir = path.parent
        self.last_file = path.name
        settings.create_project(path)
        saved_path = Path(settings.get_project("path"))
        if saved_path.exists():
            mask = settings.get_project("mask")[1:]
            crop = settings.get_project("cropped")
            mask = mask if mask else None
            crop = crop if crop else None
            self.sync_project_preloading()
            with LoadingManager(self, f"Loading {saved_path}", blocking_label=self.ui.blocking_status) as lm:
                self.ui.image_viewer.load_data(
                    saved_path,
                    progress_callback=lm.set_progress,
                    message_callback=lm.set_message,
                    size_ratio=self.ui.spinbox_load_size.value(),
                    subset_ratio=self.ui.spinbox_load_subset.value(),
                    max_ram=self.ui.spinbox_max_ram.value(),
                    convert_to_8_bit=
                        self.ui.checkbox_load_8bit.isChecked(),
                    grayscale=self.ui.checkbox_load_grayscale.isChecked(),
                    mask=mask,
                    crop=crop,
                )
            log(f"Loaded in {lm.duration:.2f}s")
            self.last_file_dir = saved_path.parent
            self.last_file = saved_path.name
            self.update_statusbar()
            self.reset_options()
            self.sync_project_postloading()
        else:
            log("Path to dataset in .blitz project file does not point to "
                "a valid file or folder location. Deleting entry...",
                color="red")
            settings.set("path", "")

    def load_images(self, path: Path) -> None:
        # Bei 8-bit / Grayscale Quelle: Checkboxen direkt setzen
        if (
            DataLoader._is_video(path)
            or (path.is_file() and DataLoader._is_image(path))
            or (path.is_dir() and get_image_metadata(path) is not None)
        ):
            is_gray, is_uint8 = get_sample_format(path)
            if is_uint8:
                self.ui.checkbox_load_8bit.setChecked(True)
            if is_gray:
                self.ui.checkbox_load_grayscale.setChecked(True)

        params = {
            "size_ratio": self.ui.spinbox_load_size.value(),
            "subset_ratio": self.ui.spinbox_load_subset.value(),
            "max_ram": self.ui.spinbox_max_ram.value(),
            "convert_to_8_bit": self.ui.checkbox_load_8bit.isChecked(),
            "grayscale": self.ui.checkbox_load_grayscale.isChecked(),
        }

        if DataLoader._is_video(path):
            try:
                meta = DataLoader.get_video_metadata(path)

                # Sanity Check for ROI
                if "roi_state" in self._video_session_defaults:
                    if not self._is_roi_valid(
                        self._video_session_defaults["roi_state"],
                        meta.size,
                        self._video_session_defaults
                    ):
                        self._video_session_defaults.pop("roi_state", None)
                        self._video_session_defaults.pop("mask", None)
                        self._video_session_defaults.pop("mask_rel", None)

                show_dialog = self.ui.checkbox_video_dialog_always.isChecked()
                if show_dialog:
                    dlg = VideoLoadOptionsDialog(
                        path, meta, parent=self,
                        initial_params=self._video_session_defaults,
                    )
                    if dlg.exec():
                        user_params = dlg.get_params()
                        params.update(user_params)
                        self._video_session_defaults = {
                            "size_ratio": user_params["size_ratio"],
                            "step": user_params["step"],
                            "grayscale": user_params["grayscale"],
                            "convert_to_8_bit": user_params.get("convert_to_8_bit", False),
                            "flip_xy": user_params.get("flip_xy", False),
                            "rotate_90": user_params.get("rotate_90", False),
                        }
                        if "mask_rel" in user_params:
                            self._video_session_defaults["mask_rel"] = user_params["mask_rel"]
                        else:
                            self._video_session_defaults.pop("mask_rel", None)
                        if "roi_state" in user_params:
                            self._video_session_defaults["roi_state"] = user_params["roi_state"]
                        else:
                            self._video_session_defaults.pop("roi_state", None)

                        self.ui.spinbox_load_size.setValue(
                            user_params["size_ratio"],
                        )
                        self.ui.spinbox_load_subset.setValue(
                            1.0 / user_params["step"],
                        )
                        self.ui.checkbox_load_grayscale.setChecked(
                            user_params["grayscale"],
                        )
                        self.ui.checkbox_load_8bit.setChecked(
                            user_params.get("convert_to_8_bit", False),
                        )
                    else:
                        return
                else:
                    # Dialog nicht gezeigt: Session-Defaults anwenden
                    if self._video_session_defaults:
                        params["size_ratio"] = self._video_session_defaults.get(
                            "size_ratio", params["size_ratio"]
                        )
                        params["step"] = self._video_session_defaults.get(
                            "step", int(1.0 / params["subset_ratio"])
                        )
                        params["subset_ratio"] = 1.0 / params["step"]
                        params["grayscale"] = self._video_session_defaults.get(
                            "grayscale", params["grayscale"]
                        )
                        params["convert_to_8_bit"] = self._video_session_defaults.get(
                            "convert_to_8_bit", params["convert_to_8_bit"]
                        )
                        self.ui.spinbox_load_size.setValue(params["size_ratio"])
                        self.ui.spinbox_load_subset.setValue(params["subset_ratio"])
                        self.ui.checkbox_load_grayscale.setChecked(params["grayscale"])
                        self.ui.checkbox_load_8bit.setChecked(params["convert_to_8_bit"])
                        if "mask_rel" in self._video_session_defaults:
                            r = self._video_session_defaults["mask_rel"]
                            sr = params["size_ratio"]
                            h = int(meta.size[0] * sr)
                            w = int(meta.size[1] * sr)
                            x0 = max(0, int(r[0] * w))
                            y0 = max(0, int(r[1] * h))
                            x1 = min(w, int(r[2] * w))
                            y1 = min(h, int(r[3] * h))
                            if x1 > x0 and y1 > y0 and (x1 - x0 < w or y1 - y0 < h):
                                params["mask"] = (slice(x0, x1), slice(y0, y1))
                        # Inherit flags
                        params["flip_xy"] = self._video_session_defaults.get("flip_xy", False)
                        params["rotate_90"] = self._video_session_defaults.get("rotate_90", False)
                        if "roi_state" in self._video_session_defaults:
                            params["roi_state"] = self._video_session_defaults["roi_state"]
            except Exception as e:
                log(f"Error reading video metadata: {e}", color="red")

        elif (
            (path.is_file() and DataLoader._is_image(path))
            or (path.is_dir() and get_image_metadata(path) is not None)
        ):
            try:
                meta = get_image_metadata(path)
                if meta is not None:
                    # Sanity Check for ROI
                    if "roi_state" in self._image_session_defaults:
                        # meta["size"] is (h, w)
                        if not self._is_roi_valid(
                            self._image_session_defaults["roi_state"],
                            meta["size"],
                            self._image_session_defaults
                        ):
                            self._image_session_defaults.pop("roi_state", None)
                            self._image_session_defaults.pop("mask", None)
                            self._image_session_defaults.pop("mask_rel", None)

                    show_dialog = self.ui.checkbox_video_dialog_always.isChecked()
                    if show_dialog:
                        dlg = ImageLoadOptionsDialog(
                            path, meta, parent=self,
                            initial_params=self._image_session_defaults,
                        )
                        if dlg.exec():
                            user_params = dlg.get_params()
                            params.update(user_params)
                            self._image_session_defaults = {
                                "size_ratio": user_params["size_ratio"],
                                "grayscale": user_params["grayscale"],
                                "convert_to_8_bit": user_params.get("convert_to_8_bit", False),
                                "flip_xy": user_params.get("flip_xy", False),
                                "rotate_90": user_params.get("rotate_90", False),
                            }
                            if "subset_ratio" in user_params:
                                self._image_session_defaults["subset_ratio"] = user_params["subset_ratio"]
                            if "mask_rel" in user_params:
                                self._image_session_defaults["mask_rel"] = user_params["mask_rel"]
                            else:
                                self._image_session_defaults.pop("mask_rel", None)
                            if "roi_state" in user_params:
                                self._image_session_defaults["roi_state"] = user_params["roi_state"]
                            else:
                                self._image_session_defaults.pop("roi_state", None)

                            self.ui.spinbox_load_size.setValue(
                                user_params["size_ratio"],
                            )
                            self.ui.spinbox_load_subset.setValue(
                                user_params.get("subset_ratio", params["subset_ratio"]),
                            )
                            self.ui.checkbox_load_grayscale.setChecked(
                                user_params["grayscale"],
                            )
                            self.ui.checkbox_load_8bit.setChecked(
                                user_params.get("convert_to_8_bit", False),
                            )
                        else:
                            return
                    else:
                        if self._image_session_defaults:
                            params["size_ratio"] = self._image_session_defaults.get(
                                "size_ratio", params["size_ratio"]
                            )
                            params["grayscale"] = self._image_session_defaults.get(
                                "grayscale", params["grayscale"]
                            )
                            params["subset_ratio"] = self._image_session_defaults.get(
                                "subset_ratio", params["subset_ratio"]
                            )
                            params["convert_to_8_bit"] = self._image_session_defaults.get(
                                "convert_to_8_bit", params["convert_to_8_bit"]
                            )
                            self.ui.spinbox_load_size.setValue(params["size_ratio"])
                            self.ui.spinbox_load_subset.setValue(params["subset_ratio"])
                            self.ui.checkbox_load_grayscale.setChecked(params["grayscale"])
                            self.ui.checkbox_load_8bit.setChecked(params["convert_to_8_bit"])
                            if "mask_rel" in self._image_session_defaults:
                                r = self._image_session_defaults["mask_rel"]
                                sr = params["size_ratio"]
                                h_m, w_m = meta["size"]
                                h = int(h_m * sr)
                                w = int(w_m * sr)
                                x0 = max(0, int(r[0] * w))
                                y0 = max(0, int(r[1] * h))
                                x1 = min(w, int(r[2] * w))
                                y1 = min(h, int(r[3] * h))
                                if x1 > x0 and y1 > y0 and (x1 - x0 < w or y1 - y0 < h):
                                    params["mask"] = (slice(x0, x1), slice(y0, y1))
                            params["flip_xy"] = self._image_session_defaults.get("flip_xy", False)
                            params["rotate_90"] = self._image_session_defaults.get("rotate_90", False)
                            if "roi_state" in self._image_session_defaults:
                                params["roi_state"] = self._image_session_defaults["roi_state"]
            except Exception as e:
                log(f"Error reading image metadata: {e}", color="red")

        params.pop("mask_rel", None)
        params.pop("roi_state", None) # Don't pass to DataLoader
        flip_xy = params.pop("flip_xy", False)
        rotate_90 = params.pop("rotate_90", False)

        # We need roi_state later for Viewer, so retrieve it from defaults or user_params if available?
        # params dictionary passed to load_data shouldn't have extra keys if DataLoader doesn't support them.
        # But we need to use them after load.
        # I removed roi_state from params passed to load_data.
        # But I need to access it.
        # I can use _image_session_defaults or _video_session_defaults.
        # But that's ugly if we just loaded defaults.
        # Better: store target_roi_state locally.

        target_roi_state = None
        # Retrieve from defaults if not in params (params.pop removed it)
        if DataLoader._is_video(path):
            target_roi_state = self._video_session_defaults.get("roi_state")
        elif (path.is_file() and DataLoader._is_image(path)) or path.is_dir():
            target_roi_state = self._image_session_defaults.get("roi_state")

        with LoadingManager(self, f"Loading {path}", blocking_label=self.ui.blocking_status) as lm:
            self.ui.image_viewer.load_data(
                path,
                progress_callback=lm.set_progress,
                message_callback=lm.set_message,
                **params,
            )
        log(f"Loaded in {lm.duration:.2f}s")

        # Apply transforms and set ROI
        if flip_xy:
            self.ui.image_viewer.manipulate("transpose")
        if rotate_90:
            self.ui.image_viewer.manipulate("rotate_90")

        if target_roi_state:
            s = target_roi_state
            self.ui.image_viewer.square_roi.setPos(s['pos'])
            self.ui.image_viewer.square_roi.setSize(s['size'])
            self.ui.image_viewer.square_roi.setAngle(s['angle'])
            if self.ui.image_viewer.roi is not self.ui.image_viewer.square_roi:
                self.ui.image_viewer.change_roi()

        self.last_file_dir = path.parent
        self.last_file = path.name
        self.update_statusbar()
        self.reset_options()
