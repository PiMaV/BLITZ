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
from .isoline import IsolineAdapter
from .rosee import ROSEEAdapter
from .simulated_live import SimulatedLiveWidget
from .tof import TOFAdapter
from .pca import PCAAdapter
from .ui import UI_MainWindow

URL_GITHUB = QUrl("https://github.com/CodeSchmiedeHGW/BLITZ")
URL_INP = QUrl("https://www.inp-greifswald.de/")
URL_MESS = QUrl("https://mess.engineering")


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
        self.isoline_adapter = IsolineAdapter(self.ui.image_viewer)
        self._iso_throttle_timer = QTimer(self)
        self._iso_throttle_timer.setSingleShot(True)
        self._iso_throttle_timer.timeout.connect(self.update_isocurves)
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
        self.ui.action_link_mess.triggered.connect(
            lambda: QDesktopServices.openUrl(URL_MESS)  # type: ignore
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
        self.ui.button_reset_roi.clicked.connect(
            self.ui.image_viewer.reset_roi
        )
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
        self.ui.spinbox_crop_range_start.valueChanged.connect(
            self._on_selection_changed
        )
        self.ui.spinbox_crop_range_end.valueChanged.connect(
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
        self.ui.spinbox_ops_norm_window.valueChanged.connect(lambda: self.apply_ops())
        self.ui.spinbox_ops_norm_lag.valueChanged.connect(lambda: self.apply_ops())
        self.ui.checkbox_ops_sliding_apply_full.stateChanged.connect(lambda: self.apply_ops())
        for cb in (self.ui.combobox_ops_subtract_src, self.ui.combobox_ops_divide_src):
            cb.currentIndexChanged.connect(lambda: self.apply_ops())
        self.ui.combobox_ops_range_method.currentIndexChanged.connect(lambda: self.apply_ops())
        self.ui.slider_ops_subtract.valueChanged.connect(
            self._update_ops_slider_labels
        )
        self.ui.slider_ops_subtract.valueChanged.connect(lambda: self.apply_ops())
        self.ui.slider_ops_divide.valueChanged.connect(
            self._update_ops_slider_labels
        )
        self.ui.slider_ops_divide.valueChanged.connect(lambda: self.apply_ops())
        self.ui.button_ops_load_file.clicked.connect(self.load_ops_file)
        self.ui.spinbox_crop_range_start.editingFinished.connect(lambda: self.apply_ops())
        self.ui.spinbox_crop_range_end.editingFinished.connect(lambda: self.apply_ops())
        # crop range: _on_selection_changed calls apply_ops(skip_update=True) when aggregate
        # to avoid double mean computation (apply_aggregation handles refresh)
        # Reduce dropdown: only _on_reduce_changed (not apply_ops) to avoid double mean computation
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
        self.ui.spinbox_isocurves.valueChanged.connect(self.update_isocurves)
        self.ui.button_pca_calc.clicked.connect(self.pca_calculate)
        self.ui.button_pca_show.clicked.connect(self.pca_toggle_show)
        self.ui.checkbox_pca_exact.stateChanged.connect(self._pca_on_exact_changed)
        self.ui.spinbox_pcacomp.valueChanged.connect(self.pca_update_view)
        self.ui.combobox_pca.currentIndexChanged.connect(self._pca_on_mode_changed)
        self.ui.checkbox_pca_include_mean.stateChanged.connect(self.pca_update_view)
        self.pca_adapter.started.connect(self.pca_on_started)
        self.pca_adapter.finished.connect(self.pca_on_finished)
        self.pca_adapter.error.connect(self.pca_on_error)
        self.ui.spinbox_iso_smoothing.valueChanged.connect(
            self.update_isocurves
        )
        self.ui.image_viewer.timeLine.sigPositionChanged.connect(
            self._schedule_isocurves_update
        )
        self.ui.image_viewer.image_changed.connect(
            self._schedule_isocurves_update
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

    def _update_envelope_options(self) -> None:
        per_image = self.ui.checkbox_minmax_per_image.isChecked()
        per_crosshair = self.ui.checkbox_envelope_per_crosshair.isChecked()
        per_ds = self.ui.checkbox_envelope_per_dataset.isChecked()
        pct = float(self.ui.spinbox_envelope_pct.value())
        self.ui.h_plot.set_show_minmax_per_image(per_image)
        self.ui.h_plot.set_show_envelope_per_crosshair(per_crosshair)
        self.ui.h_plot.set_show_envelope_per_dataset(per_ds)
        self.ui.h_plot.set_envelope_percentile(pct)
        self.ui.v_plot.set_show_minmax_per_image(per_image)
        self.ui.v_plot.set_show_envelope_per_crosshair(per_crosshair)
        self.ui.v_plot.set_show_envelope_per_dataset(per_ds)
        self.ui.v_plot.set_envelope_percentile(pct)
        rosee_active = self.ui.checkbox_rosee_active.isChecked()
        if not (rosee_active and self.ui.checkbox_rosee_h.isChecked()):
            self.ui.h_plot.draw_line()
        if not (rosee_active and self.ui.checkbox_rosee_v.isChecked()):
            self.ui.v_plot.draw_line()

    def _on_timeline_clicked_to_frame(self, idx: int) -> None:
        """User clicked main timeline -> switch to frame (Reduce = None)."""
        was_agg = self._is_aggregate_view()
        if was_agg:
            self.ui.combobox_reduce.blockSignals(True)
            self.ui.combobox_reduce.setCurrentIndex(0)
            self.ui.combobox_reduce.blockSignals(False)
            self.update_view_mode()
        n = self.ui.image_viewer.image.shape[0]
        idx = max(0, min(idx, n - 1))
        self.ui.image_viewer.setCurrentIndex(idx)
        self.ui.image_viewer.timeLine.setPos((idx, 0))
        if not was_agg:
            self.update_statusbar()

    def _on_reduce_changed(self) -> None:
        """Reduce dropdown changed -> update view (frame vs aggregate)."""
        self.update_view_mode()

    def _on_timeline_options_changed(self) -> None:
        """Frame-Tab: Upper/lower band + Mean/Median fuer Timeline-Kurve."""
        mode = self.ui.combobox_timeline_aggregation.currentData()
        self.ui.image_viewer.set_timeline_options(
            mode,
            self.ui.checkbox_timeline_bands.isChecked(),
        )

    def _on_agg_update_on_drag_changed(self) -> None:
        """Update on drag: connect/disconnect crop_range.sigRegionChanged."""
        on_drag = self.ui.checkbox_agg_update_on_drag.isChecked()
        try:
            self.ui.roi_plot.crop_range.sigRegionChanged.disconnect(
                self._on_selection_changed_from_drag
            )
        except TypeError:
            pass
        if on_drag:
            self.ui.roi_plot.crop_range.sigRegionChanged.connect(
                self._on_selection_changed_from_drag
            )

    def _on_selection_changed_from_drag(self) -> None:
        """Live update during range drag. Throttled: nur bei geaenderten Bounds."""
        s = self.ui.spinbox_crop_range_start.value()
        e = self.ui.spinbox_crop_range_end.value()
        prev = getattr(self, "_last_agg_bounds_from_drag", None)
        if prev == (s, e):
            return
        self._last_agg_bounds_from_drag = (s, e)
        self._on_selection_changed()

    def _is_aggregate_view(self) -> bool:
        """True when Reduce is not None (Mean, Max, etc.)."""
        return self.ui.combobox_reduce.currentData() is not None

    def _update_selection_visibility(self) -> None:
        """Idx immer aktiv (wenn Daten). Range + aggregate band nur bei Multi-Frame."""
        data = getattr(self.ui.image_viewer, "data", None)
        n = data.n_images if data else 0
        needs_range = n > 1
        is_agg = self._is_aggregate_view()

        if n <= 1:
            self.ui.combobox_reduce.blockSignals(True)
            self.ui.combobox_reduce.setCurrentIndex(0)
            self.ui.combobox_reduce.blockSignals(False)
        self.ui.spinbox_current_frame.setEnabled(not is_agg and n > 0)
        if is_agg:
            self.ui.image_viewer.timeLine.hide()
        else:
            self.ui.image_viewer.timeLine.show()
        self.ui.range_section_widget.setEnabled(needs_range)
        self.ui.combobox_reduce.setEnabled(n > 1)
        if needs_range:
            self.ui.timeline_stack.agg_sep.show()
            self.ui.timeline_stack.agg_sep_spacer.show()
            self.ui.timeline_stack.agg_band.show()
            self.ui.roi_plot.crop_range.show()
            self._update_full_range_button_style()
        else:
            self.ui.timeline_stack.agg_sep.hide()
            self.ui.timeline_stack.agg_sep_spacer.hide()
            self.ui.timeline_stack.agg_band.hide()
            self.ui.roi_plot.crop_range.hide()

    def setup_sync(self) -> None:
        screen_geometry = QApplication.primaryScreen().availableGeometry()
        relative_size = settings.get("window/relative_size")
        width = int(screen_geometry.width() * relative_size)
        height = int(screen_geometry.height() * relative_size)
        self.setGeometry(
            (screen_geometry.width() - width) // 2,
            (screen_geometry.height() - height) // 2,
            width,
            height,
        )
        if relative_size == 1.0:
            self.showMaximized()

        if (docks_arrangement := settings.get("window/docks")):
            self.ui.dock_area.restoreState(docks_arrangement)

        settings.connect_sync(
            "default/load_8bit",
            self.ui.checkbox_load_8bit.stateChanged,
            self.ui.checkbox_load_8bit.isChecked,
            self.ui.checkbox_load_8bit.setChecked,
        )
        settings.connect_sync(
            "default/load_grayscale",
            self.ui.checkbox_load_grayscale.stateChanged,
            self.ui.checkbox_load_grayscale.isChecked,
            self.ui.checkbox_load_grayscale.setChecked,
        )
        settings.connect_sync(
            "default/max_ram",
            self.ui.spinbox_max_ram.editingFinished,
            self.ui.spinbox_max_ram.value,
            self.ui.spinbox_max_ram.setValue,
        )
        settings.connect_sync(
            "default/show_load_dialog",
            self.ui.checkbox_video_dialog_always.stateChanged,
            self.ui.checkbox_video_dialog_always.isChecked,
            self.ui.checkbox_video_dialog_always.setChecked,
        )

        # very dirty workaround of getting the name of the user-chosen
        # gradient from the LUT
        def loadPreset(name: str):
            self.ui.checkbox_auto_colormap.setChecked(False)
            self.ui.image_viewer.ui.histogram.gradient.lastCM = name
            self.ui.image_viewer.ui.histogram.gradient.restoreState(
                Gradients[name]  # type: ignore
            )
        def lastColorMap():
            return self.ui.image_viewer.ui.histogram.gradient.lastCM
        self.ui.image_viewer.ui.histogram.gradient.lastColorMap = lastColorMap
        self.ui.image_viewer.ui.histogram.gradient.loadPreset = loadPreset
        self.ui.image_viewer.ui.histogram.gradient.lastCM = (
            settings.get("default/colormap")
        )
        if settings.get("default/colormap") not in ("greyclip", "plasma", "bipolar"):
            self.ui.checkbox_auto_colormap.setChecked(False)

        settings.connect_sync(
            "default/colormap",
            self.ui.image_viewer.ui.histogram
                .gradient.sigGradientChangeFinished,
            self.ui.image_viewer.ui.histogram.gradient.lastColorMap,
            lambda name:
                self.ui.image_viewer.ui.histogram.gradient.restoreState(
                    Gradients[name]
            ),
        )
        settings.connect_sync(
            "web/address",
            self.ui.address_edit.editingFinished,
            self.ui.address_edit.text,
            self.ui.address_edit.setText,
        )
        settings.connect_sync(
            "web/token",
            self.ui.token_edit.editingFinished,
            self.ui.token_edit.text,
            self.ui.token_edit.setText,
        )
        # Timeline splitter: match Options width (run after geometry/dock restore)
        QTimer.singleShot(300, getattr(self.ui, "_set_timeline_splitter_sizes", lambda: None))

    def sync_project_preloading(self) -> None:
        settings.connect_sync_project(
            "size_ratio",
            self.ui.spinbox_load_size.editingFinished,
            self.ui.spinbox_load_size.value,
            self.ui.spinbox_load_size.setValue,
        )
        settings.connect_sync_project(
            "subset_ratio",
            self.ui.spinbox_load_subset.editingFinished,
            self.ui.spinbox_load_subset.value,
            self.ui.spinbox_load_subset.setValue,
        )

    def sync_project_postloading(self) -> None:
        settings.connect_sync_project(
            "flipped_x",
            self.ui.checkbox_flipx.stateChanged,
            self.ui.checkbox_flipx.isChecked,
            self.ui.checkbox_flipx.setChecked,
            lambda: self.ui.image_viewer.manipulate("flip_x"),
            True,
        )
        settings.connect_sync_project(
            "flipped_y",
            self.ui.checkbox_flipy.stateChanged,
            self.ui.checkbox_flipy.isChecked,
            self.ui.checkbox_flipy.setChecked,
            lambda: self.ui.image_viewer.manipulate("flip_y"),
            True,
        )
        settings.connect_sync_project(
            "rotated_90",
            self.ui.checkbox_rotate_90.stateChanged,
            self.ui.checkbox_rotate_90.isChecked,
            self.ui.checkbox_rotate_90.setChecked,
            lambda: self.ui.image_viewer.manipulate("rotate_90"),
            True,
        )
        settings.connect_sync_project(
            "measure_tool_pixels",
            self.ui.spinbox_pixel.editingFinished,
            self.ui.spinbox_pixel.value,
            self.ui.spinbox_pixel.setValue,
        )
        settings.connect_sync_project(
            "measure_tool_au",
            self.ui.spinbox_mm.editingFinished,
            self.ui.spinbox_mm.value,
            self.ui.spinbox_mm.setValue,
        )
        settings.connect_sync_project(
            "isocurve_smoothing",
            self.ui.spinbox_iso_smoothing.editingFinished,
            self.ui.spinbox_iso_smoothing.value,
            self.ui.spinbox_iso_smoothing.setValue,
        )
        settings.connect_sync_project(
            "mask",
            self.ui.image_viewer.image_mask_changed,
            self.ui.image_viewer.data.get_mask,
        )
        settings.connect_sync_project(
            "cropped",
            self.ui.image_viewer.image_crop_changed,
            self.ui.image_viewer.data.get_crop,
        )
        self.ui.image_viewer.image_crop_changed.connect(
            self._update_ops_crop_buttons
        )

    def _update_ops_crop_buttons(self) -> None:
        """Enable Undo Crop when crop is reversible (Keep in RAM)."""
        data = getattr(self.ui.image_viewer, "data", None)
        can_undo = data is not None and data.can_undo_crop()
        self.ui.button_ops_undo_crop.setEnabled(can_undo)

    def _update_ops_file_visibility(self) -> None:
        """Show Load button when subtract or divide uses File."""
        sub = self.ui.combobox_ops_subtract_src.currentData() == "file"
        div = self.ui.combobox_ops_divide_src.currentData() == "file"
        self.ui.ops_file_widget.setVisible(sub or div)

    def _update_ops_norm_visibility(self) -> None:
        """Show window/lag and Apply to full when Subtract or Divide uses Sliding mean."""
        sub = self.ui.combobox_ops_subtract_src.currentData() == "sliding_aggregate"
        div = self.ui.combobox_ops_divide_src.currentData() == "sliding_aggregate"
        visible = sub or div
        self.ui.ops_norm_widget.setVisible(visible)
        self.ui.checkbox_ops_sliding_apply_full.setVisible(visible)

    def _update_ops_slider_labels(self) -> None:
        self.ui.label_ops_subtract.setText(
            f"{self.ui.slider_ops_subtract.value()}%"
        )
        self.ui.label_ops_divide.setText(
            f"{self.ui.slider_ops_divide.value()}%"
        )

    def _on_crop_range_for_ops(self) -> None:
        """Crop range changed (drag). Pipeline/refresh via _on_selection_changed (same signal)."""
        s = self.ui.spinbox_crop_range_start.value()
        self.ui.image_viewer.setCurrentIndex(s)
        self.ui.image_viewer.timeLine.setPos((s, 0))

    def reset_options(self) -> None:
        # Signale blockieren waehrend Batch-Update (verhindert 21s emit-Kaskaden nach Load)
        _batch = [
            self.ui.roi_plot.crop_range,
            self.ui.spinbox_crop_range_start,
            self.ui.spinbox_crop_range_end,
            self.ui.spinbox_selection_window,
            self.ui.spinbox_current_frame,
            self.ui.combobox_reduce,
            self.ui.combobox_ops_subtract_src,
            self.ui.combobox_ops_divide_src,
            self.ui.combobox_ops_range_method,
            self.ui.slider_ops_subtract,
            self.ui.slider_ops_divide,
            self.ui.spinbox_ops_norm_window,
            self.ui.spinbox_ops_norm_lag,
            self.ui.checkbox_ops_sliding_apply_full,
        ]
        for w in _batch:
            w.blockSignals(True)
        try:
            self._reset_options_body()
        finally:
            for w in _batch:
                w.blockSignals(False)

    def _reset_options_body(self) -> None:
        self._aggregate_first_open = True
        self.pca_adapter.invalidate()
        self.ui.image_viewer.clear_reference_timeline_curve()
        self.ui.button_pca_show.setChecked(False)
        self.ui.button_pca_show.setText("View PCA")
        self.ui.button_pca_show.setEnabled(False)
        self.ui.spinbox_pcacomp.setEnabled(False)
        self.ui.spinbox_pcacomp.setVisible(True)
        self.ui.combobox_pca.setEnabled(False)
        self.ui.label_pca_time.setText("")
        self._pca_update_target_spinner_state()
        self.ui.pca_variance_plot.hide()
        self.ui.pca_variance_plot.clear()
        self.ui.table_pca_results.hide()
        self.ui.table_pca_results.setRowCount(0)
        self.ui.timeline_stack.label_timeline_mode.setText("Frame")
        self.ui.combobox_reduce.setCurrentIndex(0)
        self.ui.checkbox_flipx.setChecked(False)
        self.ui.checkbox_flipy.setChecked(False)
        self.ui.checkbox_rotate_90.setChecked(False)
        self.ui.combobox_ops_subtract_src.setCurrentIndex(0)
        self.ui.combobox_ops_divide_src.setCurrentIndex(0)
        self.ui.combobox_ops_range_method.setCurrentIndex(0)  # Mean
        self.ui.slider_ops_subtract.setValue(100)
        self.ui.slider_ops_divide.setValue(0)
        self.ui.spinbox_ops_norm_window.setValue(10)
        self.ui.spinbox_ops_norm_lag.setValue(0)
        self.ui.checkbox_ops_sliding_apply_full.setChecked(False)
        self.ui.button_ops_load_file.setText("Load reference image")
        self.ui.image_viewer._background_image = None
        self.ui.checkbox_measure_roi.setChecked(False)
        self.ui.spinbox_crop_range_start.setValue(0)
        self.ui.spinbox_crop_range_start.setMaximum(
            self.ui.image_viewer.data.n_images - 1
        )
        self.ui.spinbox_crop_range_end.setMaximum(
            self.ui.image_viewer.data.n_images - 1
        )
        self.ui.spinbox_crop_range_end.setValue(
            self.ui.image_viewer.data.n_images - 1
        )
        n_frames = max(1, self.ui.image_viewer.data.n_images)
        target_max = min(n_frames, 500)
        self.ui.spinbox_pcacomp_target.setMaximum(target_max)
        default_target = min(n_frames // 2, 50, target_max)
        self.ui.spinbox_pcacomp_target.setValue(default_target)
        self.ui.spinbox_current_frame.setMaximum(n_frames - 1)
        self.ui.spinbox_current_frame.setValue(
            min(self.ui.image_viewer.currentIndex, n_frames - 1)
        )
        self.ui.roi_plot.crop_range.setRegion(
            (0, self.ui.image_viewer.data.n_images - 1)
        )
        self._update_ops_file_visibility()
        self._update_ops_norm_visibility()
        self._update_ops_crop_buttons()
        self._update_ops_slider_labels()
        self.apply_ops()
        self.ui.spinbox_selection_window.setMaximum(
            self.ui.image_viewer.data.n_images
        )
        self._sync_selection_window_from_range()
        self._update_selection_visibility()
        self.ui.checkbox_roi_drop.setChecked(
            self.ui.image_viewer.is_roi_on_drop_update()
        )
        self.ui.spinbox_width_v.setRange(
            0, self.ui.image_viewer.data.shape[0] // 2
        )
        self.ui.spinbox_width_h.setRange(
            0, self.ui.image_viewer.data.shape[1] // 2
        )
        self.ui.spinbox_rosee_smoothing.setValue(0)
        self.ui.spinbox_rosee_smoothing.setMaximum(
            min(self.ui.image_viewer.data.shape)
        )
        self.ui.checkbox_rosee_active.setChecked(False)
        self.ui.checkbox_rosee_h.setEnabled(False)
        self.ui.checkbox_rosee_h.setChecked(True)
        self.ui.checkbox_rosee_v.setEnabled(False)
        self.ui.checkbox_rosee_v.setChecked(True)
        self.ui.checkbox_rosee_normalize.setEnabled(False)
        self.ui.spinbox_rosee_smoothing.setEnabled(False)
        self.ui.spinbox_isocurves.setValue(1)
        self.ui.checkbox_show_isocurve.setChecked(False)
        self.ui.checkbox_rosee_local_extrema.setEnabled(False)
        self.ui.checkbox_rosee_show_lines.setEnabled(False)
        self.ui.checkbox_rosee_show_indices.setEnabled(False)
        self.ui.checkbox_rosee_in_image_h.setEnabled(False)
        self.ui.checkbox_rosee_in_image_h.setChecked(False)
        self.ui.checkbox_rosee_in_image_v.setEnabled(False)
        self.ui.checkbox_rosee_in_image_v.setChecked(False)
        self.ui.label_rosee_plots.setEnabled(False)
        self.ui.label_rosee_image.setEnabled(False)
        self.update_view_mode()

    def update_crop_range_labels(self) -> None:
        """Region-Drag -> Snap auf Int, Spinboxen + Window aktualisieren.
        Win const. gilt nur bei Spinner/Aendern: Beim Handle-Ziehen immer Window
        aus neuer Spannweite berechnen (wie ohne Win const.)."""
        crop_range_ = self.ui.roi_plot.crop_range.getRegion()
        data = getattr(self.ui.image_viewer, "data", None)
        n = max(1, data.n_images) if data is not None else 1
        n_max = max(0, n - 1)

        left = max(0, min(int(round(crop_range_[0])), n_max)) if n > 0 else 0
        right = max(0, min(int(round(crop_range_[1])), n_max)) if n > 0 else 0
        if left > right:
            left, right = right, left
        w = max(1, right - left + 1)

        self.ui.roi_plot.crop_range.blockSignals(True)
        self.ui.roi_plot.crop_range.setRegion((left, right))
        self.ui.roi_plot.crop_range.blockSignals(False)
        self.ui.spinbox_crop_range_start.blockSignals(True)
        self.ui.spinbox_crop_range_start.setValue(left)
        self.ui.spinbox_crop_range_start.blockSignals(False)
        self.ui.spinbox_crop_range_end.blockSignals(True)
        self.ui.spinbox_crop_range_end.setValue(right)
        self.ui.spinbox_crop_range_end.blockSignals(False)
        self.ui.spinbox_selection_window.blockSignals(True)
        self.ui.spinbox_selection_window.setValue(w)
        self.ui.spinbox_selection_window.blockSignals(False)
        self._update_window_const_bounds()
        self._update_full_range_button_style()

    def _sync_current_frame_spinbox(self) -> None:
        """Timeline/Cursor -> Idx-Spinbox."""
        try:
            idx = int(round(self.ui.image_viewer.timeLine.pos()[0]))
        except (AttributeError, TypeError):
            return
        data = getattr(self.ui.image_viewer, "data", None)
        n = max(1, data.n_images) if data is not None else 1
        idx = max(0, min(idx, n - 1))
        self.ui.spinbox_current_frame.blockSignals(True)
        self.ui.spinbox_current_frame.setMaximum(max(0, n - 1))
        self.ui.spinbox_current_frame.setValue(idx)
        self.ui.spinbox_current_frame.blockSignals(False)

    def _on_current_frame_spinbox_changed(self) -> None:
        """Idx-Spinbox -> setCurrentIndex."""
        idx = self.ui.spinbox_current_frame.value()
        self.ui.image_viewer.setCurrentIndex(idx)

    def _sync_selection_from_window_const(self) -> None:
        """Beim Toggle von Win const.: Range anpassen."""
        if self.ui.checkbox_window_const.isChecked():
            self._sync_selection_range_from_window()
        self._update_window_const_bounds()

    def _update_window_const_bounds(self) -> None:
        """Bei Win const.: Start max = End - Window + 1, End min = Start + Window - 1."""
        data = getattr(self.ui.image_viewer, "data", None)
        n_max = max(0, data.n_images - 1) if data else 0

        if not self.ui.checkbox_window_const.isChecked():
            self.ui.spinbox_crop_range_start.setMaximum(n_max)
            self.ui.spinbox_crop_range_end.setMinimum(0)
            self.ui.spinbox_crop_range_end.setMaximum(n_max)
            return

        w = max(1, self.ui.spinbox_selection_window.value())
        # Start: 0 .. n_max - w + 1 (End folgt); End: w - 1 .. n_max (Start folgt)
        self.ui.spinbox_crop_range_start.setMaximum(max(0, n_max - w + 1))
        self.ui.spinbox_crop_range_end.setMinimum(max(0, w - 1))
        self.ui.spinbox_crop_range_end.setMaximum(n_max)

    def _sync_selection_window_from_range(self, from_start: bool = True) -> None:
        """Start <= End. Bei Win const.: anderes Bound bewegen, sonst Window anpassen."""
        s, e = (
            self.ui.spinbox_crop_range_start.value(),
            self.ui.spinbox_crop_range_end.value(),
        )
        w = max(1, self.ui.spinbox_selection_window.value())
        n = 0
        if hasattr(self.ui.image_viewer, "data") and self.ui.image_viewer.data:
            n = max(1, self.ui.image_viewer.data.n_images)

        if self.ui.checkbox_window_const.isChecked():
            if from_start:
                e = max(s, min(s + w - 1, n - 1)) if n > 0 else s + w - 1
                self.ui.spinbox_crop_range_end.blockSignals(True)
                self.ui.spinbox_crop_range_end.setValue(e)
                self.ui.spinbox_crop_range_end.blockSignals(False)
            else:
                s = max(0, min(e - w + 1, n - w)) if n > 0 else max(0, e - w + 1)
                e = s + w - 1
                self.ui.spinbox_crop_range_start.blockSignals(True)
                self.ui.spinbox_crop_range_start.setValue(s)
                self.ui.spinbox_crop_range_start.blockSignals(False)
                self.ui.spinbox_crop_range_end.blockSignals(True)
                self.ui.spinbox_crop_range_end.setValue(e)
                self.ui.spinbox_crop_range_end.blockSignals(False)

        else:
            if s > e:
                if from_start:
                    e = s
                    self.ui.spinbox_crop_range_end.blockSignals(True)
                    self.ui.spinbox_crop_range_end.setValue(e)
                    self.ui.spinbox_crop_range_end.blockSignals(False)
                else:
                    s = e
                    self.ui.spinbox_crop_range_start.blockSignals(True)
                    self.ui.spinbox_crop_range_start.setValue(s)
                    self.ui.spinbox_crop_range_start.blockSignals(False)
            w = max(1, e - s + 1)
            self.ui.spinbox_selection_window.blockSignals(True)
            self.ui.spinbox_selection_window.setValue(w)
            self.ui.spinbox_selection_window.blockSignals(False)

        s, e = self.ui.spinbox_crop_range_start.value(), self.ui.spinbox_crop_range_end.value()
        self.ui.roi_plot.crop_range.setRegion((s, e))
        self._update_window_const_bounds()

    def _sync_selection_range_from_window(self) -> None:
        """End = Start + Window - 1 (from Window). Window min 1, Start <= End."""
        s = self.ui.spinbox_crop_range_start.value()
        w = max(1, self.ui.spinbox_selection_window.value())
        if w != self.ui.spinbox_selection_window.value():
            self.ui.spinbox_selection_window.blockSignals(True)
            self.ui.spinbox_selection_window.setValue(w)
            self.ui.spinbox_selection_window.blockSignals(False)
        mx = self.ui.spinbox_crop_range_end.maximum()
        e = max(s, min(s + w - 1, mx))
        self.ui.spinbox_crop_range_end.blockSignals(True)
        self.ui.spinbox_crop_range_end.setValue(e)
        self.ui.spinbox_crop_range_end.blockSignals(False)
        self.ui.roi_plot.crop_range.setRegion((s, e))
        self._update_window_const_bounds()

    def toggle_hvplot_markings(self) -> None:
        self.ui.h_plot.toggle_mark_position()
        self.ui.h_plot.draw_line()
        self.ui.v_plot.toggle_mark_position()
        self.ui.v_plot.draw_line()

    def reset_selection_range(self, skip_apply: bool = False) -> None:
        """Set Selection auf volle Range [0, n-1].
        skip_apply: when True, do not call _on_selection_changed (caller handles apply).
        Used by update_view_mode to avoid double mean computation."""
        if not hasattr(self.ui.image_viewer, "data") or self.ui.image_viewer.data is None:
            return
        n = max(1, self.ui.image_viewer.data.n_images)
        self.ui.spinbox_crop_range_start.blockSignals(True)
        self.ui.spinbox_crop_range_end.blockSignals(True)
        self.ui.spinbox_selection_window.blockSignals(True)
        self.ui.spinbox_crop_range_start.setValue(0)
        self.ui.spinbox_crop_range_end.setValue(n - 1)
        self.ui.spinbox_selection_window.setValue(n)
        self.ui.spinbox_crop_range_start.blockSignals(False)
        self.ui.spinbox_crop_range_end.blockSignals(False)
        self.ui.spinbox_selection_window.blockSignals(False)
        self.ui.roi_plot.crop_range.setRegion((0, n - 1))
        self._update_full_range_button_style()
        if not skip_apply:
            self._on_selection_changed()

    def _update_full_range_button_style(self) -> None:
        """Full Range button: highlighted when range is [0, n-1], default otherwise."""
        data = getattr(self.ui.image_viewer, "data", None)
        n = max(1, data.n_images) if data else 1
        n_max = max(0, n - 1)
        s = self.ui.spinbox_crop_range_start.value()
        e = self.ui.spinbox_crop_range_end.value()
        is_full = s == 0 and e == n_max
        btn = self.ui.button_reset_range
        if is_full:
            btn.setStyleSheet(
                "QPushButton { background-color: #2d6a4f; color: white; "
                "font-weight: bold; }"
            )
            btn.setToolTip("Full Range (active)")
        else:
            btn.setStyleSheet("")
            btn.setToolTip("Reset selection to full range [0, " + str(n_max) + "]")

    def update_crop_range(self) -> None:
        self.ui.roi_plot.crop_range.setRegion(
            (self.ui.spinbox_crop_range_start.value(),
             self.ui.spinbox_crop_range_end.value())
        )

    def _crop_index_to_frame(self) -> None:
        """Show the edited crop index live in the image view."""
        s = self.sender()
        val = s.value() if s and hasattr(s, "value") else 0
        n = self.ui.image_viewer.image.shape[0]
        idx = max(0, min(int(val), n - 1))
        self.ui.image_viewer.setCurrentIndex(idx)

    def _crop_region_to_frame(self) -> None:
        """When crop region is dragged, show the moved edge's frame live."""
        reg = self.ui.roi_plot.crop_range.getRegion()
        vals = (int(round(reg[0])), int(round(reg[1])))
        prev = getattr(self, "_last_crop_region", (None, None))
        self._last_crop_region = vals
        left_changed = prev[0] is not None and vals[0] != prev[0]
        right_changed = prev[1] is not None and vals[1] != prev[1]
        if not left_changed and not right_changed:
            return  # programmatic setRegion (e.g. round); keep current frame
        if right_changed and not left_changed:
            idx = vals[1]
        else:
            idx = vals[0]
        n = self.ui.image_viewer.image.shape[0]
        idx = max(0, min(idx, n - 1))
        self.ui.image_viewer.setCurrentIndex(idx)

    def crop(self) -> None:
        with LoadingManager(self, "Cropping...", blocking_label=self.ui.blocking_status, blocking_delay_ms=0):
            self.ui.image_viewer.crop(
                left=self.ui.spinbox_crop_range_start.value(),
                right=self.ui.spinbox_crop_range_end.value(),
                keep=False,
            )
        self.reset_options()

    def _open_crop_dialog(self) -> None:
        """Open Crop Timeline dialog and apply crop with chosen mode (mask/destructive)."""
        data = getattr(self.ui.image_viewer, "data", None)
        if data is None or data.n_images <= 0:
            return
        s = self.ui.spinbox_crop_range_start.value()
        e = self.ui.spinbox_crop_range_end.value()
        n = data.n_images
        if s > e:
            s, e = e, s
        s = max(0, min(s, n - 1))
        e = max(0, min(e, n - 1))
        if s > e:
            return
        dlg = CropTimelineDialog(start=s, end=e, n_total=n, parent=self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        keep = dlg.get_keep()
        with LoadingManager(
            self, "Cropping...", blocking_label=self.ui.blocking_status, blocking_delay_ms=0
        ):
            self.ui.image_viewer.crop(left=s, right=e, keep=keep)
        self.reset_options()

    def _undo_crop(self) -> None:
        """Undo timeline crop (only when applied with Keep in RAM)."""
        success = self.ui.image_viewer.undo_crop()
        if success:
            self._sync_range_after_undo()

    def _sync_range_after_undo(self) -> None:
        """Update range UI after undo_crop restored full dataset."""
        data = getattr(self.ui.image_viewer, "data", None)
        if data is None:
            return
        n = max(1, data.n_images)
        n_max = max(0, n - 1)
        self.ui.spinbox_crop_range_start.blockSignals(True)
        self.ui.spinbox_crop_range_end.blockSignals(True)
        self.ui.spinbox_selection_window.blockSignals(True)
        self.ui.spinbox_crop_range_start.setMaximum(n_max)
        self.ui.spinbox_crop_range_end.setMaximum(n_max)
        self.ui.spinbox_crop_range_start.setValue(0)
        self.ui.spinbox_crop_range_end.setValue(n_max)
        self.ui.spinbox_selection_window.setMaximum(n)
        self.ui.spinbox_selection_window.setValue(n)
        self.ui.spinbox_crop_range_start.blockSignals(False)
        self.ui.spinbox_crop_range_end.blockSignals(False)
        self.ui.spinbox_selection_window.blockSignals(False)
        self.ui.roi_plot.crop_range.setRegion((0, n_max))
        self.ui.spinbox_current_frame.setMaximum(n_max)
        self._update_full_range_button_style()
        self._update_ops_crop_buttons()
        self.apply_ops()

    def apply_mask(self) -> None:
        with LoadingManager(self, "Masking...", blocking_label=self.ui.blocking_status, blocking_delay_ms=0):
            self.ui.image_viewer.apply_mask()

    def reset_mask(self) -> None:
        with LoadingManager(self, "Reset...", blocking_label=self.ui.blocking_status, blocking_delay_ms=0):
            self.ui.image_viewer.reset_mask()

    def change_roi(self) -> None:
        with LoadingManager(self, "Change ROI...", blocking_label=self.ui.blocking_status, blocking_delay_ms=0):
            self.ui.checkbox_roi_drop.setChecked(False)
            self.ui.image_viewer.change_roi()

    def image_mask(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            directory=str(self.last_file_dir),
        )
        with LoadingManager(self, "Masking...", blocking_label=self.ui.blocking_status, blocking_delay_ms=0):
            self.ui.image_viewer.image_mask(Path(file_path))

    def toggle_rosee(self) -> None:
        enabled = self.ui.checkbox_rosee_active.isChecked()
        if enabled:
            self.ui.checkbox_rosee_h.setEnabled(True)
            self.ui.checkbox_rosee_v.setEnabled(True)
            self.ui.checkbox_rosee_normalize.setEnabled(True)
            self.ui.spinbox_rosee_smoothing.setEnabled(True)
            self.ui.checkbox_rosee_local_extrema.setEnabled(True)
            self.ui.checkbox_rosee_show_lines.setEnabled(True)
            self.ui.checkbox_rosee_show_indices.setEnabled(True)
            self.ui.checkbox_rosee_in_image_h.setEnabled(True)
            self.ui.checkbox_rosee_in_image_v.setEnabled(True)
            self.ui.label_rosee_plots.setEnabled(True)
            self.ui.label_rosee_image.setEnabled(True)
        else:
            self.ui.checkbox_rosee_h.setEnabled(False)
            self.ui.checkbox_rosee_v.setEnabled(False)
            self.ui.checkbox_rosee_normalize.setEnabled(False)
            self.ui.spinbox_rosee_smoothing.setEnabled(False)
            self.ui.checkbox_rosee_local_extrema.setEnabled(False)
            self.ui.checkbox_rosee_show_lines.setEnabled(False)
            self.ui.checkbox_rosee_show_indices.setEnabled(False)
            self.ui.checkbox_rosee_in_image_h.setEnabled(False)
            self.ui.checkbox_rosee_in_image_v.setEnabled(False)
            self.ui.label_rosee_plots.setEnabled(False)
            self.ui.label_rosee_image.setEnabled(False)
        self.rosee_adapter.toggle(
            h_plot=self.ui.checkbox_rosee_h.isChecked() and enabled,
            v_plot=self.ui.checkbox_rosee_v.isChecked() and enabled,
            h_image=self.ui.checkbox_rosee_in_image_h.isChecked() and enabled,
            v_image=self.ui.checkbox_rosee_in_image_v.isChecked() and enabled,
        )
        self.rosee_adapter.update(
            use_local_extrema=self.ui.checkbox_rosee_local_extrema.isChecked(),
            smoothing=self.ui.spinbox_rosee_smoothing.value(),
            normalized=self.ui.checkbox_rosee_normalize.isChecked(),
            show_indices=self.ui.checkbox_rosee_show_indices.isChecked(),
            show_index_lines=self.ui.checkbox_rosee_show_lines.isChecked(),
        )

    def _schedule_isocurves_update(self) -> None:
        """Throttle isoline updates during timeline scrub / image changes."""
        self._iso_throttle_timer.stop()
        self._iso_throttle_timer.start(80)

    def update_isocurves(self) -> None:
        self.isoline_adapter.update(
            on=self.ui.checkbox_show_isocurve.isChecked(),
            n=self.ui.spinbox_isocurves.value(),
            smoothing=self.ui.spinbox_iso_smoothing.value(),
        )

    def load_ops_file(self) -> None:
        """Load or remove reference image for Ops (subtract/divide)."""
        if "Remove" not in self.ui.button_ops_load_file.text():
            file, _ = QFileDialog.getOpenFileName(
                caption="Choose Reference File",
                directory=str(self.last_file_dir),
            )
            if file and self.ui.image_viewer.load_background_file(Path(file)):
                self.ui.button_ops_load_file.setText("[Remove]")
                self.apply_ops()
        else:
            self.ui.image_viewer.unload_background_file()
            self.ui.button_ops_load_file.setText("Load reference image")
            self.apply_ops()

    def on_strgC(self) -> None:
        cb = QApplication.clipboard()
        cb.clear()
        cb.setText(self.ui.position_label.text())

    def update_statusbar_position(self, pos: tuple[int, int]) -> None:
        x, y, value = self.ui.image_viewer.get_position_info(pos)
        self.ui.position_label.setText(f"X: {x} | Y: {y} | Value: {value}")

    def _update_numba_dot(self) -> None:
        """Update numba dot in LUT panel: green = active, gray = off/unavailable."""
        data = getattr(self.ui.image_viewer, "data", None)
        if data is None:
            self.ui.numba_dot.setStyleSheet(
                "background-color: #565f89; border-radius: 5px;"
            )
            self.ui.numba_dot.setToolTip(
                "Green: Numba accelerating pipeline/aggregation. Gray: not in use."
            )
        elif not optimized.HAS_NUMBA:
            self.ui.numba_dot.setStyleSheet(
                "background-color: #565f89; border-radius: 5px;"
            )
            self.ui.numba_dot.setToolTip("Numba unavailable (not installed).")
        else:
            on = _numba_active(data)
            if on:
                self.ui.numba_dot.setStyleSheet(
                    "background-color: #9ece6a; border-radius: 5px;"
                )
                self.ui.numba_dot.setToolTip(
                    "Numba active: accelerating pipeline (subtract/divide) or "
                    "aggregation (Mean/Max/Min/Std)."
                )
            else:
                self.ui.numba_dot.setStyleSheet(
                    "background-color: #565f89; border-radius: 5px;"
                )
                self.ui.numba_dot.setToolTip(
                    "Numba not in use (no pipeline or aggregation, or using Median)."
                )

    def update_statusbar(self) -> None:
        frame, max_frame, name = self.ui.image_viewer.get_frame_info()
        self.ui.frame_label.setText(f"Frame: {frame} / {max_frame}")
        self.ui.file_label.setText(f"File: {name}")
        self.ui.ram_label.setText(
            f"Available RAM: {get_available_ram():.2f} GB"
        )
        x, y, value = self.ui.image_viewer.get_position_info()
        self.ui.position_label.setText(f"X: {x} | Y: {y} | Value: {value}")
        self._update_numba_dot()

    def _on_bench_show_stats_changed(self) -> None:
        """Show/hide CPU load in LUT panel. Timer runs for Bench tab or compact."""
        on = self.ui.checkbox_bench_show_stats.isChecked()
        settings.set("bench/show_stats", on)
        self.ui.bench_compact.setVisible(on)
        self._update_bench_timer()

    def _update_bench_timer(self) -> None:
        """Start timer when Bench tab visible or compact enabled; else stop."""
        bench_idx = getattr(self.ui, "bench_tab_index", self.ui.option_tabwidget.count() - 1)
        bench_tab_visible = self.ui.option_tabwidget.currentIndex() == bench_idx
        compact_enabled = self.ui.checkbox_bench_show_stats.isChecked()
        if bench_tab_visible or compact_enabled:
            self._bench_timer.start(500)
        else:
            self._bench_timer.stop()
            self.ui.label_bench_live.setText("")

    def _sync_pca_target_comp_to_data(self) -> None:
        """Update Target Comp max and default from current data (e.g. when PCA tab is shown)."""
        data = getattr(self.ui.image_viewer, "data", None)
        n_frames = max(1, data.n_images) if data else 1
        target_max = min(n_frames, 500)
        target = self.ui.spinbox_pcacomp_target
        target.setMaximum(target_max)
        current = target.value()
        if current < 1 or current > target_max or (current == 1 and target_max > 1):
            default_target = min(n_frames // 2, 50, target_max)
            default_target = max(1, default_target)
            target.setValue(default_target)
        self._pca_update_target_spinner_state()

    def _on_option_tab_changed(self, index: int) -> None:
        """Toggle LIVE indicator and timer when Bench tab visible. Sync PCA Target Comp when PCA tab shown.
        Force refresh Ops pipeline when Ops tab is selected."""
        bench_idx = getattr(
            self.ui, "bench_tab_index",
            self.ui.option_tabwidget.count() - 1,
        )
        if index == bench_idx:
            self._bench_live_tick = 0
            self.update_bench()
        else:
            self.ui.label_bench_live.setText("")
        self._update_bench_timer()
        pca_idx = getattr(self.ui, "pca_tab_index", -1)
        if index == pca_idx:
            self._sync_pca_target_comp_to_data()
        ops_idx = getattr(self.ui, "ops_tab_index", -1)
        if index == ops_idx:
            QTimer.singleShot(0, self.apply_ops)

    def _bench_tick(self) -> None:
        """Sample CPU/RAM/Disk, feed shared BenchData, refresh Bench tab + compact (if shown)."""
        ram_free = get_available_ram()
        ram_used = get_used_ram()
        cpu = get_cpu_percent()
        disk_r, disk_w = get_disk_io_mbs()
        self.ui.bench_data.add(cpu, ram_used, ram_free, disk_r, disk_w)
        self.ui.bench_sparklines.refresh_from_data()
        if self.ui.checkbox_bench_show_stats.isChecked():
            self.ui.bench_compact.refresh()
        bench_idx = getattr(self.ui, "bench_tab_index", self.ui.option_tabwidget.count() - 1)
        if self.ui.option_tabwidget.currentIndex() == bench_idx:
            tick = getattr(self, "_bench_live_tick", 0)
            self._bench_live_tick = tick + 1
            self.ui.label_bench_live.setText("\u25cb LIVE" if tick % 2 else "\u25cf LIVE")

    def update_bench(self) -> None:
        """Update Bench tab labels (matrix stats, cache, numba). CPU/RAM/Disk via _bench_tick."""
        data = getattr(self.ui.image_viewer, "data", None)
        if data is None:
            self.ui.label_bench_raw.setText("Raw: —")
            self.ui.label_bench_result.setText("Result: —")
            self.ui.label_bench_mode.setText("View mode: —")
            self.ui.label_bench_cache.setText("Cache: —")
            self.ui.label_bench_numba.setText("Numba: —")
            return
        shape = data._image.shape
        dtype = data._image.dtype.name
        self.ui.label_bench_raw.setText(
            f"Raw: {format_size_mb(data._image.nbytes)} "
            f"({shape[0]}x{shape[1]}x{shape[2]}x{shape[3]}, {dtype})"
        )
        result_cache = getattr(data, "_result_cache", {})
        is_aggregate = getattr(data, "_redop", None) is not None
        ops_in_cache = (
            sorted(set(k[0] for k in result_cache.keys()), key=lambda o: getattr(o, "name", str(o)))
            if result_cache
            else []
        )
        names = [getattr(o, "name", str(o)) for o in ops_in_cache]
        cache_str = f"Ready: {', '.join(names)}" if names else "—"
        self.ui.label_bench_cache.setText(f"Result cache: {cache_str}")

        if is_aggregate:
            op = data._redop
            bounds = getattr(data, "_agg_bounds", None)
            self.ui.label_bench_mode.setText(
                f"View mode: Range ({getattr(op, 'name', op)})"
            )
            key = (op, bounds)
            cached = result_cache.get(key) if result_cache else None
            if cached is not None:
                self.ui.label_bench_result.setText(
                    f"Result: {format_size_mb(cached.nbytes)} (cached)"
                )
            else:
                self.ui.label_bench_result.setText("Result: computing...")
        else:
            self.ui.label_bench_mode.setText("View mode: Frame")
            self.ui.label_bench_result.setText("Result: —")

        # Numba status: pipeline (subtract/divide) or aggregation (MEAN/MAX/MIN/STD)
        if optimized.HAS_NUMBA:
            on = _numba_active(data)
            self.ui.label_bench_numba.setText("Numba: on" if on else "Numba: off")
        else:
            self.ui.label_bench_numba.setText("Numba: unavailable")

    def _load_ascii(self, path: Path) -> None:
        """Load ASCII (.asc, .dat) via options dialog. Path from Open File/Folder or drop."""
        if path.exists():
            path = path.resolve()
        meta = get_ascii_metadata(path)
        if meta is None:
            log("Cannot read ASCII metadata", color="red")
            return
        ascii_data_size = tuple(meta["size"])
        ascii_defaults = self._ascii_session_defaults
        ascii_initial = ascii_defaults if (
            ascii_defaults and
            ascii_defaults.get("_data_size") == ascii_data_size
        ) else None
        dlg = AsciiLoadOptionsDialog(
            path, meta, parent=self,
            initial_params=ascii_initial,
        )
        if not dlg.exec():
            return
        user_params = dlg.get_params()
        skip_keys = {"mask_rel", "roi_state", "flip_xy"}
        params = {k: v for k, v in user_params.items() if k not in skip_keys}
        flip_xy = user_params.get("flip_xy", False)
        target_roi_state = user_params.get("roi_state")
        self._ascii_session_defaults = {
            "size_ratio": user_params["size_ratio"],
            "convert_to_8_bit": user_params["convert_to_8_bit"],
            "delimiter": user_params["delimiter"],
            "first_col_is_row_number": user_params["first_col_is_row_number"],
            "flip_xy": user_params.get("flip_xy", False),
            "_data_size": ascii_data_size,
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

        # Apply transforms from dialog
        if flip_xy:
            self.ui.image_viewer.manipulate("transpose")
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
            except Exception as e:
                log(f"LUT could not be loaded. Make sure it is an "
                    f"appropriately structured '.json' file. Error: {e}", color="red")

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
        if levels is None or len(levels) != 2:
            return
        mn_f, mx_f = map(float, levels)

        # Skip if values already match (avoid redundant updates)
        if (abs(self.ui.spin_lut_min.value() - mn_f) < 1e-12
                and abs(self.ui.spin_lut_max.value() - mx_f) < 1e-12):
            return

        spinners = (self.ui.spin_lut_min, self.ui.spin_lut_max)
        for s in spinners:
            s.blockSignals(True)
        for s, val in zip(spinners, (mn_f, mx_f)):
            s.setDecimals(2)
            s.setValue(val)
        for s in spinners:
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
                # Estimate RAM usage for full load at current settings
                # ImageData keeps uint8 by default (1 byte)

                show_dialog = self.ui.checkbox_video_dialog_always.isChecked()
                video_data_size = tuple(meta.size)
                video_defaults = self._video_session_defaults
                video_initial = video_defaults if (
                    video_defaults and
                    video_defaults.get("_data_size") == video_data_size
                ) else None
                if show_dialog:
                    dlg = VideoLoadOptionsDialog(
                        path, meta, parent=self,
                        initial_params=video_initial,
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
                            "_data_size": video_data_size,
                        }
                        if "roi_state" in user_params:
                            self._video_session_defaults["roi_state"] = user_params["roi_state"]
                        else:
                            self._video_session_defaults.pop("roi_state", None)
                        if "mask_rel" in user_params:
                            self._video_session_defaults["mask_rel"] = user_params["mask_rel"]
                        else:
                            self._video_session_defaults.pop("mask_rel", None)
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
                    # Dialog nicht gezeigt: Session-Defaults anwenden (nur wenn gleiche Datengroesse)
                    if video_initial:
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
                        params["flip_xy"] = self._video_session_defaults.get("flip_xy", False)
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
            except Exception as e:
                log(f"Error reading video metadata: {e}", color="red")

        elif (
            (path.is_file() and DataLoader._is_image(path))
            or (path.is_dir() and get_image_metadata(path) is not None)
        ):
            try:
                meta = get_image_metadata(path)
                if meta is not None:
                    show_dialog = self.ui.checkbox_video_dialog_always.isChecked()
                    image_data_size = tuple(meta["size"])
                    image_defaults = self._image_session_defaults
                    image_initial = image_defaults if (
                        image_defaults and
                        image_defaults.get("_data_size") == image_data_size
                    ) else None
                    if show_dialog:
                        dlg = ImageLoadOptionsDialog(
                            path, meta, parent=self,
                            initial_params=image_initial,
                        )
                        if dlg.exec():
                            user_params = dlg.get_params()
                            params.update(user_params)
                            self._image_session_defaults = {
                                "size_ratio": user_params["size_ratio"],
                                "grayscale": user_params["grayscale"],
                                "convert_to_8_bit": user_params.get("convert_to_8_bit", False),
                                "flip_xy": user_params.get("flip_xy", False),
                                "_data_size": image_data_size,
                            }
                            if "roi_state" in user_params:
                                self._image_session_defaults["roi_state"] = user_params["roi_state"]
                            else:
                                self._image_session_defaults.pop("roi_state", None)
                            if "subset_ratio" in user_params:
                                self._image_session_defaults["subset_ratio"] = user_params["subset_ratio"]
                            if "mask_rel" in user_params:
                                self._image_session_defaults["mask_rel"] = user_params["mask_rel"]
                            else:
                                self._image_session_defaults.pop("mask_rel", None)
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
                        if image_initial:
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
                            params["flip_xy"] = self._image_session_defaults.get("flip_xy", False)
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
            except Exception as e:
                log(f"Error reading image metadata: {e}", color="red")

        # Pop dialog-only params (DataLoader does not accept these)
        params.pop("mask_rel", None)
        target_roi_state = params.pop("roi_state", None)
        flip_xy = params.pop("flip_xy", False)
        if "step" in params:
            params["subset_ratio"] = 1.0 / params.pop("step")
        params.pop("frame_range", None)

        with LoadingManager(self, f"Loading {path}", blocking_label=self.ui.blocking_status) as lm:
            self.ui.image_viewer.load_data(
                path,
                progress_callback=lm.set_progress,
                message_callback=lm.set_message,
                **params,
            )
        log(f"Loaded in {lm.duration:.2f}s")

        # Apply transforms from load dialog (flip_xy)
        if flip_xy:
            self.ui.image_viewer.manipulate("transpose")
        # ROI (target_roi_state) determines mask at load time; no post-load crop.
        # Program ROI stays independent (init_roi default).
        self.last_file_dir = path.parent
        self.last_file = path.name
        self.update_statusbar()
        self.reset_options()

    def _is_sliding_mean_preview(self) -> bool:
        """True if Frame mode + pipeline has sliding mean (sub or div) with apply_full=False."""
        if self._is_aggregate_view():
            return False
        data = self.ui.image_viewer.data
        if data.is_single_image() or data.n_images <= 1:
            return False
        p = getattr(data, "_ops_pipeline", None) or {}
        for step in (p.get("subtract"), p.get("divide")):
            if (
                step
                and step.get("source") == "sliding_aggregate"
                and step.get("amount", 0) > 0
                and not step.get("apply_full", False)
            ):
                return True
        return False

    def _set_preview_frame_for_ops(self) -> None:
        """Set data.preview_frame when in sliding mean preview (single-frame processing)."""
        data = self.ui.image_viewer.data
        if self._is_sliding_mean_preview():
            try:
                idx = int(round(self.ui.image_viewer.timeLine.pos()[0]))
            except (AttributeError, TypeError):
                idx = 0
            idx = max(0, min(idx, data.n_images - 1))
            data.preview_frame = idx
        else:
            data.preview_frame = None

    def _on_timeline_for_sliding_preview(self) -> None:
        """Debounced: refresh image when scrubbing in sliding mean preview (current frame only)."""
        if not self._is_sliding_mean_preview():
            return
        t = getattr(self, "_sliding_preview_timer", None)
        if t is None:
            t = QTimer(self)
            t.setSingleShot(True)
            t.timeout.connect(self._refresh_sliding_preview)
            self._sliding_preview_timer = t
        t.start(50)

    def _refresh_sliding_preview(self) -> None:
        """Re-fetch image with preview_frame = current; pipeline unchanged."""
        if not self._is_sliding_mean_preview():
            return
        self._set_preview_frame_for_ops()
        self.ui.image_viewer.update_image(keep_timestep=True)

    def apply_ops(self, skip_update: bool = False) -> None:
        """Build Ops pipeline from UI and set on data.
        skip_update: when True, set pipeline but do not call update_image (caller does refresh).
        Used when in aggregate view to avoid double mean computation."""
        data = self.ui.image_viewer.data
        if data is None:
            return
        bounds = (
            self.ui.spinbox_crop_range_start.value(),
            self.ui.spinbox_crop_range_end.value(),
        )
        bg = self.ui.image_viewer._background_image

        def _step(src: str, amount: int) -> dict | None:
            if not src or src == "off" or amount <= 0:
                return None
            if src == "aggregate":
                op = range_method
                return {"source": "aggregate", "bounds": bounds, "method": op, "amount": amount / 100.0}
            if src == "file" and bg is not None:
                return {"source": "file", "reference": bg, "amount": amount / 100.0}
            if src == "sliding_aggregate":
                window = max(1, self.ui.spinbox_ops_norm_window.value())
                lag = max(0, self.ui.spinbox_ops_norm_lag.value())
                apply_full = self.ui.checkbox_ops_sliding_apply_full.isChecked()
                return {
                    "source": "sliding_aggregate",
                    "window": window,
                    "lag": lag,
                    "method": range_method,
                    "amount": amount / 100.0,
                    "apply_full": apply_full,
                }
            return None

        sub_src = self.ui.combobox_ops_subtract_src.currentData()
        sub_amt = self.ui.slider_ops_subtract.value()
        div_src = self.ui.combobox_ops_divide_src.currentData()
        div_amt = self.ui.slider_ops_divide.value()

        has_range_step = (
            (sub_src == "aggregate" and sub_amt > 0)
            or (div_src == "aggregate" and div_amt > 0)
        )
        has_sliding_step = (
            (sub_src == "sliding_aggregate" and sub_amt > 0)
            or (div_src == "sliding_aggregate" and div_amt > 0)
        )
        need_range_method = has_range_step or has_sliding_step
        was_hidden = not self.ui.ops_range_method_widget.isVisible()
        self.ui.ops_range_method_widget.setVisible(need_range_method)
        if need_range_method:
            QApplication.processEvents()
            if was_hidden:
                self.ui.combobox_ops_range_method.blockSignals(True)
                self.ui.combobox_ops_range_method.setCurrentIndex(0)
                self.ui.combobox_ops_range_method.blockSignals(False)
                QApplication.processEvents()
            # Prefer currentData(); fallback to index (avoids Qt currentData() returning None when just shown)
            range_method = self.ui.combobox_ops_range_method.currentData()
            if range_method is None:
                idx = self.ui.combobox_ops_range_method.currentIndex()
                ops_list = list(ReduceOperation)
                range_method = ops_list[idx] if 0 <= idx < len(ops_list) else ReduceOperation.MEAN
            # Defer second apply_ops when range widget was just shown, so UI state is fully stable
            if was_hidden and has_range_step:
                QTimer.singleShot(0, self.apply_ops)
        else:
            range_method = ReduceOperation.MEAN
        if has_range_step and self._is_aggregate_view():
            # Range subtraction in Aggregate view = mean minus mean (artifact).
            # Switch to Frame so user sees frame - mean instead.
            self.ui.combobox_reduce.blockSignals(True)
            self.ui.combobox_reduce.setCurrentIndex(0)
            self.ui.combobox_reduce.blockSignals(False)
            with LoadingManager(self, "Switching...", blocking_label=self.ui.blocking_status, blocking_delay_ms=0):
                self.ui.image_viewer.unravel()
            self._update_selection_visibility()
            self.update_statusbar()
            self.update_bench()

        pipeline: dict = {}
        if sub := _step(sub_src, sub_amt):
            pipeline["subtract"] = sub
        if div := _step(div_src, div_amt):
            pipeline["divide"] = div

        if pipeline:
            def _method_label(step: dict) -> str:
                m = step.get("method", "MEAN")
                name = m.name if hasattr(m, "name") else str(m)
                return name.capitalize()
            parts = []
            if "subtract" in pipeline:
                s = pipeline["subtract"]
                lbl = _method_label(s) if s.get("source") == "aggregate" else ""
                parts.append(f"Subtracting ({lbl})" if lbl else "Subtracting")
            if "divide" in pipeline:
                d = pipeline["divide"]
                lbl = _method_label(d) if d.get("source") == "aggregate" else ""
                if div_src != "sliding_aggregate":
                    parts.append(f"Dividing ({lbl})" if lbl else "Dividing")
                else:
                    parts.append("Normalizing")
            msg = " & ".join(parts) + "..."
        else:
            msg = None
        if msg:
            with LoadingManager(self, msg, blocking_label=self.ui.blocking_status, blocking_delay_ms=0):
                self.ui.image_viewer.data.set_ops_pipeline(pipeline)
                self._set_preview_frame_for_ops()
                if not skip_update:
                    keep = self._is_sliding_mean_preview() or (has_range_step and not self._is_aggregate_view())
                    self.ui.image_viewer.update_image(keep_timestep=keep)
        else:
            self.ui.image_viewer.data.set_ops_pipeline(None)
            self.ui.image_viewer.data.preview_frame = None
            if not skip_update:
                self.ui.image_viewer.update_image()

    def update_view_mode(self) -> None:
        """Switch between Single Frame (Reduce=None) and Aggregated (Reduce=Mean/Max/etc)."""
        is_agg = self._is_aggregate_view()
        if is_agg:
            data = getattr(self.ui.image_viewer, "data", None)
            cache_empty = (
                data is not None
                and getattr(data, "_redop", None) is None
                and len(getattr(data, "_result_cache", {})) == 0
            )
            if getattr(self, "_aggregate_first_open", True) and cache_empty:
                self.reset_selection_range(skip_apply=True)
                self._aggregate_first_open = False
        self._update_selection_visibility()
        if is_agg:
            self.apply_ops(skip_update=True)
            self.apply_aggregation()
        else:
            with LoadingManager(self, "Switching...", blocking_label=self.ui.blocking_status, blocking_delay_ms=0):
                self.ui.image_viewer.unravel()
            self.update_statusbar()
            self.update_bench()

    def _on_selection_changed(self) -> None:
        """Selection geaendert (Range-Drag oder Spinbox) -> Frame auf Range-Start, Aggregation neu."""
        s = self.ui.spinbox_crop_range_start.value()
        self.ui.image_viewer.setCurrentIndex(s)
        self.ui.image_viewer.timeLine.setPos((s, 0))
        # Update pipeline (bounds) first; skip update_image when aggregate to avoid double mean
        self.apply_ops(skip_update=self._is_aggregate_view())
        if self._is_aggregate_view():
            self.apply_aggregation()

    def apply_aggregation(self) -> None:
        """Apply reduction over Selection [Start, End] when Reduce != None."""
        if not self._is_aggregate_view():
            return
        data = getattr(self.ui.image_viewer, "data", None)
        if data is not None:
            data.preview_frame = None
        op = self.ui.combobox_reduce.currentData()
        if op is None:  # "None - current frame"
            with LoadingManager(self, "Switching...", blocking_label=self.ui.blocking_status, blocking_delay_ms=0):
                self.ui.image_viewer.unravel()
            self.update_statusbar()
            self.update_bench()
            return
        bounds = (
            self.ui.spinbox_crop_range_start.value(),
            self.ui.spinbox_crop_range_end.value(),
        )
        # Include range method when pipeline has subtract/divide with Range
        p = getattr(data, "_ops_pipeline", None) or {}
        sub = p.get("subtract") or {}
        div = p.get("divide") or {}
        range_label = ""
        for s in (sub, div):
            if s.get("source") == "aggregate":
                m = s.get("method", "MEAN")
                range_label = f" (range: {m.name if hasattr(m, 'name') else m})"
                break
        msg = f"Computing {op.name}{range_label}..."
        with LoadingManager(self, msg, blocking_label=self.ui.blocking_status, blocking_delay_ms=0) as lm:
            self.ui.image_viewer.reduce(op, bounds=bounds)
        s, e = bounds
        self.ui.image_viewer.timeLine.setPos((s, 0))
        self.update_statusbar()
        self.update_bench()
        log(f"{op.name} in {lm.duration:.2f}s")

    def update_roi_settings(self) -> None:
        self.ui.measure_roi.show_in_mm = self.ui.checkbox_mm.isChecked()
        self.ui.measure_roi.n_px = self.ui.spinbox_pixel.value()
        self.ui.measure_roi.px_in_mm = self.ui.spinbox_mm.value()
        if not self.ui.checkbox_measure_roi.isChecked():
            return
        self.ui.measure_roi.update_labels()

    def save_settings(self):
        settings.set("window/docks", self.ui.dock_area.saveState())
        screen_geometry = QApplication.primaryScreen().availableGeometry()
        settings.set("window/relative_size",
            self.width() / screen_geometry.width(),
        )

    # --- PCA ---
    def pca_calculate(self) -> None:
        if self.ui.button_pca_show.isChecked():
            self.ui.button_pca_show.setChecked(False)
            self.pca_adapter.reset_view()
            self.ui.button_pca_show.setText("View PCA")
        self.ui.label_pca_time.setText("")
        self.ui.pca_variance_plot.hide()
        self.ui.pca_variance_plot.clear()
        self.ui.table_pca_results.hide()
        self.ui.table_pca_results.setRowCount(0)
        n = self.ui.spinbox_pcacomp_target.value()
        exact = self.ui.checkbox_pca_exact.isChecked()
        self.pca_adapter.calculate(n_components=n, exact=exact)

    def pca_on_started(self) -> None:
        self._pca_calc_start_time = time.perf_counter()
        self.ui.button_pca_calc.setEnabled(False)
        self.ui.checkbox_pca_exact.setEnabled(False)
        self.ui.spinbox_pcacomp_target.setEnabled(False)
        self.ui.label_pca_time.setText("Calculating...")
        self.ui.blocking_status.setText("BUSY")
        self.ui.blocking_status.setStyleSheet(get_style("busy"))

    def pca_on_finished(self) -> None:
        duration = time.perf_counter() - getattr(self, "_pca_calc_start_time", 0)
        self.ui.button_pca_calc.setEnabled(True)
        self.ui.checkbox_pca_exact.setEnabled(True)
        self.ui.label_pca_time.setText(f"Calculated in {duration:.2f}s")
        self.ui.blocking_status.setText("IDLE")
        self.ui.blocking_status.setStyleSheet(get_style("idle"))

        n_comps = self.pca_adapter.max_components
        self.ui.spinbox_pcacomp.setMaximum(n_comps)
        self.ui.spinbox_pcacomp.setEnabled(True)
        self.ui.spinbox_pcacomp.setValue(n_comps if self.ui.checkbox_pca_exact.isChecked() else 1)
        self.ui.combobox_pca.setEnabled(True)
        self.ui.button_pca_show.setEnabled(True)
        self._pca_on_mode_changed()
        self._pca_update_variance_plot()
        self._pca_update_timeline_label()
        QTimer.singleShot(0, self._pca_update_target_spinner_state)

    def _pca_update_variance_plot(self) -> None:
        """Fill variance plot with dual axis: cumulative (red, 0-100), individual (green, actual)."""
        if not self.pca_adapter.is_calculated:
            self.ui.pca_variance_plot.hide()
            self.ui.table_pca_results.hide()
            if getattr(self.ui.pca_variance_plot, "_pca_vb2", None):
                self.ui.pca_variance_plot._pca_vb2 = None
            return
        old_vb = getattr(self.ui.pca_variance_plot, "_pca_vb2", None)
        if old_vb is not None and old_vb.scene():
            old_vb.scene().removeItem(old_vb)
        self.ui.pca_variance_plot._pca_vb2 = None
        self.ui.pca_variance_plot.clear()
        x, indiv, cumul = self.pca_adapter.variance_curve_data()
        if len(x) == 0:
            self.ui.pca_variance_plot.hide()
            self.ui.table_pca_results.hide()
            return
        self.ui.pca_variance_plot.show()
        self.ui.table_pca_results.show()
        max_modes = len(x)
        pi = self.ui.pca_variance_plot.getPlotItem()
        pi.setXRange(0, max_modes)
        pi.getViewBox().setLimits(xMin=0, xMax=max_modes)
        indiv_max = float(indiv.max()) if indiv.size > 0 else 1.0
        pen_cumul = (220, 80, 80)
        pen_indiv = (80, 200, 100)
        pi.getAxis("left").setPen(pg.mkPen(pen_cumul))
        pi.getAxis("left").setTextPen(pg.mkPen(pen_cumul))
        pi.setLabel("left", "cum. Var [%]", color=pen_cumul)
        pi.setYRange(0, 100)
        cumul_curve = self.ui.pca_variance_plot.plot(x, cumul, pen=pen_cumul, name="Cumulative")
        cumul_curve.setPen(pen_cumul, width=2)
        vb2 = pg.ViewBox()
        pi.showAxis("right")
        pi.scene().addItem(vb2)
        pi.getAxis("right").linkToView(vb2)
        vb2.setXLink(pi)
        pi.getAxis("right").setPen(pg.mkPen(pen_indiv))
        pi.getAxis("right").setTextPen(pg.mkPen(pen_indiv))
        pi.getAxis("right").setLabel("ind. Var [%]", color=pen_indiv)
        vb2.setGeometry(pi.getViewBox().sceneBoundingRect())
        vb2.linkedViewChanged(pi.getViewBox(), vb2.XAxis)
        if not getattr(self.ui.pca_variance_plot, "_pca_sync_connected", False):
            pi.getViewBox().sigResized.connect(lambda: _pca_sync_vb2(self.ui.pca_variance_plot))
            self.ui.pca_variance_plot._pca_sync_connected = True
        indiv_item = pg.PlotCurveItem(x, indiv, pen=pen_indiv)
        vb2.addItem(indiv_item)
        vb2.setYRange(0, indiv_max)
        self.ui.pca_variance_plot._pca_vb2 = vb2
        k = self.ui.spinbox_pcacomp.value() if self.ui.combobox_pca.currentText() == "Reconstruction" else 0
        if k > 0 and k <= len(cumul):
            kx, ky = float(x[k - 1]), float(cumul[k - 1])
            pt = pg.ScatterPlotItem([kx], [ky], symbol="o", size=12, pen=pg.mkPen("w", width=2))
            pi.addItem(pt)
            txt = pg.TextItem(text=f"{ky:.1f}%", anchor=(0, 0.5), color="w")
            txt.setPos(kx, ky)
            pi.addItem(txt)
        pi.showGrid(x=True, y=True, alpha=0.4)
        self.ui.table_pca_results.setRowCount(3)
        self.ui.table_pca_results.setColumnCount(max_modes + 1)
        self.ui.table_pca_results.setItem(0, 0, QTableWidgetItem("[%]"))
        self.ui.table_pca_results.setItem(1, 0, QTableWidgetItem("Var"))
        self.ui.table_pca_results.setItem(2, 0, QTableWidgetItem("Cumul"))
        for i in range(max_modes):
            self.ui.table_pca_results.setItem(0, i + 1, QTableWidgetItem(str(int(x[i]))))
            self.ui.table_pca_results.setItem(1, i + 1, QTableWidgetItem(f"{indiv[i]:.2f}"))
            self.ui.table_pca_results.setItem(2, i + 1, QTableWidgetItem(f"{cumul[i]:.1f}"))
        self.ui.table_pca_results.setColumnWidth(0, 48)

    def _pca_update_timeline_label(self) -> None:
        """Set timeline mode label (top-right overlay): Component vs Frame."""
        if self.ui.button_pca_show.isChecked():
            mode = self.ui.combobox_pca.currentText()
            label = "Component" if mode == "Components" else "Frame"
        else:
            label = "Frame"
        self.ui.timeline_stack.label_timeline_mode.setText(label)
        self.ui.roi_plot.getPlotItem().setLabel("bottom", "")

    def _pca_update_variance_display(self) -> None:
        """Refresh variance plot (reconstruction point) when Components spinbox changes."""
        if self.pca_adapter.is_calculated and self.ui.button_pca_show.isChecked():
            self._pca_update_variance_plot()

    def _pca_update_target_spinner_state(self) -> None:
        """When Exact: Target Comp at max and disabled. When not Exact: enabled."""
        exact = self.ui.checkbox_pca_exact.isChecked()
        target = self.ui.spinbox_pcacomp_target
        if exact:
            target.setValue(target.maximum())
            target.setEnabled(False)
        else:
            target.setEnabled(True)

    def _pca_on_exact_changed(self) -> None:
        """React to Exact checkbox: Target Comp at max and disabled when Exact, enabled when not."""
        self._pca_update_target_spinner_state()

    def _pca_on_mode_changed(self) -> None:
        is_reconstruction = self.ui.combobox_pca.currentText() == "Reconstruction"
        self.ui.spinbox_pcacomp.setVisible(is_reconstruction)
        self.ui.checkbox_pca_include_mean.setVisible(is_reconstruction)
        self._pca_update_target_spinner_state()
        if self.ui.button_pca_show.isChecked():
            self.pca_update_view()
        self._pca_update_variance_display()
        self._pca_update_timeline_label()

    def pca_on_error(self, msg: str) -> None:
        self.ui.button_pca_calc.setEnabled(True)
        self.ui.checkbox_pca_exact.setEnabled(True)
        self._pca_update_target_spinner_state()
        self.ui.label_pca_time.setText("Error")
        self.ui.blocking_status.setText("IDLE")
        self.ui.blocking_status.setStyleSheet(get_style("idle"))
        log(f"PCA Error: {msg}", color="red")

    def pca_toggle_show(self) -> None:
        if self.ui.button_pca_show.isChecked():
            self._pca_capture_original_timeline()
            self.pca_update_view()
            self.ui.button_pca_show.setText("View Data")
        else:
            self._pca_original_timeline_curve = None
            self.ui.image_viewer.clear_reference_timeline_curve()
            self.pca_adapter.reset_view()
            self.ui.button_pca_show.setText("View PCA")
        self._pca_update_timeline_label()

    def _pca_capture_original_timeline(self) -> None:
        """Capture original ROI curve for reference when viewing PCA (Reconstruction mode)."""
        curves = getattr(self.ui.image_viewer, "roiCurves", [])
        if curves and self.pca_adapter._base_data is not None:
            x, y = curves[0].getData()
            if x is not None and y is not None and len(x) > 0 and len(y) > 0:
                self._pca_original_timeline_curve = (np.asarray(x).copy(), np.asarray(y).copy())

    def pca_update_view(self) -> None:
        if not self.ui.button_pca_show.isChecked():
            return

        mode = self.ui.combobox_pca.currentText()
        if mode == "Components":
            self.ui.image_viewer.clear_reference_timeline_curve()
            n = self.pca_adapter.max_components
            self.pca_adapter.show_components(n)
        else:
            if getattr(self, "_pca_original_timeline_curve", None) is not None:
                x, y = self._pca_original_timeline_curve
                self.ui.image_viewer.set_reference_timeline_curve(x, y)
            n = self.ui.spinbox_pcacomp.value()
            add_mean = self.ui.checkbox_pca_include_mean.isChecked()
            self.pca_adapter.show_reconstruction(n, add_mean=add_mean)
        self._pca_update_variance_display()
