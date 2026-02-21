import json
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import QCoreApplication, Qt, QTimer, QUrl
from PyQt6.QtGui import QDesktopServices, QKeySequence, QShortcut
from PyQt6.QtWidgets import QApplication, QFileDialog, QMainWindow
from pyqtgraph.graphicsItems.GradientEditorItem import Gradients

from .. import settings
from ..theme import get_style
from ..data.image import ImageData
from ..data.load import DataLoader, get_image_metadata, get_sample_format
from ..data.web import WebDataLoader
from ..tools import (LoadingManager, format_size_mb, get_available_ram,
                     get_cpu_percent, get_disk_io_mbs, get_used_ram, log)
from ..data.converters import get_ascii_metadata, load_ascii
from .dialogs import (AsciiLoadOptionsDialog, ImageLoadOptionsDialog,
                     RealCameraDialog, VideoLoadOptionsDialog)
from .rosee import ROSEEAdapter
from .winamp_mock import WinampMockLiveWidget
from .tof import TOFAdapter
from .ui import UI_MainWindow
from .filter_stack import FilterItemWidget

URL_GITHUB = QUrl("https://github.com/CodeSchmiedeHGW/BLITZ")
URL_INP = QUrl("https://www.inp-greifswald.de/")


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
        self._winamp_mock: WinampMockLiveWidget | None = None
        self._real_camera_dialog: RealCameraDialog | None = None

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
        if self._winamp_mock:
            self._winamp_mock.stop_stream()
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
        self.ui.spinbox_current_frame.valueChanged.connect(
            self._on_current_frame_spinbox_changed
        )

        # lut connections
        self.ui.button_autofit.clicked.connect(self.ui.image_viewer.autoLevels)
        self.ui.checkbox_auto_colormap.stateChanged.connect(
            self.ui.image_viewer.toggle_auto_colormap
        )
        self.ui.button_load_lut.pressed.connect(self.browse_lut)
        self.ui.button_export_lut.pressed.connect(self.save_lut)

        # option connections
        self.ui.button_connect.pressed.connect(self.start_web_connection)
        self.ui.button_disconnect.pressed.connect(
            lambda: self.end_web_connection(None)
        )
        self.ui.button_mock_live.pressed.connect(self.show_winamp_mock)
        self.ui.button_real_camera.pressed.connect(self.show_real_camera_dialog)
        self.ui.checkbox_flipx.clicked.connect(
            lambda: self.ui.image_viewer.manipulate("flip_x")
        )
        self.ui.checkbox_flipy.clicked.connect(
            lambda: self.ui.image_viewer.manipulate("flip_y")
        )
        self.ui.checkbox_transpose.clicked.connect(
            lambda: self.ui.image_viewer.manipulate("transpose")
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
        self.ui.button_ops_open_aggregate.clicked.connect(
            self._ops_open_aggregate_tab
        )

        # New Filter Stack Connections
        self.ui.filter_stack.pipeline_changed.connect(self.apply_ops)
        self.ui.filter_stack.load_reference_requested.connect(self.load_reference_for_filter)
        self.ui.spinbox_crop_range_start.editingFinished.connect(self.apply_ops)
        self.ui.spinbox_crop_range_end.editingFinished.connect(self.apply_ops)
        self.ui.combobox_reduce.currentIndexChanged.connect(self.apply_ops)

        self.ui.timeline_tabwidget.currentChanged.connect(
            self._on_timeline_tab_changed
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
        self.ui.radio_time_series.toggled.connect(self.update_view_mode)
        self.ui.radio_aggregated.toggled.connect(self.update_view_mode)
        self.ui.combobox_reduce.currentIndexChanged.connect(
            self.apply_aggregation
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
        self.ui.spinbox_isocurves.editingFinished.connect(
            self.update_isocurves
        )
        self.ui.spinbox_iso_smoothing.editingFinished.connect(
            self.update_isocurves
        )
        self.ui.image_viewer.image_changed.connect(self.update_bench)
        self.ui.image_viewer.image_size_changed.connect(self.update_bench)
        self._bench_timer = QTimer(self)
        self._bench_timer.timeout.connect(self.update_bench)
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

    def _on_timeline_tab_changed(self, index: int) -> None:
        """Tab-Wechsel = Modus: Frame->Single, Agg->Aggregated."""
        if index == 0:
            self.ui.radio_time_series.setChecked(True)
        else:
            self.ui.radio_aggregated.setChecked(True)

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

    def _update_selection_visibility(self) -> None:
        """Idx immer aktiv (wenn Daten). Range + Tab Agg nur bei Multi-Frame."""
        data = getattr(self.ui.image_viewer, "data", None)
        n = data.n_images if data else 0
        is_agg = self.ui.radio_aggregated.isChecked()
        needs_range = n > 1 and is_agg

        self.ui.timeline_tabwidget.setTabEnabled(1, n > 1)
        if n <= 1 and self.ui.timeline_tabwidget.currentIndex() == 1:
            self.ui.timeline_tabwidget.setCurrentIndex(0)
            self.ui.radio_time_series.setChecked(True)
        self.ui.spinbox_current_frame.setEnabled(n > 0)
        self.ui.range_section_widget.setEnabled(needs_range)
        if needs_range:
            self.ui.roi_plot.crop_range.show()
        else:
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
        settings.connect_sync(
            "data/sync",
            self.ui.checkbox_sync_file.stateChanged,
            self.ui.checkbox_sync_file.isChecked,
            self.ui.checkbox_sync_file.setChecked,
        )

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
            "transposed",
            self.ui.checkbox_transpose.stateChanged,
            self.ui.checkbox_transpose.isChecked,
            self.ui.checkbox_transpose.setChecked,
            lambda: self.ui.image_viewer.manipulate("transpose"),
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

    def _update_ops_file_visibility(self) -> None:
        """Show Load button when subtract or divide uses File."""
        sub = self.ui.combobox_ops_subtract_src.currentData() == "file"
        div = self.ui.combobox_ops_divide_src.currentData() == "file"
        self.ui.ops_file_widget.setVisible(sub or div)

    def _update_ops_slider_labels(self) -> None:
        self.ui.label_ops_subtract.setText(
            f"{self.ui.slider_ops_subtract.value()}%"
        )
        self.ui.label_ops_divide.setText(
            f"{self.ui.slider_ops_divide.value()}%"
        )

    def _ops_open_aggregate_tab(self) -> None:
        """Switch to Aggregate tab so user can configure range and reduce."""
        self.ui.timeline_tabwidget.setCurrentIndex(1)
        self.ui.radio_aggregated.setChecked(True)

    def _on_crop_range_for_ops(self) -> None:
        """Crop range changed -> update Ops if Aggregate is used."""
        self.apply_ops()

    def reset_options(self) -> None:
        # Signale blockieren waehrend Batch-Update (verhindert 21s emit-Kaskaden nach Load)
        _batch = [
            self.ui.roi_plot.crop_range,
            self.ui.spinbox_crop_range_start,
            self.ui.spinbox_crop_range_end,
            self.ui.spinbox_selection_window,
            self.ui.spinbox_current_frame,
            self.ui.combobox_reduce,
            self.ui.timeline_tabwidget,
        ]
        for w in _batch:
            w.blockSignals(True)
        try:
            self._reset_options_body()
        finally:
            for w in _batch:
                w.blockSignals(False)

    def _reset_options_body(self) -> None:
        self.ui.combobox_reduce.setCurrentIndex(0)
        self.ui.timeline_tabwidget.setCurrentIndex(0)
        self.ui.radio_time_series.setChecked(True)
        self.ui.checkbox_flipx.setChecked(False)
        self.ui.checkbox_flipy.setChecked(False)
        self.ui.checkbox_transpose.setChecked(False)
        if self.ui.image_viewer.data.is_single_image():
            self.ui.timeline_tabwidget.setTabEnabled(1, False)
        else:
            self.ui.timeline_tabwidget.setTabEnabled(1, True)

        # Reset Filter Stack
        self.ui.filter_stack.set_pipeline([])

        # Legacy
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
        self.ui.spinbox_current_frame.setMaximum(n_frames - 1)
        self.ui.spinbox_current_frame.setValue(
            min(self.ui.image_viewer.currentIndex, n_frames - 1)
        )
        self.ui.roi_plot.crop_range.setRegion(
            (0, self.ui.image_viewer.data.n_images - 1)
        )
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
        self.ui.spinbox_isocurves.setValue(1)
        self.ui.checkbox_rosee_active.setChecked(False)
        self.ui.checkbox_rosee_h.setEnabled(False)
        self.ui.checkbox_rosee_h.setChecked(True)
        self.ui.checkbox_rosee_v.setEnabled(False)
        self.ui.checkbox_rosee_v.setChecked(True)
        self.ui.checkbox_rosee_normalize.setEnabled(False)
        self.ui.spinbox_rosee_smoothing.setEnabled(False)
        self.ui.spinbox_isocurves.setEnabled(False)
        self.ui.checkbox_show_isocurve.setEnabled(False)
        self.ui.checkbox_show_isocurve.setChecked(False)
        self.ui.spinbox_iso_smoothing.setEnabled(False)
        self.ui.checkbox_rosee_local_extrema.setEnabled(False)
        self.ui.checkbox_rosee_show_lines.setEnabled(False)
        self.ui.checkbox_rosee_show_indices.setEnabled(False)
        self.ui.checkbox_rosee_in_image_h.setEnabled(False)
        self.ui.checkbox_rosee_in_image_h.setChecked(False)
        self.ui.checkbox_rosee_in_image_v.setEnabled(False)
        self.ui.checkbox_rosee_in_image_v.setChecked(False)
        self.ui.label_rosee_plots.setEnabled(False)
        self.ui.label_rosee_image.setEnabled(False)

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

    def reset_selection_range(self) -> None:
        """Set Selection auf volle Range [0, n-1]."""
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
        self._on_selection_changed()

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
            self.ui.spinbox_isocurves.setEnabled(True)
            self.ui.checkbox_show_isocurve.setEnabled(True)
            self.ui.spinbox_iso_smoothing.setEnabled(True)
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
            self.ui.spinbox_isocurves.setEnabled(False)
            self.ui.checkbox_show_isocurve.setEnabled(False)
            self.ui.spinbox_iso_smoothing.setEnabled(False)
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
            iso_smoothing=self.ui.spinbox_iso_smoothing.value(),
            show_index_lines=self.ui.checkbox_rosee_show_lines.isChecked(),
        )

    def update_isocurves(self) -> None:
        self.rosee_adapter.update_iso(
            on=self.ui.checkbox_show_isocurve.isChecked()
                and self.ui.checkbox_rosee_active.isChecked(),
            n=self.ui.spinbox_isocurves.value(),
            smoothing=self.ui.spinbox_iso_smoothing.value(),
        )

    def load_ops_file(self) -> None:
        """Legacy method for old UI button (kept for safety if called dynamically)."""
        pass

    def load_reference_for_filter(self, item_widget: FilterItemWidget) -> None:
        """Handle request from FilterItemWidget to load a reference file."""
        # If already loaded, remove it
        if item_widget.params.get("reference_loaded"):
            item_widget.set_reference(None)
            self.apply_ops()
            return

        file, _ = QFileDialog.getOpenFileName(
            caption="Choose Reference File",
            directory=str(self.last_file_dir),
        )
        if not file:
            return

        with LoadingManager(self, f"Loading reference {Path(file).name}", blocking_label=self.ui.blocking_status) as lm:
             # Use a temporary viewer/loader to load the data
             # We can reuse ImageData/DataLoader logic
             # But we need to load it into an ImageData object
             # DataLoader.load_data is usually coupled to viewer.
             # We can use the existing viewer to load it as background file?
             # No, because that sets it on self.ui.image_viewer._background_image
             # But we need it for THIS specific filter item.
             # So we should load it into a separate ImageData object.

             # Using DataLoader directly might be tricky as it calls back to viewer usually.
             # Let's see: DataLoader is in blitz/data/load.py. It has methods.
             # Actually `viewer.load_background_file` does logic:
             # meta = get_image_metadata(path)
             # image = _load_image(path, ...)
             # return ImageData(image, [meta])

             # I should replicate that.
             try:
                 # Minimal replication of load logic
                 # Check if video or image
                 # For simplicity, assume image or single frame, or same dimensions as current data
                 # Reuse `self.ui.image_viewer` logic if possible, or extract it.
                 # `viewer.load_background_file` calls `DataLoader.load_single_image_as_array` or similar?
                 # It calls `DataLoader._load_image` etc.

                 # Let's import what we need
                 from ..data.load import DataLoader

                 # We need to respect current viewer settings (8bit, etc) if possible, or just load raw.
                 # Usually reference should match main image in dimensions.

                 # Let's use `ImageData` from existing code if available or just load new.
                 # The existing `load_background_file` on viewer was:
                 # 1. get meta
                 # 2. load array
                 # 3. create ImageData
                 # 4. set _background_image

                 # We will do 1-3.

                 # Simplified load:
                 # We assume the user wants to load a file that matches current data shape (H, W).
                 # If it's a stack, we might only use first frame or average?
                 # Legacy `load_background_file` checks `img.shape`.

                 # Let's use DataLoader. But DataLoader methods are instance methods of viewer usually or mixed.
                 # Actually `DataLoader` is a mixin or base class? No, `ImageViewer` inherits `DataLoader`.
                 # So `self.ui.image_viewer` IS a `DataLoader`.

                 # We can use a temporary method on viewer or just use the viewer to load it but not set it as main image?
                 # `viewer` has `load_data` which sets main image.
                 # `viewer` has `load_background_file` which sets `_background_image`.

                 # I can adapt `load_background_file` to return the ImageData instead of setting it.
                 # But I shouldn't change `ImageViewer` public API too much if I can avoid it.
                 # Or I can just call `self.ui.image_viewer.load_reference_data(path)` if I add such a method.

                 # Let's implement the load logic here briefly using `DataLoader` static methods if any?
                 # `DataLoader` has `_load_image`, `_load_video`, `_load_folder`. They are static-ish?
                 # No, they are methods.

                 # Accessing private methods `_load_image` from `image_viewer` instance.
                 viewer = self.ui.image_viewer
                 path = Path(file)

                 if DataLoader._is_video(path):
                     # Not supported for reference usually in legacy?
                     # Legacy `load_background_file` supported it via `load_data` logic replication?
                     # Actually `load_background_file` in `blitz/data/load.py` (if it exists there)
                     # No, `load_background_file` is in `ImageViewer`.
                     pass

                 # Let's try to reuse `viewer`'s loading capabilities.
                 # Since `viewer` logic is complex (handling different file types),
                 # and I don't want to duplicate it.
                 # But `viewer` statefully sets `self.image`.

                 # Strategy:
                 # 1. Inspect `blitz/data/load.py` to see if I can use `DataLoader` cleanly.
                 # 2. Or, use `cv2` directly for simple image loading if that's 99% of use cases.
                 # 3. Or, refactor `ImageViewer.load_background_file` to be `load_auxiliary_file(path) -> ImageData`.

                 # Given I can't easily see `load.py` right now (I read it earlier but cache might be fuzzy on exact signatures).
                 # I recall `_load_image` returns `np.ndarray`.

                 # Let's check `load_background_file` in `viewer.py` if I could?
                 # No, I didn't read `viewer.py`.

                 # I'll implement a safe generic loader using `self.ui.image_viewer` methods if possible.
                 # `self.ui.image_viewer` has `load_background_file`. It returns boolean.
                 # And sets `self._background_image`.
                 # I can use that!
                 # 1. Call `viewer.load_background_file(path)`.
                 # 2. Grab `viewer._background_image`.
                 # 3. Set it to `item_widget`.
                 # 4. Clear `viewer._background_image` (set to None).

                 if viewer.load_background_file(path):
                     ref_data = viewer._background_image
                     viewer._background_image = None # Detach from global slot
                     item_widget.set_reference(ref_data)
                     self.apply_ops()

            except Exception as e:
                log(f"Failed to load reference: {e}", color="red")


    def on_strgC(self) -> None:
        cb = QApplication.clipboard()
        cb.clear()
        cb.setText(self.ui.position_label.text())

    def update_statusbar_position(self, pos: tuple[int, int]) -> None:
        x, y, value = self.ui.image_viewer.get_position_info(pos)
        self.ui.position_label.setText(f"X: {x} | Y: {y} | Value: {value}")

    def update_statusbar(self) -> None:
        frame, max_frame, name = self.ui.image_viewer.get_frame_info()
        self.ui.frame_label.setText(f"Frame: {frame} / {max_frame}")
        self.ui.file_label.setText(f"File: {name}")
        self.ui.ram_label.setText(
            f"Available RAM: {get_available_ram():.2f} GB"
        )
        x, y, value = self.ui.image_viewer.get_position_info()
        self.ui.position_label.setText(f"X: {x} | Y: {y} | Value: {value}")

    def _on_option_tab_changed(self, index: int) -> None:
        """Start/stop Bench live timer when Bench tab is visible."""
        bench_idx = getattr(
            self.ui, "bench_tab_index",
            self.ui.option_tabwidget.count() - 1,
        )
        if index == bench_idx:
            self._bench_live_tick = 0
            self._bench_timer.start(200)  # 0.2 s
            self.update_bench()
        else:
            self._bench_timer.stop()
            self.ui.label_bench_live.setText("")

    def update_bench(self) -> None:
        """Update Bench tab labels (CPU, RAM, Disk + sparklines, matrix stats)."""
        ram_free = get_available_ram()
        ram_used = get_used_ram()
        cpu = get_cpu_percent()
        disk_r, disk_w = get_disk_io_mbs()
        if self._bench_timer.isActive():
            self.ui.bench_sparklines.add_point(
                cpu, ram_used, ram_free, disk_r, disk_w
            )
        if self._bench_timer.isActive():
            tick = getattr(self, "_bench_live_tick", 0)
            self._bench_live_tick = tick + 1
            self.ui.label_bench_live.setText(
                "\u25cb LIVE" if tick % 2 else "\u25cf LIVE"
            )
        else:
            self.ui.label_bench_live.setText("")
        data = getattr(self.ui.image_viewer, "data", None)
        if data is None:
            self.ui.label_bench_raw.setText("Raw: —")
            self.ui.label_bench_result.setText("Result: —")
            self.ui.label_bench_mode.setText("View mode: —")
            self.ui.label_bench_cache.setText("Cache: —")
            return
        shape = data._image.shape
        dtype = data._image.dtype.name
        self.ui.label_bench_raw.setText(
            f"Raw: {format_size_mb(data._image.nbytes)} "
            f"({shape[0]}x{shape[1]}x{shape[2]}x{shape[3]}, {dtype})"
        )
        result_cache = getattr(data, "_result_cache", {})
        is_aggregate = getattr(data, "_redop", None) is not None
        ops_in_cache = sorted(set(k[0] for k in result_cache.keys())) if result_cache else []
        cache_str = f"Ready: {', '.join(ops_in_cache)}" if ops_in_cache else "—"
        self.ui.label_bench_cache.setText(f"Result cache: {cache_str}")

        if is_aggregate:
            op = data._redop
            bounds = getattr(data, "_agg_bounds", None)
            self.ui.label_bench_mode.setText(f"View mode: Aggregate ({op})")
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

    def _load_ascii(self, path: Path) -> None:
        """Load ASCII (.asc, .dat) via options dialog. Path from Open File/Folder or drop."""
        if path.exists():
            path = path.resolve()
        meta = get_ascii_metadata(path)
        if meta is None:
            log("Cannot read ASCII metadata", color="red")
            return
        dlg = AsciiLoadOptionsDialog(
            path, meta, parent=self,
            initial_params=self._ascii_session_defaults,
        )
        if not dlg.exec():
            return
        user_params = dlg.get_params()
        params = {k: v for k, v in user_params.items() if k != "mask_rel"}
        self._ascii_session_defaults = {
            "size_ratio": user_params["size_ratio"],
            "convert_to_8_bit": user_params["convert_to_8_bit"],
            "delimiter": user_params["delimiter"],
            "first_col_is_row_number": user_params["first_col_is_row_number"],
        }
        if "subset_ratio" in user_params:
            self._ascii_session_defaults["subset_ratio"] = user_params["subset_ratio"]
        if "mask_rel" in user_params:
            self._ascii_session_defaults["mask_rel"] = user_params["mask_rel"]
        else:
            self._ascii_session_defaults.pop("mask_rel", None)

        with LoadingManager(self, f"Loading {path}", blocking_label=self.ui.blocking_status) as lm:
            img = load_ascii(
                path,
                progress_callback=lm.set_progress,
                message_callback=lm.set_message,
                **params,
            )
        log(f"Loaded in {lm.duration:.2f}s")
        self.ui.image_viewer.set_image(img)
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
        else:
            self._web_connection.stop()
            self.ui.address_edit.setEnabled(True)
            self.ui.token_edit.setEnabled(True)
            self.ui.button_connect.setEnabled(True)
            self.ui.button_connect.setText("Connect")
            self.ui.button_disconnect.setEnabled(False)
            self._web_connection.deleteLater()

    def show_winamp_mock(self) -> None:
        """Open Mock Live: generates Lissajous viz, streams to viewer."""
        from PyQt6.QtCore import Qt
        if self._winamp_mock is None:
            self._winamp_mock = WinampMockLiveWidget(self)
            self._winamp_mock.setWindowFlags(
                Qt.WindowType.Tool | Qt.WindowType.WindowStaysOnTopHint
            )
            def _mock_frame_cb(img):
                self.ui.image_viewer.set_image(img, live_update=True)
                self.ui.image_viewer.setCurrentIndex(img.n_images - 1)
            self._winamp_mock.set_frame_callback(_mock_frame_cb)
            self._winamp_mock.setWindowIcon(self.windowIcon())
        self._winamp_mock.show()
        self._winamp_mock.raise_()
        self._winamp_mock.activateWindow()

    def show_real_camera_dialog(self) -> None:
        """Open Webcam dialog: sliders, Start/Stop, streams to viewer."""
        from PyQt6.QtCore import Qt
        if self._real_camera_dialog is None:
            self._camera_pending: ImageData | None = None
            self._camera_apply_timer = QTimer(self)
            self._camera_apply_timer.setSingleShot(True)

            _CAMERA_APPLY_MS = 25

            def _apply_camera_frame() -> None:
                if self._camera_pending is not None:
                    img = self._camera_pending
                    self._camera_pending = None
                    self.ui.image_viewer.set_image(img, live_update=True)
                    self.ui.image_viewer.setCurrentIndex(img.n_images - 1)
                if self._camera_pending is not None:
                    self._camera_apply_timer.start(_CAMERA_APPLY_MS)

            def _on_camera_frame(img: ImageData) -> None:
                self._camera_pending = img
                if not self._camera_apply_timer.isActive():
                    self._camera_apply_timer.start(1)

            self._camera_apply_timer.timeout.connect(_apply_camera_frame)

            def _camera_stop_cleanup() -> None:
                self._camera_apply_timer.stop()
                self._camera_pending = None

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
            project_file = path.parent / (path.name.split(".")[0] + ".blitz")
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
            elif self.ui.checkbox_sync_file.isChecked() and project_file.exists():
                self.load_project(project_file)
            else:
                self.load_images(path)
                if self.ui.checkbox_sync_file.isChecked():
                    settings.create_project(
                        path.parent / (path.name.split(".")[0] + ".blitz")
                    )
                    settings.set_project("path", path)
                    self.sync_project_preloading()
                    self.sync_project_postloading()
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
                channels = 1 if params["grayscale"] else 3
                est_bytes = (
                    meta.size[0] * meta.size[1] * channels * meta.frame_count
                    * (params["size_ratio"] ** 2)
                )

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
                        }
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
                    # Dialog nicht gezeigt: Session-Defaults anwenden (gleiche Einstellungen wie letztes Video)
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
                            }
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
            except Exception as e:
                log(f"Error reading image metadata: {e}", color="red")

        params.pop("mask_rel", None)

        with LoadingManager(self, f"Loading {path}", blocking_label=self.ui.blocking_status) as lm:
            self.ui.image_viewer.load_data(
                path,
                progress_callback=lm.set_progress,
                message_callback=lm.set_message,
                **params,
            )
        log(f"Loaded in {lm.duration:.2f}s")
        self.last_file_dir = path.parent
        self.last_file = path.name
        self.update_statusbar()
        self.reset_options()

    def apply_ops(self) -> None:
        """Build Ops pipeline from UI and set on data."""
        if self.ui.image_viewer.data.is_single_image():
            return

        pipeline = self.ui.filter_stack.get_pipeline()

        # Inject global aggregate settings if needed
        # The new stack items for subtract/divide specify "aggregate" source.
        # ImageData handles looking up the aggregate result based on bounds.
        # We need to ensure bounds are set?
        # Actually ImageData.compute_ref looks up "aggregate" using step["bounds"].
        # But filter_stack doesn't know bounds.
        # We should inject current bounds into steps that need it.

        current_bounds = (
            self.ui.spinbox_crop_range_start.value(),
            self.ui.spinbox_crop_range_end.value(),
        )
        current_reduce_method = self.ui.combobox_reduce.currentText()

        # Pass bounds/method to steps that use aggregate
        active_parts = []
        for step in pipeline:
            if step.get("source") == "aggregate":
                step["bounds"] = current_bounds
                step["method"] = current_reduce_method

            active_parts.append(step.get("type"))

        if pipeline:
            msg = f"Applying: {', '.join(active_parts)}..."
            with LoadingManager(self, msg, blocking_label=self.ui.blocking_status, blocking_delay_ms=0):
                self.ui.image_viewer.data.set_ops_pipeline(pipeline)
                self.ui.image_viewer.update_image()
        else:
            self.ui.image_viewer.data.set_ops_pipeline(None)
            self.ui.image_viewer.update_image()

    def update_view_mode(self) -> None:
        """Switch between Single Frame (Time Series) and Aggregated Image."""
        is_agg = self.ui.radio_aggregated.isChecked()
        self.ui.timeline_tabwidget.blockSignals(True)
        self.ui.timeline_tabwidget.setCurrentIndex(1 if is_agg else 0)
        self.ui.timeline_tabwidget.blockSignals(False)
        self._update_selection_visibility()
        if is_agg:
            self.apply_aggregation()
        else:
            with LoadingManager(self, "Switching...", blocking_label=self.ui.blocking_status, blocking_delay_ms=0):
                self.ui.image_viewer.unravel()
            self.update_statusbar()
            self.update_bench()

    def _on_selection_changed(self) -> None:
        """Selection geaendert (Range-Drag oder Spinbox) -> Aggregation neu berechnen."""
        if (self.ui.radio_aggregated.isChecked()
                and self.ui.combobox_reduce.currentText() != "None - current frame"):
            self.apply_aggregation()

    def apply_aggregation(self) -> None:
        """Apply reduction over Selection [Start, End] when in Aggregated mode."""
        if not self.ui.radio_aggregated.isChecked():
            return
        text = self.ui.combobox_reduce.currentText()
        if text == "None - current frame":
            with LoadingManager(self, "Switching...", blocking_label=self.ui.blocking_status, blocking_delay_ms=0):
                self.ui.image_viewer.unravel()
            self.update_statusbar()
            self.update_bench()
            return
        bounds = (
            self.ui.spinbox_crop_range_start.value(),
            self.ui.spinbox_crop_range_end.value(),
        )
        with LoadingManager(self, f"Computing {text}...", blocking_label=self.ui.blocking_status, blocking_delay_ms=0) as lm:
            self.ui.image_viewer.reduce(text, bounds=bounds)
        s, e = bounds
        center = (s + e) / 2
        self.ui.image_viewer.timeLine.setPos((center, 0))
        self.update_statusbar()
        self.update_bench()
        log(f"{text} in {lm.duration:.2f}s")

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
