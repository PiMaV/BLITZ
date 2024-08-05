import json
from pathlib import Path
from typing import Optional

from PyQt5.QtCore import QCoreApplication, QUrl
from PyQt5.QtGui import QDesktopServices, QKeySequence
from PyQt5.QtWidgets import QApplication, QFileDialog, QMainWindow, QShortcut

from .. import __version__, settings
from ..data.image import ImageData
from ..data.web import WebDataLoader
from ..tools import LoadingManager, get_available_ram, log
from .pca import PCAAdapter
from .rosee import ROSEEAdapter
from .tof import TOFAdapter
from .ui import UI_MainWindow

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

        self.tof_adapter = TOFAdapter(self.ui.roi_plot)
        self.pca_adapter = PCAAdapter(self.ui.image_viewer)
        self.rosee_adapter = ROSEEAdapter(
            self.ui.image_viewer,
            self.ui.h_plot,
            self.ui.v_plot,
        )
        self.setup_connections()
        self.reset_options()

        if (lut := settings.get("viewer/LUT")):
            try:
                self.ui.image_viewer.load_lut_config(lut)
            except:
                log("Failed to load LUT config given in the .ini file",
                    color="red")

        self.ui.checkbox_roi.setChecked(False)
        self.ui.checkbox_roi.setChecked(True)
        log("Welcome to BLITZ", color="pink")

    def setup_connections(self) -> None:
        # MainWindow connections
        self.shortcut_copy = QShortcut(QKeySequence("Ctrl+C"), self)
        self.shortcut_copy.activated.connect(self.on_strgC)

        # menu connections
        self.ui.action_open_file.triggered.connect(self.browse_file)
        self.ui.action_open_folder.triggered.connect(self.browse_folder)
        self.ui.action_load_tof.triggered.connect(self.browse_tof)
        self.ui.action_export.triggered.connect(self.export)
        self.ui.action_project_save.triggered.connect(self.sync_settings)
        self.ui.action_project_open.triggered.connect(self.load_settings)
        self.ui.action_restart.triggered.connect(restart)
        self.ui.action_link_inp.triggered.connect(
            lambda: QDesktopServices.openUrl(URL_INP)  # type: ignore
        )
        self.ui.action_link_github.triggered.connect(
            lambda: QDesktopServices.openUrl(URL_GITHUB)  # type: ignore
        )

        # image_viewer connections
        self.ui.image_viewer.file_dropped.connect(self.load_images_adapter)
        self.ui.roi_plot.norm_range.sigRegionChanged.connect(
            self.update_norm_range_labels
        )
        self.ui.roi_plot.crop_range.sigRegionChanged.connect(
            self.update_crop_range_labels
        )
        self.ui.image_viewer.scene.sigMouseMoved.connect(
            self.update_statusbar_position
        )
        self.ui.image_viewer.timeLine.sigPositionChanged.connect(
            self.update_statusbar
        )

        # lut connections
        self.ui.checkbox_autofit.stateChanged.connect(
            self.ui.image_viewer.toggle_fit_levels
        )
        self.ui.button_load_lut.pressed.connect(self.browse_lut)
        self.ui.button_export_lut.pressed.connect(self.save_lut)

        # option connections
        self.ui.button_open_file.pressed.connect(self.browse_file)
        self.ui.button_open_folder.pressed.connect(self.browse_folder)
        self.ui.button_connect.pressed.connect(self.start_web_connection)
        self.ui.button_disconnect.pressed.connect(
            lambda: self.end_web_connection(None)
        )
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
        self.ui.checkbox_crop_show_range.stateChanged.connect(
            self.ui.roi_plot.toggle_crop_range
        )
        self.ui.spinbox_crop_range_start.editingFinished.connect(
            self.update_crop_range
        )
        self.ui.spinbox_crop_range_end.editingFinished.connect(
            self.update_crop_range
        )
        self.ui.button_crop.clicked.connect(self.crop)
        self.ui.button_crop_undo.clicked.connect(self.undo_crop)
        self.ui.combobox_reduce.currentIndexChanged.connect(
            self.operation_changed
        )
        self.ui.spinbox_norm_range_start.editingFinished.connect(
            self.update_norm_range
        )
        self.ui.spinbox_norm_range_end.editingFinished.connect(
            self.update_norm_range
        )
        self.ui.combobox_norm.currentIndexChanged.connect(
            self._normalization_update
        )
        self.ui.checkbox_norm_show_range.stateChanged.connect(
            self.ui.roi_plot.toggle_norm_range
        )
        self.ui.button_bg_input.clicked.connect(self.search_background_file)
        self.ui.spinbox_norm_window.editingFinished.connect(
            self._normalization_update
        )
        self.ui.spinbox_norm_lag.editingFinished.connect(
            self._normalization_update
        )
        self.ui.spinbox_norm_beta.editingFinished.connect(
            self._normalization_update
        )
        self.ui.checkbox_norm_subtract.clicked.connect(
            lambda: self._normalization("subtract")
        )
        self.ui.checkbox_norm_divide.clicked.connect(
            lambda: self._normalization("divide")
        )
        self.ui.checkbox_measure_roi.stateChanged.connect(
            self.ui.measure_roi.toggle
        )
        self.ui.checkbox_show_circ_area_label.stateChanged.connect(
            self.ui.measure_roi.toggle_area_circ_labels
        )
        self.ui.checkbox_mm.stateChanged.connect(self.update_roi_settings)
        self.ui.spinbox_pixel.valueChanged.connect(self.update_roi_settings)
        self.ui.spinbox_mm.valueChanged.connect(self.update_roi_settings)
        self.ui.button_pca.clicked.connect(self.toggle_pca)
        self.ui.checkbox_rosee_h.stateChanged.connect(self.toggle_rosee)
        self.ui.checkbox_rosee_v.stateChanged.connect(self.toggle_rosee)

    def reset_options(self) -> None:
        self.ui.button_apply_mask.setChecked(False)
        self.ui.checkbox_flipx.setChecked(settings.get("data/flipped_x"))
        self.ui.checkbox_flipy.setChecked(settings.get("data/flipped_y"))
        self.ui.checkbox_transpose.setChecked(settings.get("data/transposed"))
        self.ui.combobox_reduce.setCurrentIndex(0)
        if self.ui.image_viewer.data.is_single_image():
            self.ui.combobox_reduce.setEnabled(False)
        else:
            self.ui.combobox_reduce.setEnabled(True)
        self.ui.checkbox_norm_range.setChecked(True)
        self.ui.checkbox_norm_range.setEnabled(True)
        self.ui.checkbox_norm_bg.setEnabled(False)
        self.ui.checkbox_norm_bg.setChecked(False)
        self.ui.checkbox_norm_lag.setChecked(False)
        self.ui.checkbox_norm_subtract.setChecked(False)
        self.ui.checkbox_norm_divide.setChecked(False)
        self.ui.spinbox_norm_beta.setValue(100)
        self.ui.button_bg_input.setText("[Select]")
        self.ui.image_viewer._background_image = None
        self.ui.checkbox_measure_roi.setChecked(False)
        self.ui.spinbox_crop_range_start.setValue(0)
        self.ui.spinbox_norm_window.setValue(1)
        self.ui.spinbox_norm_lag.setValue(0)
        self.ui.spinbox_norm_window.setMinimum(1)
        self.ui.spinbox_norm_window.setMaximum(
            self.ui.image_viewer.data.n_images-1
        )
        self.ui.spinbox_norm_lag.setMaximum(
            self.ui.image_viewer.data.n_images-1
        )
        self.ui.spinbox_crop_range_start.setMaximum(
            self.ui.image_viewer.data.n_images-1
        )
        self.ui.spinbox_crop_range_end.setMaximum(
            self.ui.image_viewer.data.n_images-1
        )
        self.ui.spinbox_crop_range_end.setValue(
            self.ui.image_viewer.data.n_images-1
        )
        self.ui.checkbox_crop_show_range.setChecked(False)
        self.ui.spinbox_norm_range_start.setValue(0)
        self.ui.spinbox_norm_range_start.setMaximum(
            self.ui.image_viewer.data.n_images-1
        )
        self.ui.spinbox_norm_range_end.setMaximum(
            self.ui.image_viewer.data.n_images-1
        )
        self.ui.spinbox_norm_range_end.setValue(
            self.ui.image_viewer.data.n_images-1
        )
        self.ui.checkbox_norm_show_range.setChecked(False)
        self.ui.checkbox_roi_drop.setChecked(
            self.ui.image_viewer.is_roi_on_drop_update()
        )
        self.ui.spinbox_width_v.setRange(
            0, self.ui.image_viewer.data.shape[0] // 2
        )
        self.ui.spinbox_width_h.setRange(
            0, self.ui.image_viewer.data.shape[1] // 2
        )
        self.ui.spinbox_pcacomp.setMaximum(self.ui.image_viewer.data.n_images)
        self.ui.spinbox_pcacomp.setValue(self.ui.image_viewer.data.n_images)

    def update_norm_range_labels(self) -> None:
        norm_range_ = self.ui.roi_plot.norm_range.getRegion()
        left, right = map(round, norm_range_)  # type: ignore
        self.ui.spinbox_norm_range_start.setValue(left)
        self.ui.spinbox_norm_range_end.setValue(right)
        self.ui.roi_plot.norm_range.setRegion((left, right))

    def update_crop_range_labels(self) -> None:
        crop_range_ = self.ui.roi_plot.crop_range.getRegion()
        left, right = map(round, crop_range_)  # type: ignore
        self.ui.spinbox_crop_range_start.setValue(left)
        self.ui.spinbox_crop_range_end.setValue(right)
        self.ui.roi_plot.crop_range.setRegion((left, right))

    def update_norm_range(self) -> None:
        self.ui.roi_plot.norm_range.setRegion(
            (self.ui.spinbox_norm_range_start.value(),
             self.ui.spinbox_norm_range_end.value())
        )
        self._normalization_update()

    def toggle_hvplot_markings(self) -> None:
        self.ui.h_plot.toggle_mark_position()
        self.ui.h_plot.draw_line()
        self.ui.v_plot.toggle_mark_position()
        self.ui.v_plot.draw_line()

    def update_crop_range(self) -> None:
        self.ui.roi_plot.crop_range.setRegion(
            (self.ui.spinbox_crop_range_start.value(),
             self.ui.spinbox_crop_range_end.value())
        )

    def crop(self) -> None:
        with LoadingManager(self, "Cropping..."):
            self.ui.image_viewer.crop(
                left=self.ui.spinbox_crop_range_start.value(),
                right=self.ui.spinbox_crop_range_end.value(),
                keep=self.ui.checkbox_crop_keep.isChecked(),
            )
        self.reset_options()

    def undo_crop(self) -> None:
        with LoadingManager(self, "Undo Cropping..."):
            success = self.ui.image_viewer.undo_crop()
            if not success:
                self.load_images(self.last_file_dir / self.last_file)
        self.reset_options()

    def apply_mask(self) -> None:
        with LoadingManager(self, "Masking..."):
            self.ui.image_viewer.apply_mask()

    def reset_mask(self) -> None:
        with LoadingManager(self, "Reset..."):
            self.ui.image_viewer.reset_mask()

    def change_roi(self) -> None:
        with LoadingManager(self, "Change ROI..."):
            self.ui.checkbox_roi_drop.setChecked(False)
            self.ui.image_viewer.change_roi()

    def image_mask(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            directory=str(self.last_file_dir),
        )
        with LoadingManager(self, "Masking..."):
            self.ui.image_viewer.image_mask(Path(file_path))

    def _normalization_update(self) -> None:
        name = None
        if self.ui.checkbox_norm_subtract.isChecked():
            name = "subtract"
        if self.ui.checkbox_norm_divide.isChecked():
            name = "divide"
        if name is not None:
            bounds = None
            if self.ui.checkbox_norm_range.isChecked():
                bounds = (
                    self.ui.spinbox_norm_range_start.value(),
                    self.ui.spinbox_norm_range_end.value(),
                )
            window_lag = None
            if self.ui.checkbox_norm_lag.isChecked():
                window_lag = (
                    self.ui.spinbox_norm_window.value(),
                    self.ui.spinbox_norm_lag.value(),
                )
            with LoadingManager(self, "Calculating...") as lm:
                self.ui.image_viewer.norm(
                    operation=name,
                    use=self.ui.combobox_norm.currentText(),
                    beta=self.ui.spinbox_norm_beta.value() / 100.0,
                    bounds=bounds,
                    background=self.ui.checkbox_norm_bg.isChecked(),
                    window_lag=window_lag,
                    force_calculation=True,
                )
            log(f"Normalized in {lm.duration:.2f}s")
            self.update_statusbar()

    def _normalization(self, name: str) -> None:
        if ((not self.ui.checkbox_norm_range.isChecked()
                and not self.ui.checkbox_norm_bg.isChecked()
                and not self.ui.checkbox_norm_lag.isChecked())
                or self.ui.image_viewer.data.is_single_image()):
            self.ui.checkbox_norm_subtract.setChecked(False)
            self.ui.checkbox_norm_divide.setChecked(False)
            return
        if (self.ui.checkbox_norm_divide.isChecked()
                or self.ui.checkbox_norm_subtract.isChecked()):
            self.ui.checkbox_norm_range.setEnabled(False)
            self.ui.checkbox_norm_bg.setEnabled(False)
            self.ui.button_bg_input.setEnabled(False)
            self.ui.combobox_reduce.setEnabled(False)
        else:
            self.ui.checkbox_norm_range.setEnabled(True)
            if self.ui.button_bg_input.text() == "[Remove]":
                self.ui.checkbox_norm_bg.setEnabled(True)
            self.ui.button_bg_input.setEnabled(True)
            self.ui.combobox_reduce.setEnabled(True)
        if name == "subtract" and self.ui.checkbox_norm_divide.isChecked():
            self.ui.checkbox_norm_divide.setChecked(False)
        elif name == "divide" and self.ui.checkbox_norm_subtract.isChecked():
            self.ui.checkbox_norm_subtract.setChecked(False)
        bounds = None
        if self.ui.checkbox_norm_range.isChecked():
            bounds = (
                self.ui.spinbox_norm_range_start.value(),
                self.ui.spinbox_norm_range_end.value(),
            )
        window_lag = None
        if self.ui.checkbox_norm_lag.isChecked():
            window_lag = (
                self.ui.spinbox_norm_window.value(),
                self.ui.spinbox_norm_lag.value(),
            )
        with LoadingManager(self, "Calculating...") as lm:
            normalized = self.ui.image_viewer.norm(
                operation=name,
                use=self.ui.combobox_norm.currentText(),
                beta=self.ui.spinbox_norm_beta.value() / 100.0,
                bounds=bounds,
                background=self.ui.checkbox_norm_bg.isChecked(),
                window_lag=window_lag,
            )
        if normalized:
            log(f"Normalized in {lm.duration:.2f}s")
        self.update_statusbar()

    def toggle_pca(self) -> None:
        show = (
            (self.ui.combobox_pca.currentText() == "Components" and not
                self.pca_adapter.components_active) or
            (self.ui.combobox_pca.currentText() == "Reconstruction" and not
                self.pca_adapter.reconstruction_active)
        )

        if show:
            with LoadingManager(self, "Calculating PCA...") as lm:
                self.pca_adapter.calculate(self.ui.spinbox_pcacomp.value())
            log(f"PCA in {lm.duration:.2f}s")
            if self.ui.combobox_pca.currentText() == "Components":
                self.pca_adapter.show_components()
            elif self.ui.combobox_pca.currentText() == "Reconstruction":
                self.pca_adapter.show_reconstruction()
        else:
            self.pca_adapter.hide()

    def toggle_rosee(self) -> None:
        self.rosee_adapter.toggle(
            horizontal=self.ui.checkbox_rosee_h.isChecked(),
            vertical=self.ui.checkbox_rosee_v.isChecked(),
        )
        self.rosee_adapter.update()

    def search_background_file(self) -> None:
        if self.ui.button_bg_input.text() == "[Select]":
            file, _ = QFileDialog.getOpenFileName(
                caption="Choose Background File",
                directory=str(self.last_file_dir),
            )
            if file and self.ui.image_viewer.load_background_file(Path(file)):
                self.ui.button_bg_input.setText("[Remove]")
                self.ui.checkbox_norm_bg.setEnabled(True)
                self.ui.checkbox_norm_bg.setChecked(True)
        else:
            self.ui.checkbox_norm_bg.setEnabled(False)
            self.ui.checkbox_norm_bg.setChecked(False)
            self.ui.image_viewer.unload_background_file()
            self.ui.button_bg_input.setText("[Select]")

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

    def browse_tof(self) -> None:
        path, _ = QFileDialog.getOpenFileName()
        if not path:
            return
        with LoadingManager(self, "Loading TOF data..."):
            self.tof_adapter.set_data(path, self.ui.image_viewer.data.meta)
        self.ui.checkbox_tof.setEnabled(True)
        self.ui.checkbox_tof.setChecked(True)

    def browse_file(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            directory=str(self.last_file_dir),
        )
        if file_path:
            self.load_images(Path(file_path))

    def browse_folder(self) -> None:
        folder_path = QFileDialog.getExistingDirectory(
            directory=str(self.last_file_dir),
        )
        if folder_path:
            self.load_images(Path(folder_path))

    def export(self) -> None:
        with LoadingManager(self, "Exporting..."):
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
            except:
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

    def load_images_adapter(
        self,
        file_path: Optional[Path | str] = None,
    ) -> None:
        if isinstance(file_path, str):
            file_path = Path(file_path)
        if file_path is not None:
            if file_path.suffix == ".ini":
                settings.new_settings(file_path)
                restart()
            else:
                self.load_images(file_path)
        else:
            self.load_images()

    def load_images(self, file_path: Optional[Path] = None) -> None:
        text = f"Loading {'...' if file_path is None else file_path}"
        with LoadingManager(self, text) as lm:
            self.ui.image_viewer.load_data(
                file_path,
                size_ratio=self.ui.spinbox_load_size.value(),
                subset_ratio=self.ui.spinbox_load_subset.value(),
                max_ram=self.ui.spinbox_max_ram.value(),
                convert_to_8_bit=self.ui.checkbox_load_8bit.isChecked(),
                grayscale=self.ui.checkbox_load_grayscale.isChecked(),
            )
        if file_path is not None:
            log(f"Loaded in {lm.duration:.2f}s")
            self.last_file_dir = Path(file_path).parent
            self.last_file = Path(file_path).name
            self.update_statusbar()
            self.reset_options()

    def operation_changed(self) -> None:
        self.update_statusbar()
        text = self.ui.combobox_reduce.currentText()
        if text != "-":
            self.ui.checkbox_norm_subtract.setEnabled(False)
            self.ui.checkbox_norm_divide.setEnabled(False)
        else:
            self.ui.checkbox_norm_subtract.setEnabled(True)
            self.ui.checkbox_norm_divide.setEnabled(True)
        if text == "-":
            self.ui.checkbox_autofit.setChecked(True)
            with LoadingManager(self, f"Unpacking ..."):
                self.ui.image_viewer.unravel()
            self.update_statusbar()
        else:
            self.ui.checkbox_autofit.setChecked(False)
            with LoadingManager(self, f"Loading {text}...") as lm:
                self.ui.image_viewer.reduce(text)
            self.update_statusbar()
            log(f"{text} in {lm.duration:.2f}s")

    def update_roi_settings(self) -> None:
        self.ui.measure_roi.show_in_mm = self.ui.checkbox_mm.isChecked()
        self.ui.measure_roi.n_px = self.ui.spinbox_pixel.value()
        self.ui.measure_roi.px_in_mm = self.ui.spinbox_mm.value()
        if not self.ui.checkbox_measure_roi.isChecked():
            return
        self.ui.measure_roi.update_labels()

    def sync_settings(self) -> None:
        if (path := settings.get("data/path")) != "":
            path = Path(path)
            self.last_file_dir = path.parent
            self.last_file = path.name
        settings.export()
        settings.set("viewer/LUT", self.ui.image_viewer.get_lut_config())
        self.ui.image_viewer.data.save_options()
        if self.last_file != "":
            settings.set("data/path", str(self.last_file_dir / self.last_file))
        screen_geometry = QApplication.primaryScreen().availableGeometry()
        settings.set(
            "window/relative_size",
            self.width() / screen_geometry.width(),
        )
        settings.set(
            "window/docks",
            self.ui.dock_area.saveState(),
        )
        settings.set("data/max_ram", self.ui.spinbox_max_ram.value())

    def load_settings(self) -> None:
        if settings.select():
            restart()
