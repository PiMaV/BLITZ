import json
from pathlib import Path
from typing import Optional

import pyqtgraph as pg
from PyQt5.QtCore import QCoreApplication, Qt
from PyQt5.QtGui import QFont, QIcon, QKeySequence
from PyQt5.QtWidgets import (QApplication, QCheckBox, QComboBox,
                             QDoubleSpinBox, QFileDialog, QFrame, QGridLayout,
                             QHBoxLayout, QLabel, QMainWindow, QMenu, QMenuBar,
                             QPushButton, QScrollArea, QShortcut, QSpinBox,
                             QStatusBar, QStyle, QTabWidget, QTextEdit,
                             QVBoxLayout, QWidget)
from pyqtgraph.dockarea import Dock, DockArea

from .. import resources, settings
from ..tools import (LoadingManager, LoggingTextEdit, get_available_ram, log,
                     setup_logger)
from .tof import TOFAdapter
from .viewer import ImageViewer

TITLE = (
    "BLITZ: Bulk Loading and Interactive Time series Zonal analysis "
    "(INP Greifswald)"
)


def restart(self) -> None:
    QCoreApplication.exit(settings.get("app/restart_exit_code"))


class MainWindow(QMainWindow):

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(TITLE)
        self.setWindowIcon(QIcon(":/icon/blitz.ico"))

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
        window_ratio = settings.get("window/ratio")
        self.image_viewer_size = int(window_ratio * self.width())
        self.border_size = int((1 - window_ratio) * self.width() / 2)

        self.dock_area = DockArea()
        self.setup_docks()
        self.setCentralWidget(self.dock_area)

        self.setup_logger()
        self.setup_image_and_line_viewers()
        self.setup_lut_dock()

        self.last_file_dir = Path.cwd()

        self.setup_option_dock()
        self.setup_menu_and_status_bar()
        self.image_viewer.file_dropped.connect(self.load_images_adapter)

        self.shortcut_copy = QShortcut(QKeySequence("Ctrl+C"), self)
        self.shortcut_copy.activated.connect(self.on_strgC)

        log("Welcome to BLITZ")
        self.load_images()

    def setup_docks(self) -> None:
        viewer_height = self.height() - 2 * self.border_size

        lut_ratio = settings.get("window/LUT_vertical_ratio")
        self.dock_lookup = Dock(
            "LUT",
            size=(self.border_size, lut_ratio*viewer_height),
            hideTitle=True,
        )
        self.dock_option = Dock(
            "Options",
            size=(self.border_size, (1-lut_ratio)*viewer_height),
            hideTitle=True,
        )
        self.dock_status = Dock(
            "File Metadata",
            size=(self.border_size, self.border_size),
            hideTitle=True,
        )
        self.dock_v_plot = Dock(
            "V Plot",
            size=(self.border_size, viewer_height),
            hideTitle=True,
        )
        self.dock_h_plot = Dock(
            "H Plot",
            size=(self.image_viewer_size, self.border_size),
            hideTitle=True,
        )
        self.dock_viewer = Dock(
            "Image Viewer",
            size=(self.image_viewer_size, viewer_height),
            hideTitle=True,
        )
        self.dock_t_line = Dock(
            "Timeline",
            size=(self.image_viewer_size, self.border_size),
            hideTitle=True,
        )

        self.dock_area.addDock(self.dock_t_line, 'bottom')
        self.dock_area.addDock(self.dock_viewer, 'top', self.dock_t_line)
        self.dock_area.addDock(self.dock_v_plot, 'left', self.dock_viewer)
        self.dock_area.addDock(self.dock_lookup, 'right', self.dock_viewer)
        self.dock_area.addDock(self.dock_option, 'top', self.dock_lookup)
        self.dock_area.addDock(self.dock_h_plot, 'top', self.dock_viewer)
        self.dock_area.addDock(self.dock_status, 'top', self.dock_v_plot)

    def setup_menu_and_status_bar(self) -> None:
        menubar = QMenuBar()

        file_menu = QMenu("File", self)
        file_menu.addAction("Open...").triggered.connect(self.browse_file)
        file_menu.addAction("Load TOF").triggered.connect(self.browse_tof)
        file_menu.addAction("Export").triggered.connect(
            self.image_viewer.exportClicked
        )
        file_menu.addSeparator()
        file_menu.addAction("Write .ini").triggered.connect(settings.export)
        file_menu.addAction("Select .ini").triggered.connect(settings.select)
        file_menu.addSeparator()
        file_menu.addAction("Restart").triggered.connect(restart)
        menubar.addMenu(file_menu)

        view_menu = QMenu("View", self)
        view_menu.addAction("Show / Hide Crosshair").triggered.connect(
            self.image_viewer.toggle_crosshair
        )
        menubar.addMenu(view_menu)

        self.setMenuBar(menubar)

        font_status = QFont()
        font_status.setPointSize(9)

        statusbar = QStatusBar()
        self.position_label = QLabel("")
        self.position_label.setFont(font_status)
        statusbar.addPermanentWidget(self.position_label)
        self.frame_label = QLabel("")
        self.frame_label.setFont(font_status)
        statusbar.addWidget(self.frame_label)
        self.file_label = QLabel("")
        self.file_label.setFont(font_status)
        statusbar.addWidget(self.file_label)
        self.loading_label = QLabel("")
        self.loading_label.setFont(font_status)
        statusbar.addWidget(self.loading_label)

        self.setStatusBar(statusbar)

    def setup_lut_dock(self) -> None:
        self.image_viewer.ui.histogram.setParent(None)
        self.dock_lookup.addWidget(self.image_viewer.ui.histogram)
        levels_button = QPushButton("Fit Levels")
        levels_button.pressed.connect(self.image_viewer.autoLevels)
        range_button = QPushButton("Show Full Range")
        range_button.pressed.connect(self.image_viewer.auto_histogram_range)
        lut_button_container = QWidget(self)
        lut_button_layout = QGridLayout()
        lut_button_layout.addWidget(levels_button, 0, 0)
        lut_button_layout.addWidget(range_button, 0, 1)
        load_button = QPushButton("Load")
        load_button.pressed.connect(self.browse_lut)
        export_button = QPushButton("Export")
        export_button.pressed.connect(self.save_lut)
        lut_button_layout.addWidget(load_button, 1, 0)
        lut_button_layout.addWidget(export_button, 1, 1)
        lut_button_container.setLayout(lut_button_layout)
        self.dock_lookup.addWidget(lut_button_container)

    def setup_option_dock(self) -> None:
        self.option_tabwidget = QTabWidget()

        style_heading = (
            "background-color: rgb(50, 50, 78);"
            "qproperty-alignment: AlignCenter;"
            "border-bottom: 5px solid rgb(13, 0, 26);"
            "font-size: 14pt;"
            "font: bold;"
        )
        style_options = (
            "font-size: 9pt;"
        )

        self.option_tabwidget.setStyleSheet(style_options)

        file_layout = QVBoxLayout()
        load_label = QLabel("Load Options")
        load_label.setStyleSheet(style_heading)
        file_layout.addWidget(load_label)
        load_hlay = QHBoxLayout()
        self.load_8bit_checkbox = QCheckBox("8 bit")
        self.load_8bit_checkbox.setStyleSheet(style_options)
        load_hlay.addWidget(self.load_8bit_checkbox)
        self.load_grayscale_checkbox = QCheckBox("Grayscale")
        self.load_grayscale_checkbox.setStyleSheet(style_options)
        load_hlay.addWidget(self.load_grayscale_checkbox)
        load_btn = QPushButton("Open ...")
        load_btn.pressed.connect(self.browse_file)
        load_btn.setStyleSheet(style_options)
        load_hlay.addWidget(load_btn)
        file_layout.addLayout(load_hlay)
        self.size_ratio_spinbox = QDoubleSpinBox()
        self.size_ratio_spinbox.setRange(0, 1)
        self.size_ratio_spinbox.setValue(1)
        self.size_ratio_spinbox.setSingleStep(0.1)
        self.size_ratio_spinbox.setPrefix("Size-ratio: ")
        self.size_ratio_spinbox.setStyleSheet(style_options)
        file_layout.addWidget(self.size_ratio_spinbox)
        self.subset_ratio_spinbox = QDoubleSpinBox()
        self.subset_ratio_spinbox.setRange(0, 1)
        self.subset_ratio_spinbox.setValue(1)
        self.subset_ratio_spinbox.setSingleStep(0.1)
        self.subset_ratio_spinbox.setPrefix("Subset-ratio: ")
        self.subset_ratio_spinbox.setStyleSheet(style_options)
        file_layout.addWidget(self.subset_ratio_spinbox)
        self.max_ram_spinbox = QDoubleSpinBox()
        self.max_ram_spinbox.setRange(.1, .8 * get_available_ram())
        self.max_ram_spinbox.setValue(1)
        self.max_ram_spinbox.setSingleStep(0.1)
        self.max_ram_spinbox.setPrefix("Max. RAM: ")
        self.max_ram_spinbox.setStyleSheet(style_options)
        file_layout.addWidget(self.max_ram_spinbox)
        mask_label = QLabel("Mask")
        mask_label.setStyleSheet(style_heading)
        file_layout.addWidget(mask_label)
        mask_checkbox = QCheckBox("Show")
        mask_checkbox.clicked.connect(self.image_viewer.toggle_mask)
        mask_checkbox.setStyleSheet(style_options)
        file_layout.addWidget(mask_checkbox)
        mask_holay = QHBoxLayout()
        apply_btn = QPushButton("Apply")
        apply_btn.pressed.connect(self.image_viewer.apply_mask)
        apply_btn.pressed.connect(lambda: mask_checkbox.setChecked(False))
        apply_btn.setStyleSheet(style_options)
        mask_holay.addWidget(apply_btn)
        reset_btn = QPushButton("Reset")
        reset_btn.pressed.connect(self.image_viewer.reset)
        reset_btn.setStyleSheet(style_options)
        mask_holay.addWidget(reset_btn)
        file_layout.addLayout(mask_holay)
        view_label = QLabel("View")
        view_label.setStyleSheet(style_heading)
        file_layout.addWidget(view_label)
        view_layout = QHBoxLayout()
        flip_x = QCheckBox("Flip x")
        flip_x.stateChanged.connect(
            lambda: self.image_viewer.manipulation("flip_x")
        )
        flip_x.setStyleSheet(style_options)
        view_layout.addWidget(flip_x)
        flip_y = QCheckBox("Flip y")
        flip_y.stateChanged.connect(
            lambda: self.image_viewer.manipulation("flip_y")
        )
        flip_y.setStyleSheet(style_options)
        view_layout.addWidget(flip_y)
        transpose = QCheckBox("Transpose")
        transpose.stateChanged.connect(
            lambda: self.image_viewer.manipulation("transpose")
        )
        transpose.setStyleSheet(style_options)
        view_layout.addWidget(transpose)
        file_layout.addLayout(view_layout)
        file_layout.addStretch()

        option_layout = QVBoxLayout()

        timeline_label = QLabel("Timeline Operation")
        timeline_label.setStyleSheet(style_heading)
        option_layout.addWidget(timeline_label)
        self.op_combobox = QComboBox()
        for op in self.image_viewer.AVAILABLE_OPERATIONS:
            self.op_combobox.addItem(op)
        self.op_combobox.currentIndexChanged.connect(self.operation_changed)
        self.op_combobox.setStyleSheet(style_options)
        option_layout.addWidget(self.op_combobox)
        auto_layout = QHBoxLayout()
        label_auto = QLabel("LUT:")
        label_auto.setStyleSheet(style_options)
        self.auto_levels_checkbox = QCheckBox("Fit Levels")
        self.auto_levels_checkbox.setStyleSheet(style_options)
        self.auto_levels_checkbox.setChecked(True)
        self.auto_range_checkbox = QCheckBox("Total Range")
        self.auto_range_checkbox.setStyleSheet(style_options)
        self.auto_range_checkbox.setChecked(True)
        auto_layout.addWidget(label_auto)
        auto_layout.addWidget(self.auto_levels_checkbox)
        auto_layout.addWidget(self.auto_range_checkbox)
        option_layout.addLayout(auto_layout)
        norm_label = QLabel("Normalization")
        norm_label.setStyleSheet(style_heading)
        option_layout.addWidget(norm_label)
        norm_range_label = QLabel("Range:")
        norm_range_label.setStyleSheet(style_options)
        norm_range_label_to = QLabel("-")
        norm_range_label_to.setStyleSheet(style_options)
        self.norm_range_start = QSpinBox()
        self.norm_range_start.setMinimum(0)
        self.norm_range_end = QSpinBox()
        self.norm_range_end.setMinimum(1)
        self.norm_range_start.valueChanged.connect(self.update_norm_range)
        self.norm_range_end.valueChanged.connect(self.update_norm_range)
        norm_range_checkbox = QCheckBox("")
        pixmapi = getattr(QStyle, "SP_DesktopIcon")
        norm_range_checkbox.setIcon(
            norm_range_checkbox.style().standardIcon(pixmapi)  # type: ignore
        )
        norm_range_checkbox.setStyleSheet(style_options)
        norm_range_checkbox.stateChanged.connect(self.toggle_norm_range_select)
        range_layout = QHBoxLayout()
        range_layout.addWidget(norm_range_label)
        range_layout.addWidget(self.norm_range_start)
        range_layout.addWidget(norm_range_label_to)
        range_layout.addWidget(self.norm_range_end)
        range_layout.addWidget(norm_range_checkbox)
        option_layout.addLayout(range_layout)
        norm_bg_label = QLabel("Background:")
        norm_bg_label.setStyleSheet(style_options)
        self.bg_input_button = QPushButton("Select")
        self.bg_input_button.clicked.connect(self.search_background_file)
        self.bg_remove_button = QPushButton()
        pixmapi = getattr(QStyle, "SP_DialogCancelButton")
        self.bg_remove_button.setIcon(
            self.bg_remove_button.style().standardIcon(pixmapi)  # type: ignore
        )
        self.bg_remove_button.clicked.connect(self.remove_background_file)
        bg_layout = QHBoxLayout()
        bg_layout.addWidget(norm_bg_label)
        bg_layout.addWidget(self.bg_input_button)
        bg_layout.addWidget(self.bg_remove_button)
        option_layout.addLayout(bg_layout)
        self.norm_range = pg.LinearRegionItem()
        self.norm_range.sigRegionChanged.connect(self.update_norm_range_labels)
        self.norm_range.setZValue(0)
        self.roi_plot.addItem(self.norm_range)
        self.norm_range.hide()
        self.norm_beta = QDoubleSpinBox()
        self.norm_beta.setPrefix("beta: ")
        self.norm_beta.setMinimum(0)
        self.norm_beta.setMaximum(1)
        self.norm_beta.setValue(1)
        self.norm_beta.setSingleStep(0.01)
        self.norm_subtract_box = QCheckBox("Subtract Mean")
        self.norm_subtract_box.clicked.connect(
            lambda: self._normalization("subtract")
        )
        self.norm_divide_box = QCheckBox("Divide Mean")
        self.norm_divide_box.clicked.connect(
            lambda: self._normalization("divide")
        )
        norm_layout = QGridLayout()
        norm_layout.addWidget(self.norm_beta, 0, 0, 2, 1)
        norm_layout.addWidget(self.norm_subtract_box, 0, 1, 1, 2)
        norm_layout.addWidget(self.norm_divide_box, 1, 1, 1, 2)
        hline = QFrame()
        hline.setFrameShape(QFrame.Shape.HLine)
        hline.setFrameShadow(QFrame.Shadow.Sunken)
        option_layout.addWidget(hline)
        option_layout.addLayout(norm_layout)
        option_layout.addStretch()

        tools_layout = QVBoxLayout()

        roi_label = QLabel("Measure Tool")
        roi_label.setStyleSheet(style_heading)
        tools_layout.addWidget(roi_label)
        self.measure_checkbox = QCheckBox("Show")
        self.measure_checkbox.stateChanged.connect(
            self.image_viewer.measure_roi.toggle
        )
        self.measure_checkbox.setStyleSheet(style_options)
        tools_layout.addWidget(self.measure_checkbox)
        self.mm_checkbox = QCheckBox("Display in mm")
        self.mm_checkbox.stateChanged.connect(self.update_roi_settings)
        self.mm_checkbox.setStyleSheet(style_options)
        tools_layout.addWidget(self.mm_checkbox)
        self.pixel_spinbox = QSpinBox()
        self.pixel_spinbox.setPrefix("Pixels: ")
        self.pixel_spinbox.setMinimum(1)
        self.pixel_spinbox.valueChanged.connect(self.update_roi_settings)
        converter_layout = QHBoxLayout()
        self.pixel_spinbox.setStyleSheet(style_options)
        converter_layout.addWidget(self.pixel_spinbox)
        self.mm_spinbox = QDoubleSpinBox()
        self.mm_spinbox.setPrefix("in mm: ")
        self.mm_spinbox.setValue(1.0)
        self.mm_spinbox.valueChanged.connect(self.update_roi_settings)
        self.mm_spinbox.setStyleSheet(style_options)
        converter_layout.addWidget(self.mm_spinbox)
        tools_layout.addLayout(converter_layout)

        roi_label = QLabel("Timeline")
        roi_label.setStyleSheet(style_heading)
        tools_layout.addWidget(roi_label)
        self.image_viewer.ui.roiBtn.setParent(None)
        self.image_viewer.ui.roiBtn = QCheckBox("ROI")
        self.roi_checkbox = self.image_viewer.ui.roiBtn
        self.roi_checkbox.stateChanged.connect(self.image_viewer.roiClicked)
        self.roi_checkbox.setChecked(True)
        self.tof_checkbox = QCheckBox("TOF")
        # NOTE: the order of connection here matters
        # roiClicked shows the plot again
        self.tof_checkbox.stateChanged.connect(self.tof_adapter.toggle_plot)
        self.tof_checkbox.setEnabled(False)
        checkbox_layout = QHBoxLayout()
        checkbox_layout.addStretch()
        self.roi_checkbox.setStyleSheet(style_options)
        checkbox_layout.addWidget(self.roi_checkbox)
        checkbox_layout.addStretch()
        self.tof_checkbox.setStyleSheet(style_options)
        checkbox_layout.addWidget(self.tof_checkbox)
        checkbox_layout.addStretch()
        tools_layout.addLayout(checkbox_layout)
        self.roi_drop_checkbox = QCheckBox("Update ROI only on Drop")
        self.roi_drop_checkbox.stateChanged.connect(
            lambda: self.image_viewer.toggle_roi_update_frequency(
                self.roi_drop_checkbox.isChecked()
            )
        )
        self.roi_drop_checkbox.setStyleSheet(style_options)
        tools_layout.addWidget(self.roi_drop_checkbox)
        tools_layout.addStretch()

        option_scrollarea = QScrollArea()
        option_scrollarea.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        option_scrollarea.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        tools_scrollarea = QScrollArea()
        tools_scrollarea.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        tools_scrollarea.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )

        file_scrollarea = QScrollArea()
        file_scrollarea.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        file_scrollarea.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )

        file_container = QWidget()
        file_container.setLayout(file_layout)
        file_scrollarea.setWidget(file_container)
        file_scrollarea.setWidgetResizable(True)

        option_container = QWidget()
        option_container.setLayout(option_layout)
        option_scrollarea.setWidget(option_container)
        option_scrollarea.setWidgetResizable(True)

        tools_container = QWidget()
        tools_container.setLayout(tools_layout)
        tools_scrollarea.setWidget(tools_container)
        tools_scrollarea.setWidgetResizable(True)

        self.option_tabwidget.addTab(file_scrollarea, "File")
        self.option_tabwidget.addTab(option_scrollarea, "Manipulation")
        self.option_tabwidget.addTab(tools_scrollarea, "Tools")

        self.dock_option.addWidget(self.option_tabwidget)

    def setup_logger(self) -> None:
        self.logger = LoggingTextEdit()
        self.logger.setReadOnly(True)

        file_info_widget = QWidget()
        layout = QVBoxLayout()
        layout.addWidget(self.logger)
        file_info_widget.setLayout(layout)
        self.dock_status.addWidget(file_info_widget)
        setup_logger(self.logger)

    def setup_image_and_line_viewers(self) -> None:
        v_plot_viewbox = pg.ViewBox()
        v_plot_viewbox.invertX()
        v_plot_viewbox.invertY()
        v_plot_item = pg.PlotItem(viewBox=v_plot_viewbox)
        v_plot_item.showGrid(x=True, y=True, alpha=0.4)
        self.v_plot = pg.PlotWidget(plotItem=v_plot_item)
        self.dock_v_plot.addWidget(self.v_plot)

        h_plot_viewbox = pg.ViewBox()
        h_plot_item = pg.PlotItem(viewBox=h_plot_viewbox)
        h_plot_item.showGrid(x=True, y=True, alpha=0.4)
        self.h_plot = pg.PlotWidget(plotItem=h_plot_item)
        self.dock_h_plot.addWidget(self.h_plot)

        self.image_viewer = ImageViewer(
            h_plot=self.h_plot,
            v_plot=self.v_plot,
        )
        self.dock_viewer.addWidget(self.image_viewer)

        # relocate the roiPlot to the timeline dock
        self.roi_plot = self.image_viewer.ui.roiPlot
        self.roi_plot.setParent(None)
        # this decoy prevents an error being thrown in the ImageView
        timeline_decoy = pg.PlotWidget(self.image_viewer.ui.splitter)
        timeline_decoy.hide()
        self.dock_t_line.addWidget(self.roi_plot)
        self.image_viewer.ui.menuBtn.setParent(None)

        self.tof_adapter = TOFAdapter(self.roi_plot)

        self.image_viewer.scene.sigMouseMoved.connect(
            self.update_statusbar_position
        )
        self.image_viewer.timeLine.sigPositionChanged.connect(
            self.update_statusbar_frame
        )
        self.roi_plot.keyPressEvent = self.override_timeline_keyPressEvent
        self.roi_plot.keyReleaseEvent = self.image_viewer.keyReleaseEvent

        self.v_plot.setYLink(self.image_viewer.getView())
        self.h_plot.setXLink(self.image_viewer.getView())

    def toggle_norm_range_select(self) -> None:
        if self.norm_range.isVisible():
            self.norm_range.hide()
        else:
            self.norm_range.show()

    def update_norm_range_labels(self) -> None:
        norm_range_ = self.norm_range.getRegion()
        self.norm_range_start.setValue(int(norm_range_[0]))  # type: ignore
        self.norm_range_end.setValue(int(norm_range_[1]))  # type: ignore

    def update_norm_range(self) -> None:
        self.norm_range.setRegion(
            (self.norm_range_start.value(), self.norm_range_end.value())
        )

    def _normalization(self, name: str) -> None:
        if name == "subtract" and self.norm_divide_box.isChecked():
            self.norm_divide_box.setChecked(False)
        elif name == "divide" and self.norm_subtract_box.isChecked():
            self.norm_subtract_box.setChecked(False)
        self.image_viewer.norm(
            self.norm_range_start.value(), self.norm_range_end.value(),
            self.norm_divide_beta.value(), name=name,
        )

    def search_background_file(self) -> None:
        file, _ = QFileDialog.getOpenFileName(
            caption="Choose Background File",
            directory=str(self.last_file_dir),
        )
        pixmapi = getattr(QStyle, "SP_DialogApplyButton")
        self.bg_input_button.setIcon(
            self.bg_input_button.style().standardIcon(pixmapi)  # type: ignore
        )

    def remove_background_file(self) -> None:
        self.bg_input_button.setText("Select")

    def override_timeline_keyPressEvent(self, ev) -> None:
        self.roi_plot.scene().keyPressEvent(ev)
        self.image_viewer.keyPressEvent(ev)

    def on_strgC(self) -> None:
        cb = QApplication.clipboard()
        cb.clear()
        cb.setText(self.position_label.text())
        self.spinner.start()

    def update_statusbar_position(self, pos: tuple[int, int]) -> None:
        x, y, value = self.image_viewer.get_position_info(pos)
        self.position_label.setText(f"X: {x} | Y: {y} | Value: {value}")

    def update_statusbar_frame(self) -> None:
        frame, max_frame, name = self.image_viewer.get_frame_info()
        self.frame_label.setText(f"Frame: {frame} / {max_frame}")
        self.file_label.setText(f"File: {name}")

    def browse_tof(self) -> None:
        path, _ = QFileDialog.getOpenFileName()
        if not path:
            return
        with LoadingManager(self, "Loading TOF data..."):
            self.tof_adapter.set_data(path, self.image_viewer.data.meta)
        self.tof_checkbox.setEnabled(True)
        self.tof_checkbox.setChecked(True)

    def browse_file(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            directory=str(self.last_file_dir),
        )
        if file_path:
            self.last_file_dir = Path(file_path).parent
            self.load_images(Path(file_path))

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
                self.image_viewer.load_lut_config(lut_config)
            except:
                log("LUT could not be loaded. Make sure it is an "
                    "appropriately structured '.json' file.")

    def save_lut(self) -> None:
        path = QFileDialog.getExistingDirectory(
            caption="Choose LUT File Location",
            directory=str(self.last_file_dir),
        )
        if path:
            lut_config = self.image_viewer.get_lut_config()
            file = Path(path) / "lut_config.json"
            with open(file, "w", encoding="utf-8") as f:
                lut_config = json.dump(
                    lut_config,
                    f,
                    ensure_ascii=False,
                    indent=4,
                )

    def load_images_adapter(self, file_path: Optional[Path] = None) -> None:
        self.load_images(Path(file_path) if file_path is not None else None)

    def load_images(self, file_path: Optional[Path] = None) -> None:
        text = f"Loading {'...' if file_path is None else file_path}"
        with LoadingManager(self, text) as lm:
            self.image_viewer.load_data(
                file_path,
                size=self.size_ratio_spinbox.value(),
                ratio=self.subset_ratio_spinbox.value(),
                convert_to_8_bit=self.load_8bit_checkbox.isChecked(),
                ram_size=self.max_ram_spinbox.value(),
                grayscale=self.load_grayscale_checkbox.isChecked(),
            )
        if file_path is not None:
            data_size_MB = self.image_viewer.data.image.nbytes / 2**20
            log(f"Loaded {data_size_MB:.2f} MB")
            log(f"Available RAM: {get_available_ram():.2f} GB")
            log(f"Seconds needed: {lm.duration:.2f}")
            self.update_statusbar_frame()
            self.roi_drop_checkbox.setChecked(
                self.image_viewer.is_roi_on_drop_update()
            )
            self.norm_range_start.setValue(0)
            self.norm_range_start.setMaximum(self.image_viewer.data.n_images-1)
            self.norm_range_end.setMaximum(self.image_viewer.data.n_images)
            self.norm_range_end.setValue(self.image_viewer.data.n_images-1)

    def operation_changed(self) -> None:
        log(f"Available RAM: {get_available_ram():.2f} GB")
        text = self.op_combobox.currentText()
        with LoadingManager(self, f"Loading {text}...") as lm:
            self.image_viewer.manipulation(
                self.image_viewer.AVAILABLE_OPERATIONS[text],
                auto_levels=self.auto_levels_checkbox.isChecked(),
                auto_histogram_range=self.auto_range_checkbox.isChecked(),
            )
        log(f"Seconds needed: {lm.duration:.2f}")
        log(f"Available RAM: {get_available_ram():.2f} GB")

    def update_roi_settings(self) -> None:
        self.image_viewer.measure_roi.show_in_mm = self.mm_checkbox.isChecked()
        self.image_viewer.measure_roi.n_px = self.pixel_spinbox.value()
        self.image_viewer.measure_roi.px_in_mm = self.mm_spinbox.value()
        if not self.measure_checkbox.isChecked():
            return
        self.image_viewer.measure_roi.update_labels()
