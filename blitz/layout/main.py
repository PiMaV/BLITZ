import os
from timeit import default_timer as clock
from typing import Optional

import pyqtgraph as pg
from PyQt5.QtGui import QFont, QIcon
from PyQt5.QtWidgets import (QApplication, QButtonGroup, QCheckBox, QComboBox,
                             QDoubleSpinBox, QFileDialog, QHBoxLayout, QLabel,
                             QMainWindow, QMenu, QMenuBar, QPushButton,
                             QSpinBox, QStatusBar, QTabWidget, QVBoxLayout,
                             QWidget)
from pyqtgraph.dockarea import Dock, DockArea

from ..tools import get_available_ram, log, set_logger
from .tof import TOFAdapter
from .viewer import ImageViewer
from .widgets import LoadingDialog, LoggingTextEdit

TITLE = (
    "BLITZ V 1.5.2 : Bulk Loading & Interactive Time-series Zonal-analysis "
    "from INP Greifswald"
)


class MainWindow(QMainWindow):

    def __init__(
        self,
        window_ratio: float = .75,
        relative_size: float = .85,
    ) -> None:
        super().__init__()
        self.setWindowTitle(TITLE)

        self.window_ratio = window_ratio
        screen_geometry = QApplication.primaryScreen().availableGeometry()
        width = int(screen_geometry.width() * relative_size)
        height = int(screen_geometry.height() * relative_size)
        self.setGeometry(
            (screen_geometry.width() - width) // 2,
            (screen_geometry.height() - height) // 2,
            width,
            height,
        )
        self.image_viewer_size = int(window_ratio * self.width())
        self.border_size = int((1 - window_ratio) * self.width() / 2)

        self.dock_area = DockArea()
        self.setup_docks()
        self.setCentralWidget(self.dock_area)

        self.setup_logger()
        self.setup_image_and_line_viewers()

        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.last_file_dir = script_dir
        icon_path = os.path.join(script_dir, 'BLITZ.ico')
        self.setWindowIcon(QIcon(icon_path))

        self.setup_option_dock()
        self.setup_menu_and_status_bar()
        self.image_viewer.file_dropped.connect(self.load_images)

        log("Welcome to BLITZ")

    def setup_docks(self) -> None:
        viewer_height = self.height() - 2 * self.border_size

        self.dock_option = Dock(
            "Options",
            size=(self.border_size, viewer_height),
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
        self.dock_area.addDock(self.dock_option, 'right', self.dock_viewer)
        self.dock_area.addDock(self.dock_h_plot, 'top', self.dock_viewer)
        self.dock_area.addDock(self.dock_status, 'top', self.dock_v_plot)

    def setup_menu_and_status_bar(self) -> None:
        menubar = QMenuBar()

        file_menu = QMenu("File", self)
        file_menu.addAction("Open...").triggered.connect(self.browse_file)
        file_menu.addAction("Load TOF").triggered.connect(self.browse_tof)
        menubar.addMenu(file_menu)

        view_menu = QMenu("View", self)
        view_menu.addAction("Show / Hide Crosshair").triggered.connect(
            self.image_viewer.toggle_crosshair
        )
        menubar.addMenu(view_menu)

        self.setMenuBar(menubar)

        statusbar = QStatusBar()
        self.position_label = QLabel("")
        statusbar.addPermanentWidget(self.position_label)
        self.frame_label = QLabel("")
        statusbar.addWidget(self.frame_label)
        self.file_label = QLabel("")
        statusbar.addWidget(self.file_label)

        self.setStatusBar(statusbar)

    def setup_option_dock(self) -> None:
        self.option_tabwidget = QTabWidget()

        font_heading = QFont()
        font_heading.setBold(True)
        font_heading.setPointSize(14)
        style_heading = (
            "background-color: rgb(70, 70, 100);"
            "qproperty-alignment: AlignCenter;"
            "border: 3px solid rgb(10, 10, 40);"
        )

        lut_container = QWidget(self)
        lut_layout = QVBoxLayout()
        lut_container.setLayout(lut_layout)
        self.image_viewer.ui.histogram.setParent(None)
        lut_layout.addWidget(self.image_viewer.ui.histogram)

        file_container = QWidget(self)
        file_layout = QVBoxLayout()
        file_container.setLayout(file_layout)
        load_label = QLabel("Load File")
        load_label.setFont(font_heading)
        load_label.setStyleSheet(style_heading)
        file_layout.addWidget(load_label)
        self.load_8bit_checkbox = QCheckBox("8 bit")
        file_layout.addWidget(self.load_8bit_checkbox)
        self.size_ratio_spinbox = QDoubleSpinBox()
        self.size_ratio_spinbox.setRange(0, 1)
        self.size_ratio_spinbox.setValue(1)
        self.size_ratio_spinbox.setSingleStep(0.1)
        self.size_ratio_spinbox.setPrefix("Size-ratio: ")
        file_layout.addWidget(self.size_ratio_spinbox)
        self.subset_ratio_spinbox = QDoubleSpinBox()
        self.subset_ratio_spinbox.setRange(0, 1)
        self.subset_ratio_spinbox.setValue(1)
        self.subset_ratio_spinbox.setSingleStep(0.1)
        self.subset_ratio_spinbox.setPrefix("Subset-ratio: ")
        file_layout.addWidget(self.subset_ratio_spinbox)
        self.max_ram_spinbox = QDoubleSpinBox()
        self.max_ram_spinbox.setRange(.1, .8 * get_available_ram())
        self.max_ram_spinbox.setValue(1)
        self.max_ram_spinbox.setSingleStep(0.1)
        self.max_ram_spinbox.setPrefix("Max. RAM: ")
        file_layout.addWidget(self.max_ram_spinbox)
        load_btn = QPushButton("Open")
        load_btn.pressed.connect(self.browse_file)
        file_layout.addWidget(load_btn)
        save_label = QLabel("Save")
        save_label.setFont(font_heading)
        save_label.setStyleSheet(style_heading)
        file_layout.addWidget(save_label)
        export_btn = QPushButton("Export")
        export_btn.pressed.connect(self.image_viewer.exportClicked)
        file_layout.addWidget(export_btn)
        file_layout.addStretch()

        option_container = QWidget(self)
        option_layout = QVBoxLayout()
        option_container.setLayout(option_layout)
        view_label = QLabel("View")
        view_label.setFont(font_heading)
        view_label.setStyleSheet(style_heading)
        option_layout.addWidget(view_label)
        flip_x = QCheckBox("Flip x")
        flip_x.stateChanged.connect(
            lambda: self.image_viewer.manipulation("flip_x")
        )
        option_layout.addWidget(flip_x)
        flip_y = QCheckBox("Flip y")
        flip_y.stateChanged.connect(
            lambda: self.image_viewer.manipulation("flip_y")
        )
        option_layout.addWidget(flip_y)
        transpose = QCheckBox("Transpose")
        transpose.stateChanged.connect(
            lambda: self.image_viewer.manipulation("transpose")
        )
        option_layout.addWidget(transpose)

        timeline_label = QLabel("Timeline Operation")
        timeline_label.setFont(font_heading)
        timeline_label.setStyleSheet(style_heading)
        option_layout.addWidget(timeline_label)
        self.op_combobox = QComboBox()
        for op in self.image_viewer.AVAILABLE_OPERATIONS:
            self.op_combobox.addItem(op)
        self.op_combobox.currentIndexChanged.connect(self.operation_changed)
        option_layout.addWidget(self.op_combobox)
        norm_label = QLabel("Normalization")
        norm_label.setFont(font_heading)
        norm_label.setStyleSheet(style_heading)
        option_layout.addWidget(norm_label)
        norm_checkbox = QCheckBox("Open Toolbox")
        norm_checkbox.stateChanged.connect(self.image_viewer.normToggled)
        option_layout.addWidget(norm_checkbox)
        option_layout.addStretch()

        tools_container = QWidget(self)
        tools_layout = QVBoxLayout()
        tools_container.setLayout(tools_layout)
        mask_label = QLabel("Mask")
        mask_label.setFont(font_heading)
        mask_label.setStyleSheet(style_heading)
        tools_layout.addWidget(mask_label)
        mask_checkbox = QCheckBox("Show")
        mask_checkbox.clicked.connect(self.image_viewer.toggle_mask)
        tools_layout.addWidget(mask_checkbox)
        apply_btn = QPushButton("Apply")
        apply_btn.pressed.connect(self.image_viewer.apply_mask)
        apply_btn.pressed.connect(lambda: mask_checkbox.setChecked(False))
        tools_layout.addWidget(apply_btn)
        reset_btn = QPushButton("Reset")
        reset_btn.pressed.connect(self.image_viewer.reset)
        tools_layout.addWidget(reset_btn)

        roi_label = QLabel("Measure Tool")
        roi_label.setFont(font_heading)
        roi_label.setStyleSheet(style_heading)
        tools_layout.addWidget(roi_label)
        self.measure_checkbox = QCheckBox("Show")
        self.measure_checkbox.stateChanged.connect(
            self.image_viewer.measure_roi.toggle
        )
        tools_layout.addWidget(self.measure_checkbox)
        self.mm_checkbox = QCheckBox("Display in mm")
        self.mm_checkbox.stateChanged.connect(self.update_roi_settings)
        tools_layout.addWidget(self.mm_checkbox)
        self.pixel_spinbox = QSpinBox()
        self.pixel_spinbox.setPrefix("Pixels: ")
        self.pixel_spinbox.setMinimum(1)
        self.pixel_spinbox.valueChanged.connect(self.update_roi_settings)
        converter_layout = QHBoxLayout()
        converter_layout.addWidget(self.pixel_spinbox)
        self.mm_spinbox = QDoubleSpinBox()
        self.mm_spinbox.setPrefix("in mm: ")
        self.mm_spinbox.setValue(1.0)
        self.mm_spinbox.valueChanged.connect(self.update_roi_settings)
        converter_layout.addWidget(self.mm_spinbox)
        tools_layout.addLayout(converter_layout)
        tools_layout.addStretch()

        roi_label = QLabel("Timeline")
        roi_label.setFont(font_heading)
        roi_label.setStyleSheet(style_heading)
        tools_layout.addWidget(roi_label)
        self.image_viewer.ui.roiBtn.setParent(None)
        self.roi_button: QPushButton = self.image_viewer.ui.roiBtn
        self.roi_button.clicked.connect(
            lambda: self.tof_adapter.toggle_plot(set_off=True)
        )
        tools_layout.addWidget(self.roi_button)
        self.tof_button = QPushButton("TOF")
        self.tof_button.setCheckable(True)
        # NOTE: the order of connection here matters
        # roiClicked shows the plot again
        self.tof_button.clicked.connect(self.image_viewer.roiClicked)
        self.tof_button.clicked.connect(
            lambda: self.tof_adapter.toggle_plot(set_off=False)
        )
        self.off_button = QPushButton("OFF")
        self.off_button.setCheckable(True)
        self.off_button.clicked.connect(self.image_viewer.roiClicked)
        self.off_button.clicked.connect(
            lambda: self.tof_adapter.toggle_plot(set_off=True)
        )
        self.button_group = QButtonGroup()
        self.button_group.addButton(self.roi_button)
        self.button_group.addButton(self.tof_button)
        self.button_group.addButton(self.off_button)
        self.button_group.buttonClicked.connect(self.button_group_clicked)
        self.button_group.setExclusive(True)
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.roi_button)
        button_layout.addWidget(self.tof_button)
        button_layout.addWidget(self.off_button)
        tools_layout.addLayout(button_layout)

        self.option_tabwidget.addTab(lut_container, "LUT")
        self.option_tabwidget.addTab(file_container, "File")
        self.option_tabwidget.addTab(option_container, "Manipulation")
        self.option_tabwidget.addTab(tools_container, "Tools")

        self.dock_option.addWidget(self.option_tabwidget)

    def setup_logger(self) -> None:
        file_info_widget = QWidget()
        layout = QVBoxLayout()

        self.logger = LoggingTextEdit()
        self.logger.setReadOnly(True)
        layout.addWidget(self.logger)
        set_logger(self.logger)

        file_info_widget.setLayout(layout)
        self.dock_status.addWidget(file_info_widget)

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

        self.v_plot.setYLink(self.image_viewer.getView())
        self.h_plot.setXLink(self.image_viewer.getView())

    def update_statusbar_position(self, pos: tuple[int, int]) -> None:
        x, y, value = self.image_viewer.get_position_info(pos)
        self.position_label.setText(f"X: {x} | Y: {y} | Value: {value}")

    def update_statusbar_frame(self) -> None:
        frame, max_frame, name = self.image_viewer.get_frame_info()
        self.frame_label.setText(f"Frame: {frame} / {max_frame}")
        self.file_label.setText(f"File: {name}")

    def button_group_clicked(self, button: QPushButton) -> None:
        for btn in self.button_group.buttons():
            btn.setEnabled(True)
        button.setEnabled(False)

    def browse_tof(self) -> None:
        path, _ = QFileDialog.getOpenFileName()
        if not path:
            return
        self.tof_adapter.set_data(path, self.image_viewer.data.meta)
        self.tof_button.click()

    def browse_file(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            directory=self.last_file_dir
        )
        if file_path:
            self.last_filepath = os.path.dirname(file_path)
            self.load_images(file_path)

    def load_images(self, file_path: Optional[str] = None) -> None:
        start_time = clock()
        dialog = self.show_loading_dialog(
            f"Loading from file: {'...' if file_path is None else file_path}"
        )

        self.image_viewer.load_data(
            file_path,
            size=self.size_ratio_spinbox.value(),
            ratio=self.subset_ratio_spinbox.value(),
            convert_to_8_bit=self.load_8bit_checkbox.isChecked(),
            ram_size=self.max_ram_spinbox.value(),
        )

        self.close_loading_dialog(dialog)
        if file_path is not None:
            data_size_MB = self.image_viewer.data.image.nbytes / 2**20
            log(f"Loaded {data_size_MB:.2f} MB")
            log(f"Available RAM: {get_available_ram():.2f} GB")
            log(f"Seconds needed: {clock() - start_time:.2f}")
            self.update_statusbar_frame()

    def operation_changed(self) -> None:
        start_time = clock()
        dialog = self.show_loading_dialog(message="Calculating statistics...")
        log(f"Available RAM: {get_available_ram():.2f} GB")

        self.image_viewer.manipulation(self.image_viewer.AVAILABLE_OPERATIONS[
            self.op_combobox.currentText()
        ])

        self.close_loading_dialog(dialog)
        log(f"Seconds needed: {clock() - start_time:.2f}")
        log(f"Available RAM: {get_available_ram():.2f} GB")

    def show_loading_dialog(
        self,
        message: str = "Loading images...",
    ) -> LoadingDialog:
        self.loading_dialog = LoadingDialog(self, message)
        self.loading_dialog.dialog.show()
        QApplication.processEvents()
        return self.loading_dialog.dialog

    def close_loading_dialog(self, dialog) -> None:
        dialog.accept()

    def update_roi_settings(self) -> None:
        self.image_viewer.measure_roi.show_in_mm = self.mm_checkbox.isChecked()
        self.image_viewer.measure_roi.n_px = self.pixel_spinbox.value()
        self.image_viewer.measure_roi.px_in_mm = self.mm_spinbox.value()
        if not self.measure_checkbox.isChecked():
            return
        self.image_viewer.measure_roi.update_labels()
