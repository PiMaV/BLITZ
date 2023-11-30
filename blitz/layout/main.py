import os
from typing import Optional

import pyqtgraph as pg
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QIcon, QKeySequence
from PyQt5.QtWidgets import (QApplication, QCheckBox, QComboBox,
                             QDoubleSpinBox, QFileDialog, QHBoxLayout, QLabel,
                             QMainWindow, QMenu, QMenuBar, QPushButton,
                             QScrollArea, QShortcut, QSpinBox, QStatusBar,
                             QTabWidget, QVBoxLayout, QWidget)
from pyqtgraph.dockarea import Dock, DockArea

from .. import resources
from ..tools import (LoadingManager, LoggingTextEdit, get_available_ram, log,
                     setup_logger)
from .tof import TOFAdapter
from .viewer import ImageViewer

TITLE = (
    "BLITZ: Bulk Loading and Interactive Time series Zonal analysis "
    "(INP Greifswald)"
)


class MainWindow(QMainWindow):

    def __init__(
        self,
        window_ratio: float = .75,
        relative_size: float = .85,
    ) -> None:
        super().__init__()
        self.setWindowTitle(TITLE)
        self.setWindowIcon(QIcon(":/icon/blitz.ico"))

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
        self.setup_lut_dock()

        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.last_file_dir = script_dir

        self.setup_option_dock()
        self.setup_menu_and_status_bar()
        self.image_viewer.file_dropped.connect(self.load_images)

        self.shortcut_copy = QShortcut(QKeySequence("Ctrl+C"), self)
        self.shortcut_copy.activated.connect(self.on_strgC)

        log("Welcome to BLITZ")

    def setup_docks(self) -> None:
        viewer_height = self.height() - 2 * self.border_size

        self.dock_lookup = Dock(
            "LUT",
            size=(self.border_size, 0.7*viewer_height),
            hideTitle=True,
        )
        self.dock_option = Dock(
            "Options",
            size=(self.border_size, 0.3*viewer_height),
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
        # lut_container = QWidget(self)
        # lut_layout = QVBoxLayout()
        # lut_container.setLayout(lut_layout)
        self.image_viewer.ui.histogram.setParent(None)
        self.dock_lookup.addWidget(self.image_viewer.ui.histogram)
        # self.option_tabwidget.addTab(lut_container, "LUT")

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
        norm_label = QLabel("Normalization")
        norm_label.setStyleSheet(style_heading)
        option_layout.addWidget(norm_label)
        norm_checkbox = QCheckBox("Open Toolbox")
        norm_checkbox.stateChanged.connect(self.image_viewer.normToggled)
        norm_checkbox.setStyleSheet(style_options)
        option_layout.addWidget(norm_checkbox)
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
            directory=self.last_file_dir
        )
        if file_path:
            self.last_file_dir = os.path.dirname(file_path)
            self.load_images(file_path)

    def load_images(self, file_path: Optional[str] = None) -> None:
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

    def operation_changed(self) -> None:
        log(f"Available RAM: {get_available_ram():.2f} GB")
        with LoadingManager(self, "Calculating statistics...") as lm:
            self.image_viewer.manipulation(
                self.image_viewer.AVAILABLE_OPERATIONS[
                    self.op_combobox.currentText()
                ]
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
