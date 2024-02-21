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
                             QStatusBar, QStyle, QTabWidget, QVBoxLayout,
                             QWidget, QLayout)
from pyqtgraph.dockarea import Dock, DockArea

from .. import resources, settings
from ..tools import (LoadingManager, LoggingTextEdit, get_available_ram, log,
                     setup_logger)
from .tof import TOFAdapter
from .viewer import ImageViewer
from .widgets import TimePlot, LineExtractorPlot, MeasureROI

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
        if relative_size == 1.0:
            self.showMaximized()

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

        self.reset_options()
        log("Welcome to BLITZ")

    def setup_docks(self) -> None:
        border_size = int(0.25 * self.width() / 2)
        viewer_height = self.height() - 2 * border_size

        self.dock_lookup = Dock(
            "LUT",
            size=(border_size, 0.6*viewer_height),
            hideTitle=True,
        )
        self.dock_option = Dock(
            "Options",
            size=(border_size, 0.4*viewer_height),
            hideTitle=True,
        )
        self.dock_status = Dock(
            "File Metadata",
            size=(border_size, border_size),
            hideTitle=True,
        )
        self.dock_v_plot = Dock(
            "V Plot",
            size=(border_size, viewer_height),
            hideTitle=True,
        )
        image_viewer_size = int(0.75 * self.width())
        self.dock_h_plot = Dock(
            "H Plot",
            size=(image_viewer_size, border_size),
            hideTitle=True,
        )
        self.dock_viewer = Dock(
            "Image Viewer",
            size=(image_viewer_size, viewer_height),
            hideTitle=True,
        )
        self.dock_t_line = Dock(
            "Timeline",
            size=(image_viewer_size, border_size),
            hideTitle=True,
        )

        self.dock_area.addDock(self.dock_t_line, 'bottom')
        self.dock_area.addDock(self.dock_viewer, 'top', self.dock_t_line)
        self.dock_area.addDock(self.dock_v_plot, 'left', self.dock_viewer)
        self.dock_area.addDock(self.dock_lookup, 'right', self.dock_viewer)
        self.dock_area.addDock(self.dock_option, 'top', self.dock_lookup)
        self.dock_area.addDock(self.dock_h_plot, 'top', self.dock_viewer)
        self.dock_area.addDock(self.dock_status, 'top', self.dock_v_plot)
        if (docks_arrangement := settings.get("window/docks")):
            self.dock_area.restoreState(docks_arrangement)

    def setup_menu_and_status_bar(self) -> None:
        menubar = QMenuBar()

        file_menu = QMenu("File", self)
        file_menu.addAction("Open...").triggered.connect(self.browse_file)
        file_menu.addAction("Load TOF").triggered.connect(self.browse_tof)
        file_menu.addAction("Export").triggered.connect(
            self.image_viewer.exportClicked
        )
        file_menu.addSeparator()
        write_ini_action = file_menu.addAction("Write .ini")
        write_ini_action.triggered.connect(settings.export)
        write_ini_action.triggered.connect(self.sync_settings)
        file_menu.addAction("Select .ini").triggered.connect(settings.select)
        file_menu.addSeparator()
        file_menu.addAction("Restart").triggered.connect(restart)
        menubar.addMenu(file_menu)

        view_menu = QMenu("View", self)
        crosshair_action = view_menu.addAction("Show / Hide Crosshair")
        crosshair_action.triggered.connect(self.h_plot.toggle_line)
        crosshair_action.triggered.connect(self.v_plot.toggle_line)
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
        self._lut_file: str = ""
        self.image_viewer.ui.histogram.setParent(None)
        self.dock_lookup.addWidget(self.image_viewer.ui.histogram)
        levels_box = QCheckBox("Auto-Fit")
        levels_box.setChecked(True)
        levels_box.stateChanged.connect(self.image_viewer.toggle_fit_levels)
        lut_button_container = QWidget(self)
        load_button = QPushButton("Load")
        load_button.pressed.connect(self.browse_lut)
        export_button = QPushButton("Export")
        export_button.pressed.connect(self.save_lut)
        lut_button_layout = QGridLayout()
        lut_button_layout.addWidget(levels_box, 0, 0)
        lut_button_layout.addWidget(load_button, 0, 1)
        lut_button_layout.addWidget(export_button, 0, 2)
        lut_button_container.setLayout(lut_button_layout)
        self.dock_lookup.addWidget(lut_button_container)
        if (file := settings.get("viewer/LUT_source")) != "":
            try:
                with open(file, "r", encoding="utf-8") as f:
                    lut_config = json.load(f)
                self.image_viewer.load_lut_config(lut_config)
                self._lut_file = file
            except:
                log("Failed to load LUT config given in the .ini file")

    def setup_option_dock(self) -> None:
        self.option_tabwidget = QTabWidget()
        self.dock_option.addWidget(self.option_tabwidget)

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

        # --- File ---
        file_layout = QVBoxLayout()
        load_label = QLabel("Load Options")
        load_label.setStyleSheet(style_heading)
        file_layout.addWidget(load_label)
        load_hlay = QHBoxLayout()
        self.load_8bit_checkbox = QCheckBox("8 bit")
        load_hlay.addWidget(self.load_8bit_checkbox)
        self.load_grayscale_checkbox = QCheckBox("Grayscale")
        self.load_grayscale_checkbox.setChecked(True)
        load_hlay.addWidget(self.load_grayscale_checkbox)
        file_layout.addLayout(load_hlay)
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
        load_btn = QPushButton("Open File")
        load_btn.pressed.connect(self.browse_file)
        file_layout.addWidget(load_btn)
        file_layout.addStretch()
        self.create_option_tab(file_layout, "File")

        # --- View ---
        view_layout = QVBoxLayout()
        mask_label = QLabel("View")
        mask_label.setStyleSheet(style_heading)
        view_layout.addWidget(mask_label)
        viewchange_layout = QHBoxLayout()
        self.flip_x_box = QCheckBox("Flip x")
        self.flip_x_box.clicked.connect(
            lambda: self.image_viewer.manipulate("flip_x")
        )
        viewchange_layout.addWidget(self.flip_x_box)
        self.flip_y_box = QCheckBox("Flip y")
        self.flip_y_box.clicked.connect(
            lambda: self.image_viewer.manipulate("flip_y")
        )
        viewchange_layout.addWidget(self.flip_y_box)
        self.transpose_box = QCheckBox("Transpose")
        self.transpose_box.clicked.connect(
            lambda: self.image_viewer.manipulate("transpose")
        )
        viewchange_layout.addWidget(self.transpose_box)
        view_layout.addLayout(viewchange_layout)
        mask_label = QLabel("Mask")
        mask_label.setStyleSheet(style_heading)
        view_layout.addWidget(mask_label)
        mask_holay = QHBoxLayout()
        self.mask_checkbox = QCheckBox("Show")
        self.mask_checkbox.clicked.connect(self.image_viewer.toggle_mask)
        mask_holay.addWidget(self.mask_checkbox)
        apply_btn = QPushButton("Apply")
        apply_btn.pressed.connect(self.image_viewer.apply_mask)
        apply_btn.pressed.connect(lambda: self.mask_checkbox.setChecked(False))
        mask_holay.addWidget(apply_btn)
        reset_btn = QPushButton("Reset")
        reset_btn.pressed.connect(self.image_viewer.reset)
        mask_holay.addWidget(reset_btn)
        view_layout.addLayout(mask_holay)
        crosshair_label = QLabel("Crosshair")
        crosshair_label.setStyleSheet(style_heading)
        view_layout.addWidget(crosshair_label)
        width_holay = QHBoxLayout()
        width_label = QLabel("Line width:")
        width_holay.addWidget(width_label)
        self.width_spinbox_v = QSpinBox()
        self.width_spinbox_v.setRange(0, 1)
        self.width_spinbox_v.setValue(0)
        self.width_spinbox_v.setPrefix("H: ")
        self.width_spinbox_v.valueChanged.connect(self.v_plot.change_width)
        width_holay.addWidget(self.width_spinbox_v)
        self.width_spinbox_h = QSpinBox()
        self.width_spinbox_h.setRange(0, 1)
        self.width_spinbox_h.setValue(0)
        self.width_spinbox_h.setPrefix("V: ")
        self.width_spinbox_h.valueChanged.connect(self.h_plot.change_width)
        width_holay.addWidget(self.width_spinbox_h)
        view_layout.addLayout(width_holay)

        roi_label = QLabel("Timeline Plot")
        roi_label.setStyleSheet(style_heading)
        view_layout.addWidget(roi_label)
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
        checkbox_layout.addWidget(self.roi_checkbox)
        checkbox_layout.addStretch()
        checkbox_layout.addWidget(self.tof_checkbox)
        checkbox_layout.addStretch()
        view_layout.addLayout(checkbox_layout)
        self.roi_drop_checkbox = QCheckBox("Update ROI only on Drop")
        self.roi_drop_checkbox.stateChanged.connect(
            lambda: self.image_viewer.toggle_roi_update_frequency(
                self.roi_drop_checkbox.isChecked()
            )
        )
        view_layout.addWidget(self.roi_drop_checkbox)
        view_layout.addStretch()
        self.create_option_tab(view_layout, "View")

        # --- Timeline Operation ---
        timeop_layout = QVBoxLayout()
        timeline_label = QLabel("Reduction")
        timeline_label.setStyleSheet(style_heading)
        timeop_layout.addWidget(timeline_label)
        self.op_combobox = QComboBox()
        for op in self.image_viewer.AVAILABLE_OPERATIONS:
            self.op_combobox.addItem(op)
        self.op_combobox.currentIndexChanged.connect(self.operation_changed)
        timeop_layout.addWidget(self.op_combobox)
        norm_label = QLabel("Normalization")
        norm_label.setStyleSheet(style_heading)
        timeop_layout.addWidget(norm_label)
        self.norm_range_box = QCheckBox("Range:")
        self.norm_range_box.setChecked(True)
        norm_range_label_to = QLabel("-")
        self.norm_range_start = QSpinBox()
        self.norm_range_start.setMinimum(0)
        self.norm_range_end = QSpinBox()
        self.norm_range_end.setMinimum(1)
        self.norm_range_start.valueChanged.connect(self.update_norm_range)
        self.norm_range_end.valueChanged.connect(self.update_norm_range)
        self.norm_range_checkbox = QCheckBox("")
        map_ = getattr(QStyle, "SP_DesktopIcon")
        self.norm_range_checkbox.setIcon(
            self.norm_range_checkbox.style().standardIcon(map_)  # type: ignore
        )
        self.norm_range_checkbox.stateChanged.connect(
            self.roi_plot.toggle_norm_range
        )
        range_layout = QHBoxLayout()
        range_layout.addWidget(self.norm_range_box)
        range_layout.addWidget(self.norm_range_start)
        range_layout.addWidget(norm_range_label_to)
        range_layout.addWidget(self.norm_range_end)
        range_layout.addWidget(self.norm_range_checkbox)
        timeop_layout.addLayout(range_layout)
        self.norm_bg_box = QCheckBox("Background:")
        self.norm_bg_box.setEnabled(False)
        self.bg_input_button = QPushButton("[Select]")
        self.bg_input_button.clicked.connect(self.search_background_file)
        bg_layout = QHBoxLayout()
        bg_layout.addWidget(self.norm_bg_box)
        bg_layout.addWidget(self.bg_input_button)
        # bg_layout.addWidget(self.bg_remove_button)
        timeop_layout.addLayout(bg_layout)
        self.norm_beta = QDoubleSpinBox()
        self.norm_beta.setPrefix("beta: ")
        self.norm_beta.setMinimum(0)
        self.norm_beta.setMaximum(1)
        self.norm_beta.setValue(1)
        self.norm_beta.setSingleStep(0.01)
        self.norm_beta.editingFinished.connect(self._normalization_beta_update)
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
        timeop_layout.addWidget(hline)
        timeop_layout.addLayout(norm_layout)
        timeop_layout.addStretch()
        self.create_option_tab(timeop_layout, "Time")

        # --- Tools ---
        tools_layout = QVBoxLayout()
        roi_label = QLabel("Measure Tool")
        roi_label.setStyleSheet(style_heading)
        tools_layout.addWidget(roi_label)
        self.measure_checkbox = QCheckBox("Show")
        self.measure_checkbox.stateChanged.connect(self.measure_roi.toggle)
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
        self.create_option_tab(tools_layout, "Tools")

    def create_option_tab(self, layout: QLayout, name: str) -> None:
        scrollarea = QScrollArea()
        scrollarea.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        scrollarea.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        container = QWidget()
        container.setLayout(layout)
        scrollarea.setWidget(container)
        scrollarea.setWidgetResizable(True)
        self.option_tabwidget.addTab(scrollarea, name)

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
        # v_plot_viewbox = pg.ViewBox()
        # v_plot_viewbox.invertX()
        # v_plot_viewbox.invertY()
        # v_plot_item = pg.PlotItem(viewBox=v_plot_viewbox)
        # v_plot_item.showGrid(x=True, y=True, alpha=0.4)
        # self.v_plot = pg.PlotWidget(plotItem=v_plot_item)
        # self.dock_v_plot.addWidget(self.v_plot)

        # h_plot_viewbox = pg.ViewBox()
        # h_plot_item = pg.PlotItem(viewBox=h_plot_viewbox)
        # h_plot_item.showGrid(x=True, y=True, alpha=0.4)
        # self.h_plot = pg.PlotWidget(plotItem=h_plot_item)
        # self.dock_h_plot.addWidget(self.h_plot)

        self.image_viewer = ImageViewer()
        self.dock_viewer.addWidget(self.image_viewer)

        self.h_plot = LineExtractorPlot(self.image_viewer)
        self.dock_h_plot.addWidget(self.h_plot)
        self.v_plot = LineExtractorPlot(self.image_viewer, vertical=True)
        self.dock_v_plot.addWidget(self.v_plot)

        self.measure_roi = MeasureROI(self.image_viewer)

        # create a new timeline replacing roiPlot
        self.norm_range = pg.LinearRegionItem()
        self.norm_range.sigRegionChanged.connect(self.update_norm_range_labels)
        self.norm_range.setZValue(0)
        self.roi_plot = TimePlot(
            self.dock_t_line,
            self.image_viewer,
            self.norm_range,
        )
        self.roi_plot.showGrid(x=True, y=True, alpha=0.4)
        self.image_viewer.ui.roiPlot.setParent(None)
        self.image_viewer.ui.roiPlot = self.roi_plot
        while len(self.image_viewer.roiCurves) > 0:
            c = self.image_viewer.roiCurves.pop()
            c.scene().removeItem(c)
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

    def reset_options(self) -> None:
        self.mask_checkbox.setChecked(False)
        self.flip_x_box.setChecked(False)
        self.flip_y_box.setChecked(False)
        self.transpose_box.setChecked(False)
        self.op_combobox.setCurrentIndex(0)
        self.norm_range_box.setChecked(True)
        self.norm_range_box.setEnabled(True)
        self.norm_bg_box.setEnabled(False)
        self.norm_bg_box.setChecked(False)
        self.norm_subtract_box.setChecked(False)
        self.norm_divide_box.setChecked(False)
        self.norm_beta.setValue(1.0)
        self.bg_input_button.setText("[Select]")
        self.norm_range_checkbox.setChecked(False)
        self.image_viewer._background_image = None
        self.measure_checkbox.setChecked(False)
        self.norm_range_start.setValue(0)
        self.norm_range_start.setMaximum(self.image_viewer.data.n_images-1)
        self.norm_range_end.setMaximum(self.image_viewer.data.n_images)
        self.norm_range_end.setValue(self.image_viewer.data.n_images-1)
        self.roi_drop_checkbox.setChecked(
            self.image_viewer.is_roi_on_drop_update()
        )
        self.width_spinbox_v.setRange(0, self.image_viewer.data.shape[0] // 2)
        self.width_spinbox_h.setRange(0, self.image_viewer.data.shape[1] // 2)

    def update_norm_range_labels(self) -> None:
        norm_range_ = self.norm_range.getRegion()
        left, right = map(round, norm_range_)  # type: ignore
        self.norm_range_start.setValue(left)
        self.norm_range_end.setValue(right)
        self.norm_range.setRegion((left, right))

    def update_norm_range(self) -> None:
        self.norm_range.setRegion(
            (self.norm_range_start.value(), self.norm_range_end.value())
        )

    def _normalization_beta_update(self) -> None:
        name = None
        if self.norm_subtract_box.isChecked():
            name = "subtract"
        if self.norm_divide_box.isChecked():
            name = "divide"
        if name is not None:
            left = right = None
            if self.norm_range_box.isChecked():
                left = self.norm_range_start.value()
                right = self.norm_range_end.value()
            self.image_viewer.norm(
                operation=name,
                beta=self.norm_beta.value(),
                left=left,
                right=right,
                background=self.norm_bg_box.isChecked(),
                force_calculation=True,
            )

    def _normalization(self, name: str) -> None:
        if (not self.norm_range_box.isChecked()
                and not self.norm_bg_box.isChecked()):
            self.norm_subtract_box.setChecked(False)
            self.norm_divide_box.setChecked(False)
            return
        if (self.norm_divide_box.isChecked()
                or self.norm_subtract_box.isChecked()):
            self.norm_range_box.setEnabled(False)
            self.norm_bg_box.setEnabled(False)
            self.bg_input_button.setEnabled(False)
            self.op_combobox.setEnabled(False)
        else:
            self.norm_range_box.setEnabled(True)
            if self.bg_input_button.text() == "[Remove]":
                self.norm_bg_box.setEnabled(True)
            self.bg_input_button.setEnabled(True)
            self.op_combobox.setEnabled(True)
        if name == "subtract" and self.norm_divide_box.isChecked():
            self.norm_divide_box.setChecked(False)
        elif name == "divide" and self.norm_subtract_box.isChecked():
            self.norm_subtract_box.setChecked(False)
        left = right = None
        if self.norm_range_box.isChecked():
            left = self.norm_range_start.value()
            right = self.norm_range_end.value()
        self.image_viewer.norm(
            operation=name,
            beta=self.norm_beta.value(),
            left=left,
            right=right,
            background=self.norm_bg_box.isChecked(),
        )

    def search_background_file(self) -> None:
        if self.bg_input_button.text() == "[Select]":
            file, _ = QFileDialog.getOpenFileName(
                caption="Choose Background File",
                directory=str(self.last_file_dir),
            )
            if file and self.image_viewer.load_background_file(Path(file)):
                self.bg_input_button.setText("[Remove]")
                self.norm_bg_box.setEnabled(True)
                self.norm_bg_box.setChecked(True)
        else:
            self.norm_bg_box.setEnabled(False)
            self.norm_bg_box.setChecked(False)
            self.image_viewer.unload_background_file()
            self.bg_input_button.setText("[Select]")

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
                self._lut_file = file
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
            self._lut_file = str(file)

    def load_images_adapter(self, file_path: Optional[Path] = None) -> None:
        self.load_images(Path(file_path) if file_path is not None else None)

    def load_images(self, file_path: Optional[Path] = None) -> None:
        text = f"Loading {'...' if file_path is None else file_path}"
        with LoadingManager(self, text) as lm:
            self.image_viewer.load_data(
                file_path,
                size_ratio=self.size_ratio_spinbox.value(),
                subset_ratio=self.subset_ratio_spinbox.value(),
                max_ram=self.max_ram_spinbox.value(),
                convert_to_8_bit=self.load_8bit_checkbox.isChecked(),
                grayscale=self.load_grayscale_checkbox.isChecked(),
            )
        if file_path is not None:
            data_size_MB = self.image_viewer.data.image.nbytes / 2**20
            log(f"Loaded {data_size_MB:.2f} MB")
            log(f"Available RAM: {get_available_ram():.2f} GB")
            log(f"Seconds needed: {lm.duration:.2f}")
            self.update_statusbar_frame()
            self.reset_options()

    def operation_changed(self) -> None:
        log(f"Available RAM: {get_available_ram():.2f} GB")
        text = self.op_combobox.currentText()
        if text != "-":
            self.norm_subtract_box.setEnabled(False)
            self.norm_divide_box.setEnabled(False)
        else:
            self.norm_subtract_box.setEnabled(True)
            self.norm_divide_box.setEnabled(True)
        with LoadingManager(self, f"Loading {text}...") as lm:
            self.image_viewer.reduce(
                self.image_viewer.AVAILABLE_OPERATIONS[text],
            )
        log(f"Seconds needed: {lm.duration:.2f}")
        log(f"Available RAM: {get_available_ram():.2f} GB")

    def update_roi_settings(self) -> None:
        self.measure_roi.show_in_mm = self.mm_checkbox.isChecked()
        self.measure_roi.n_px = self.pixel_spinbox.value()
        self.measure_roi.px_in_mm = self.mm_spinbox.value()
        if not self.measure_checkbox.isChecked():
            return
        self.measure_roi.update_labels()

    def sync_settings(self) -> None:
        settings.set("viewer/LUT_source", self._lut_file)
        screen_geometry = QApplication.primaryScreen().availableGeometry()
        settings.set(
            "window/relative_size",
            self.width() / screen_geometry.width(),
        )
        settings.set(
            "window/docks",
            self.dock_area.saveState(),
        )
