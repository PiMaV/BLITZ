import os
import sys
from timeit import default_timer as clock
from typing import Any, Optional

import pyqtgraph as pg
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QIcon
from PyQt5.QtWidgets import (QApplication, QFileDialog, QLabel, QMainWindow,
                             QMenu, QMenuBar, QSizePolicy, QVBoxLayout,
                             QWidget)
from pyqtgraph.dockarea import Dock, DockArea
from pyqtgraph.parametertree import Parameter, ParameterTree

from ..tools import get_available_ram, insert_line_breaks
from .parameters import FILE_PARAMS, ROI_PARAMS
from .tof import TOFWindow
from .viewer import ImageViewer
from .widgets import LoadingDialog, LoggingTextEdit

TITLE = (
    "BLITZ V 1.5.2 : Bulk Loading & Interactive Time-series Zonal-analysis "
    "from INP Greifswald"
)


class MainWindow(QMainWindow):

    def __init__(
        self,
        window_ratio: float = .77,
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
        self.border_size = int((1-window_ratio) * self.width() / 2)

        self.dock_area = DockArea()
        self.setup_docks()
        self.setCentralWidget(self.dock_area)

        self.parameters = {}
        self.setup_parameter_trees()
        self.setup_file_info_dock()
        self.setup_image_and_line_viewers()

        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.last_file_dir = script_dir
        icon_path = os.path.join(script_dir, 'BLITZ.ico')
        self.setWindowIcon(QIcon(icon_path))

        self.setup_menubar()
        self.setup_connections()
        sys.stdout = LoggingTextEdit(self.history)

        self.tof_window: None | TOFWindow = None

    def setup_docks(self) -> None:
        viewer_height = self.height() - self.border_size

        self.dock_config = Dock(
            "Configurations",
            size=(self.border_size, self.border_size),
            hideTitle=True,
        )
        self.dock_status = Dock(
            "File Metadata",
            size=(self.border_size, self.border_size),
            hideTitle=True,
        )
        self.dock_v_plot = Dock(
            "V Plot",
            size=(self.border_size, viewer_height - self.border_size),
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

        self.dock_area.addDock(self.dock_viewer, 'bottom')
        self.dock_area.addDock(self.dock_config, 'left')
        self.dock_area.addDock(self.dock_h_plot, 'top', self.dock_viewer)
        self.dock_area.addDock(self.dock_v_plot, 'bottom', self.dock_config)
        self.dock_area.addDock(self.dock_status, 'bottom', self.dock_v_plot)

    def setup_menubar(self) -> None:
        menubar = QMenuBar()

        file_menu = QMenu("File", self)
        file_menu.addAction("Open...").triggered.connect(self.browse_file)
        file_menu.addAction("Load TOF").triggered.connect(self.browse_tof)
        file_menu.addAction("Show Config").triggered.connect(
            lambda: self.switch_parameter_tree("file")
        )
        menubar.addMenu(file_menu)

        op_menu = QMenu("Operations", self)
        op_menu.addAction("Flip Horizontally").triggered.connect(
            lambda: self.image_viewer.manipulation("flip_x")
        )
        op_menu.addAction("Flip Vertically").triggered.connect(
            lambda: self.image_viewer.manipulation("flip_y")
        )
        op_menu.addAction("Transpose").triggered.connect(
            lambda: self.image_viewer.manipulation("transpose")
        )
        all_action = op_menu.addAction("Show All")
        all_action.triggered.connect(
            lambda: self.image_viewer.manipulation("org")
        )
        op_menu.insertSection(all_action, "All")
        op_menu.addAction("Minimum").triggered.connect(
            lambda: self.image_viewer.manipulation("min")
        )
        op_menu.addAction("Maximum").triggered.connect(
            lambda: self.image_viewer.manipulation("max")
        )
        op_menu.addAction("Mean").triggered.connect(
            lambda: self.image_viewer.manipulation("mean")
        )
        op_menu.addAction("Standard Deviation").triggered.connect(
            lambda: self.image_viewer.manipulation("std")
        )
        menubar.addMenu(op_menu)

        mask_menu = QMenu("Mask", self)
        mask_menu.addAction("Show / Hide").triggered.connect(
            self.image_viewer.toggle_mask
        )
        mask_menu.addAction("Apply").triggered.connect(
            self.image_viewer.apply_mask
        )
        mask_menu.addAction("Reset").triggered.connect(
            self.image_viewer.reset
        )
        menubar.addMenu(mask_menu)

        roi_menu = QMenu("ROI", self)
        roi_menu.addAction("Show / Hide").triggered.connect(
            self.image_viewer.measure_roi.toggle
        )
        roi_menu.addAction("Open Config").triggered.connect(
            lambda: self.switch_parameter_tree("roi")
        )
        menubar.addMenu(roi_menu)

        view_menu = QMenu("View", self)
        view_menu.addAction("Show / Hide Crosshair").triggered.connect(
            self.image_viewer.toggle_crosshair
        )
        menubar.addMenu(view_menu)

        self.setMenuBar(menubar)

    def setup_parameter_trees(self) -> None:
        self.create_parameter_tree(FILE_PARAMS, "file")
        self.create_parameter_tree(ROI_PARAMS, "roi")

        title_label = QLabel("BLITZ")
        font = QFont("Courier New", 32)
        font.setBold(True)
        title_label.setFont(font)
        title_label.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout = QVBoxLayout()
        layout.addWidget(title_label)
        self.title_widget = QWidget()
        self.title_widget.setLayout(layout)
        self.dock_config.addWidget(self.title_widget, 0, 0)

    def create_parameter_tree(
        self,
        params: list[dict[str, Any]],
        name: str,
    ) -> None:
        tree = ParameterTree()
        param = Parameter.create(name=name, type='group', children=params)
        tree.setParameters(param, showTop=False)
        layout = QVBoxLayout()
        layout.addWidget(tree)
        container = QWidget()
        container.setLayout(layout)
        self.parameters[name] = (param, container)
        self.dock_config.addWidget(container, 0, 0)
        container.setVisible(False)

    def switch_parameter_tree(self, name: Optional[str] = None) -> None:
        for name_ in self.parameters:
            self.parameters[name_][1].setVisible(False)
        self.title_widget.setVisible(False)
        if name is None:
            self.title_widget.setVisible(True)
        self.parameters[name][1].setVisible(True)

    def setup_file_info_dock(self) -> None:
        file_info_widget = QWidget()
        layout = QVBoxLayout()
        font = QFont()
        font.setPointSize(10)

        self.info_label = QLabel("Info: Not Available")
        self.info_label.setFont(font)
        layout.addWidget(self.info_label)

        self.history = LoggingTextEdit(f"")
        self.history.setReadOnly(True)
        layout.addWidget(self.history)

        # Set height ratios: 1/3 for info_label and 2/3 for history
        layout.setStretch(0, 1)
        layout.setStretch(1, 2)
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
            self.dock_viewer,
            h_plot=self.h_plot,
            v_plot=self.v_plot,
            info_label=self.info_label,
            roi_height=self.border_size,
        )
        self.dock_viewer.addWidget(self.image_viewer)

        self.v_plot.setYLink(self.image_viewer.getView())
        self.h_plot.setXLink(self.image_viewer.getView())

    def setup_connections(self) -> None:
        self.image_viewer.file_dropped.connect(self.load_images)

        self.parameters["roi"][0].param(
            'show in mm'
        ).sigValueChanged.connect(self.update_roi_settings)
        self.parameters["roi"][0].param(
            'Pixels'
        ).sigValueChanged.connect(self.update_roi_settings)
        self.parameters["roi"][0].param(
            'in mm'
        ).sigValueChanged.connect(self.update_roi_settings)

        self.parameters["file"][0].param(
            'max. RAM (GB)'
        ).setLimits((.1, .8 * get_available_ram()))

    def browse_tof(self) -> None:
        path, _ = QFileDialog.getOpenFileName()
        if not path:
            return
        if self.tof_window is None:
            self.tof_window = TOFWindow(
                self, path, self.image_viewer.data.meta,
            )
        else:
            self.tof_window.update_plot(path)
            self.tof_window.show()

    def browse_file(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            directory=self.last_file_dir
        )
        if file_path:
            self.last_filepath = os.path.dirname(file_path)
            string_temp = insert_line_breaks(self.last_filepath)
            print(f"Loading file: {string_temp}")
            self.load_images(file_path)

    def load_images(self, file_path: Optional[str] = None) -> None:
        start_time = clock()
        dialog = self.show_loading_dialog()
        size = self.parameters["file"][0].param('Size ratio').value()
        ratio = self.parameters["file"][0].param('Subset ratio').value()
        convert_to_8_bit = self.parameters["file"][0].param(
            'Load as 8 bit'
        ).value()
        ram_size = self.parameters["file"][0].param('max. RAM (GB)').value()

        self.image_viewer.load_data(
            file_path,
            size=size,
            ratio=ratio,
            convert_to_8_bit=convert_to_8_bit,
            ram_size=ram_size,
        )

        print(f"Available RAM: {get_available_ram():.2f} GB")
        self.close_loading_dialog(dialog)
        print(f"Time taken to load_data: {clock() - start_time:.2f} seconds")

    def on_operation_changed(self) -> None:
        operation = self.parameters["roi"][0].param(
            'Calculations', 'Operation'
        ).value().lower()

        start_time = clock()
        dialog = self.show_loading_dialog(message="Calculating statistics...")
        print(f"Available RAM: {get_available_ram():.2f} GB")

        self.image_viewer.manipulation(operation)

        self.close_loading_dialog(dialog)
        print(f"Time taken to calculate: {clock() - start_time:.2f} seconds")
        print(f"Available RAM: {get_available_ram():.2f} GB")

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
        self.image_viewer.measure_roi.show_in_mm = (
            self.parameters["roi"][0].param('show in mm').value()
        )
        self.image_viewer.measure_roi.n_px = (
            self.parameters["roi"][0].param('Pixels').value()
        )
        self.image_viewer.measure_roi.px_in_mm = (
            self.parameters["roi"][0].param('in mm').value()
        )
        self.image_viewer.measure_roi.update_labels()
