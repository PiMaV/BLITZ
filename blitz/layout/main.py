import os
import sys
from timeit import default_timer as clock
from typing import Any, Optional

import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QIcon
from PyQt5.QtWidgets import (QApplication, QFileDialog, QHeaderView, QLabel,
                             QMainWindow, QPushButton, QVBoxLayout, QWidget)
from pyqtgraph.dockarea import Dock, DockArea
from pyqtgraph.parametertree import Parameter, ParameterTree

from ..tools import get_available_ram, insert_line_breaks
from .parameters import EDIT_PARAMS, FILE_PARAMS
from .tof import TOFWindow
from .viewer import ImageViewer
from .widgets import DragDropButton, LoadingDialog, LoggingTextEdit

TITLE = (
    "BLITZ V 1.5.2 : Bulk Loading & Interactive Time-series Zonal-analysis "
    "from INP Greifswald"
)


class MainWindow(QMainWindow):

    def __init__(self) -> None:
        super().__init__()

        self.window_ratio = .77

        area = DockArea()
        self.setup_docks(area)
        area.docks

        self.setCentralWidget(area)
        self.setWindowTitle(TITLE)
        self.set_default_geometry()

        self.setup_parameter_tree()
        self.setup_file_info_dock()
        self.setup_image_and_line_viewers()

        script_dir = os.path.dirname(os.path.abspath(__file__))
        icon_path = os.path.join(script_dir, 'BLITZ.ico')
        self.setWindowIcon(QIcon(icon_path))

        self.default_filepath = script_dir
        self.rect_roi_positions = {}
        self.message_count = 0
        self.line_labels = []
        self.angle_labels = []
        self.setup_connections()
        sys.stdout = LoggingTextEdit(self.history)

        self.tof_window: None | TOFWindow = None

        pg.setConfigOptions(useNumba=True)

    def setup_docks(self, area: DockArea) -> None:
        image_viewer_width = int(self.window_ratio * self.width())
        self.common_size = int((self.width() - image_viewer_width) / 2)
        h_line_height = self.common_size*1.1
        h_viewer_height = self.height() - h_line_height

        self.dock_load_data = self.create_dock(
            "Load Data",
            self.common_size,
            h_line_height,
        )
        self.dock_edit_data = self.create_dock(
            "Edit Data",
            self.common_size*1.2,
            h_line_height,
        )
        self.dock_file_metadata = self.create_dock(
            "File Metadata",
            self.common_size,
            int(h_line_height),
        )
        self.dock_v_plot = self.create_dock(
            "V Line",
            self.common_size,
            h_viewer_height - int(h_line_height / 2),
        )
        self.dock_h_plot = self.create_dock(
            "H Line",
            image_viewer_width,
            h_line_height,
        )
        self.dock_image_viewer = self.create_dock(
            "Image Viewer",
            image_viewer_width,
            h_viewer_height,
        )

        area.addDock(self.dock_image_viewer, 'bottom')
        area.addDock(self.dock_load_data, 'left')
        area.addDock(self.dock_h_plot, 'top', self.dock_image_viewer)
        area.addDock(self.dock_edit_data, 'right', self.dock_h_plot)
        area.addDock(self.dock_v_plot, 'bottom', self.dock_load_data)
        area.addDock(self.dock_file_metadata, 'bottom', self.dock_v_plot)

    def create_dock(self, name: str, width: float, height: float) -> Dock:
        dock = Dock(name, size=(width, height))
        dock.hideTitleBar()
        return dock

    def setup_parameter_tree(self) -> None:
        self.paramTreeEdit, self.parameters_edit = self.create_parameter_tree(
            EDIT_PARAMS,
            "edit_params",
        )
        self.paramTreeEdit.setHeaderLabels(['Edit Data', 'Options'])
        layout = QVBoxLayout()
        layout.addWidget(self.paramTreeEdit)
        self.reset_button = QPushButton("Reset View")
        self.reset_button.setFixedSize(int(self.common_size*.9) , 20)
        layout.addWidget(self.reset_button, 0, Qt.AlignmentFlag.AlignCenter)
        container = QWidget()
        container.setLayout(layout)
        self.dock_edit_data.addWidget(container)

        self.paramTreeFile, self.parameters_file = self.create_parameter_tree(
            FILE_PARAMS,
            "file_params",
        )
        self.paramTreeFile.setHeaderLabels(['Loading Data', 'Options'])
        layout = QVBoxLayout()
        layout.addWidget(self.paramTreeFile)
        self.browse_button = DragDropButton(
            "Click to browse or\nDrag and Drop file here"
        )
        self.browse_button.setFixedSize(int(self.common_size*.9) , 50)
        layout.addStretch(1)
        layout.addWidget(self.browse_button, 0, Qt.AlignmentFlag.AlignCenter)
        layout.addStretch(1)
        container = QWidget()
        container.setLayout(layout)
        self.dock_load_data.addWidget(container)

    def create_parameter_tree(
        self,
        params: list[dict[str, Any]],
        name: str,
    ) -> tuple[ParameterTree, Parameter]:
        tree = ParameterTree()
        param = Parameter.create(name=name, type='group', children=params)
        tree.setParameters(param, showTop=False)
        header = tree.header()
        header.setStretchLastSection(False)
        header.setSectionResizeMode(0, QHeaderView.Interactive)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        header.resizeSection(0, 150)
        return tree, param

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
        self.dock_file_metadata.addWidget(file_info_widget)

    def setup_image_and_line_viewers(self) -> None:
        v_plot_viewbox = pg.ViewBox()
        v_plot_viewbox.invertX(True)
        v_plot_viewbox.invertY(True)
        v_plot_item = pg.PlotItem(viewBox=v_plot_viewbox)
        v_plot_item.showGrid(x=True, y=True, alpha=0.4)
        self.v_plot = pg.PlotWidget(plotItem=v_plot_item)
        self.dock_v_plot.addWidget(self.v_plot)

        h_plot_viewbox = pg.ViewBox()
        h_plot_viewbox.invertX(True)
        h_plot_viewbox.invertY(True)
        h_plot_item = pg.PlotItem(viewBox=h_plot_viewbox)
        h_plot_item.showGrid(x=True, y=True, alpha=0.4)
        self.h_plot = pg.PlotWidget(plotItem=h_plot_item)
        self.dock_h_plot.addWidget(self.h_plot)

        self.image_viewer = ImageViewer(
            h_plot=self.h_plot,
            v_plot=self.v_plot,
            info_label=self.info_label,
            common_size=self.common_size,
        )
        self.dock_image_viewer.addWidget(self.image_viewer)

        self.v_plot.setYLink(self.image_viewer.getView())
        self.h_plot.setXLink(self.image_viewer.getView())

    def set_default_geometry(self, size_factor: float = 0.85) -> None:
        screen_geometry = QApplication.primaryScreen().availableGeometry()
        width = int(screen_geometry.width() * size_factor)
        height = int(screen_geometry.height() * size_factor)
        self.setGeometry(
            (screen_geometry.width() - width) // 2,
            (screen_geometry.height() - height) // 2,
            width,
            height,
        )

    def setup_connections(self) -> None:
        self.browse_button.clicked.connect(self.browse_file)
        self.browse_button.file_dropped.connect(self.load_from_filepath)
        self.reset_button.clicked.connect(self.reset_view)

        self.image_viewer.roi_viewer.sigRegionChanged.connect(
            self.update_line_labels_and_angles
        )

        self.parameters_file.param(
            'Load TOF Data', 'Browse'
        ).sigActivated.connect(self.browse_tof)
        self.parameters_file.param(
            'Load TOF Data', 'Show'
        ).sigValueChanged.connect(self.show_tof)

        self.parameters_edit.param(
            'Calculations', 'Operation'
        ).sigValueChanged.connect(self.on_operation_changed)
        self.parameters_edit.param(
            'Manipulations', 'Rotate CCW'
        ).sigValueChanged.connect(self.checkbox_changed)
        self.parameters_edit.param(
            'Manipulations', 'Flip X'
        ).sigValueChanged.connect(self.checkbox_changed)
        self.parameters_edit.param(
            'Manipulations', 'Flip Y'
        ).sigValueChanged.connect(self.checkbox_changed)
        self.parameters_edit.param(
            'Mask', 'Mask'
        ).sigValueChanged.connect(self.image_viewer.toggle_mask)
        self.parameters_edit.param(
            'Mask','Apply Mask'
        ).sigActivated.connect(self.apply_mask)
        self.parameters_edit.param(
            'ROI','Enable'
        ).sigValueChanged.connect(self.image_viewer.roi_viewer.toggle)
        self.parameters_edit.param(
            'ROI','show in mm'
        ).sigValueChanged.connect(self.update_line_labels_and_angles)
        self.parameters_edit.param(
            'ROI','Pixels'
        ).sigValueChanged.connect(self.update_line_labels_and_angles)
        self.parameters_edit.param(
            'Visualisations', 'Crosshair'
        ).sigValueChanged.connect(self.image_viewer.toggle_crosshair)

        self.parameters_file.param(
            'Loading Pars', 'max. RAM (GB)'
        ).setLimits((0.1, .8*get_available_ram()))

    def browse_tof(self) -> None:
        path, _ = QFileDialog.getOpenFileName()
        if not path:
            return
        self.parameters_file.param(
            'Load TOF Data', 'Show'
        ).setOpts(enabled=True, value=True)
        if self.tof_window is None:
            self.tof_window = TOFWindow(
                self, path, self.image_viewer.data.meta,
            )
        else:
            self.tof_window.update_plot(path)
            self.tof_window.show()

    def show_tof(self) -> None:
        show = self.parameters_file.param(
            'Load TOF Data', 'Show'
        ).value()
        if self.tof_window is not None and show:
            self.tof_window.show()
        elif self.tof_window is not None and not show:
            self.tof_window.hide()

    def browse_file(self) -> None:
        if not hasattr(self, 'filepath') or not self.filepath:
            initial_directory = self.default_filepath
        else:
            initial_directory = os.path.dirname(self.filepath)

        self.filepath, _ = QFileDialog.getOpenFileName(
            directory=initial_directory
        )
        self.load_from_filepath()

    def load_from_filepath(self, filepath: Optional[str] = None) -> None:
        self.filepath = filepath if filepath is not None else self.filepath
        dirname = os.path.dirname(self.filepath)  # type: ignore
        string_temp = insert_line_breaks(dirname)
        print(f"Loading file: {string_temp}")
        self.browse_button.setText(string_temp)
        self.load_data(filepath=self.filepath)

    def load_data(self, filepath: Optional[str] = None) -> None:
        start_time = clock()
        dialog = self.show_loading_dialog()
        size = self.parameters_file.param(
            'Loading Pars', 'Size ratio'
        ).value()
        ratio = self.parameters_file.param(
            'Loading Pars', 'Subset ratio'
        ).value()
        convert_to_8_bit = self.parameters_file.param(
            'Loading Pars', 'Load as 8 bit'
        ).value()
        ram_size = self.parameters_file.param(
            'Loading Pars', 'max. RAM (GB)'
        ).value()

        self.image_viewer.load_data(
            filepath if filepath else None,
            size=size,
            ratio=ratio,
            convert_to_8_bit=convert_to_8_bit,
            ram_size=ram_size,
        )

        print(f"Available RAM: {get_available_ram():.2f} GB")
        self.parameters_edit.param(
            'Mask', 'Mask'
        ).setOpts(enabled=True, value=False)
        self.filepath = filepath
        self.close_loading_dialog(dialog)
        print(f"Time taken to load_data: {clock() - start_time:.2f} seconds")

    def on_operation_changed(self) -> None:
        operation = self.parameters_edit.param(
            'Calculations', 'Operation'
        ).value()

        start_time = clock()
        dialog = self.show_loading_dialog(message="Calculating statistics...")
        print(f"Available RAM: {get_available_ram():.2f} GB")

        self.image_viewer.manipulation(operation)

        self.close_loading_dialog(dialog)
        print(f"Time taken to calculate: {clock() - start_time:.2f} seconds")
        print(f"Available RAM: {get_available_ram():.2f} GB")

    def checkbox_changed(self) -> None:
        if self.parameters_edit.param('Manipulations', 'Rotate CCW').value():
            self.image_viewer.manipulation("rotate")
        if self.parameters_edit.param('Manipulations', 'Flip X').value():
            self.image_viewer.manipulation("flip_x")
        if self.parameters_edit.param('Manipulations', 'Flip Y').value():
            self.image_viewer.manipulation("flip_y")

    def apply_mask(self) -> None:
        self.image_viewer.apply_mask()
        self.parameters_edit.param('Calculations', 'Operation').setValue('Org')

    def reset_view(self) -> None:
        self.image_viewer.reset()
        self.parameters_edit.param(
            'Manipulations', 'Rotate CCW'
        ).setValue(False)
        self.parameters_edit.param('Manipulations', 'Flip X').setValue(False)
        self.parameters_edit.param('Manipulations', 'Flip Y').setValue(False)
        self.parameters_edit.param('Mask', 'Mask').setValue(False)

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

    def update_line_labels_and_angles(self) -> None:
        self.image_viewer.roi_viewer.update_line_labels_and_angles(
            show_in_mm=self.parameters_edit.param('ROI', 'show in mm').value(),
            pixels=float(self.parameters_edit.param('ROI', 'Pixels').value()),
            in_mm=float(self.parameters_edit.param('ROI', 'in mm').value()),
        )
