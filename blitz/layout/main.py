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

from ..data.image import ImageData
from ..data.load import from_file
from ..tools import get_available_ram, insert_line_breaks
from .parameters import EDIT_PARAMS, FILE_PARAMS
from .tof import TOFWindow
from .viewer import ImageViewer
from .widgets import (DragDropButton, DraggableCrosshair, LoadingDialog,
                      LoggingTextEdit)

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

        self.crosshair = DraggableCrosshair(self)
        self.image_viewer.roi.setPen(self.crosshair.pen)
        self.image_viewer.ui.roiPlot.setFixedHeight(self.common_size)

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
        self.add_line_labels()
        self.toggle_roi()

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
        self.dock_v_line = self.create_dock(
            "V Line",
            self.common_size,
            h_viewer_height - int(h_line_height / 2),
        )
        self.dock_h_line = self.create_dock(
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
        area.addDock(self.dock_h_line, 'top', self.dock_image_viewer)
        area.addDock(self.dock_edit_data, 'right', self.dock_h_line)
        area.addDock(self.dock_v_line, 'bottom', self.dock_load_data)
        area.addDock(self.dock_file_metadata, 'bottom', self.dock_v_line)

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
        v_line_viewbox = pg.ViewBox()
        v_line_viewbox.invertX(True)
        v_line_viewbox.invertY(True)
        v_line_item = pg.PlotItem(viewBox=v_line_viewbox)
        v_line_item.showGrid(x=True, y=True, alpha=0.4)
        self.v_line = pg.PlotWidget(plotItem=v_line_item)
        self.dock_v_line.addWidget(self.v_line)

        h_line_viewbox = pg.ViewBox()
        h_line_viewbox.invertX(True)
        h_line_viewbox.invertY(True)
        h_line_item = pg.PlotItem(viewBox=h_line_viewbox)
        h_line_item.showGrid(x=True, y=True, alpha=0.4)
        self.h_line = pg.PlotWidget(plotItem=h_line_item)
        self.dock_h_line.addWidget(self.h_line)

        self.data = ImageData()
        self.image_viewer = ImageViewer(self.data, self.info_label)
        self.dock_image_viewer.addWidget(self.image_viewer)

        self.v_line.setYLink(self.image_viewer.getView())
        self.h_line.setXLink(self.image_viewer.getView())

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
        self.image_viewer.timeLine.sigPositionChanged.connect(
            self.crosshair.update_plots
        )
        self.reset_button.clicked.connect(self.reset_view)
        self.image_viewer.poly_roi.sigRegionChanged.connect(
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
        ).sigValueChanged.connect(self.toggle_roi)
        self.parameters_edit.param(
            'ROI','show in mm'
        ).sigValueChanged.connect(self.update_line_labels_and_angles)
        self.parameters_edit.param(
            'ROI','Pixels'
        ).sigValueChanged.connect(self.update_line_labels_and_angles)
        self.parameters_edit.param(
            'Visualisations', 'Crosshair'
        ).sigValueChanged.connect(self.crosshair.toggle_crosshair)

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
            self.tof_window = TOFWindow(self, path, self.data.meta)
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
        if filepath:
            self.parameters_edit.param(
                'Calculations', 'Operation'
            ).setValue('Org')
            data, metadata = from_file(
                filepath,
                size=self.parameters_file.param(
                    'Loading Pars', 'Size ratio'
                ).value(),
                ratio=self.parameters_file.param(
                    'Loading Pars', 'Subset ratio'
                ).value(),
                convert_to_8_bit=self.parameters_file.param(
                    'Loading Pars', 'Load as 8 bit'
                ).value(),
                ram_size=self.parameters_file.param(
                    'Loading Pars', 'max. RAM (GB)'
                ).value(),
            )
        else:
            # If filepath is not provided, load random data
            data, metadata = from_file()

        self.data.set(data, metadata)

        print(f"Available RAM: {get_available_ram():.2f} GB")
        self.parameters_edit.param(
            'Mask', 'Mask'
        ).setOpts(enabled=True, value=False)

        self.init_roi_and_crosshair()
        self.filepath = filepath
        self.close_loading_dialog(dialog)
        print(f"Time taken to load_data: {clock() - start_time:.2f} seconds")

        self.image_viewer.show_image(self.data.image)
        self.image_viewer.ui.roiPlot.plotItem.vb.autoRange()

    def on_operation_changed(self) -> None:
        operation = self.parameters_edit.param(
            'Calculations', 'Operation'
        ).value()

        start_time = clock()
        dialog = self.show_loading_dialog(message="Calculating statistics...")
        print(f"Available RAM: {get_available_ram():.2f} GB")

        if operation == 'Min':
            self.image_viewer.setImage(self.data.min)
        elif operation == 'Max':
            self.image_viewer.setImage(self.data.max)
        elif operation == 'Mean':
            self.image_viewer.setImage(self.data.mean)
        elif operation == 'STD':
            self.image_viewer.setImage(self.data.std)
        elif operation == 'Org':
            self.image_viewer.setImage(self.data.image)
        else:
            print("Operation not implemented")

        self.close_loading_dialog(dialog)
        print(f"Time taken to calculate: {clock() - start_time:.2f} seconds")
        print(f"Available RAM: {get_available_ram():.2f} GB")

    def checkbox_changed(self) -> None:
        rotate = self.parameters_edit.param(
            'Manipulations', 'Rotate CCW'
        ).value()
        flip_x = self.parameters_edit.param('Manipulations', 'Flip X').value()
        flip_y = self.parameters_edit.param('Manipulations', 'Flip Y').value()

        self.image_viewer.update_data(rotate, flip_x, flip_y)

    def apply_mask(self) -> None:
        self.image_viewer.apply_mask()
        self.init_roi_and_crosshair()
        self.parameters_edit.param('Calculations', 'Operation').setValue('Org')

    def reset_view(self) -> None:
        self.image_viewer.reset()
        self.parameters_edit.param(
            'Manipulations', 'Rotate CCW'
        ).setValue(False)
        self.parameters_edit.param('Manipulations', 'Flip X').setValue(False)
        self.parameters_edit.param('Manipulations', 'Flip Y').setValue(False)
        self.parameters_edit.param('Mask', 'Mask').setValue(False)
        self.init_roi_and_crosshair()

    def init_roi_and_crosshair(self) -> None:
        height = self.data.image.shape[2]
        width = self.data.image.shape[1]
        percentage = 0.02
        roi_width = max(int(percentage * width),1)
        roi_height = max(int(percentage * height),1)
        x_pos = (width - roi_width) / 2
        y_pos = (height - roi_height) / 2
        self.image_viewer.roi.setPos([x_pos, y_pos])
        self.image_viewer.roi.setSize([roi_width, roi_height])
        self.crosshair.v_line_itself.setPos(width / 2)
        self.crosshair.h_line_itself.setPos(height / 2)

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

    def toggle_roi(self) -> None:
        is_checked = self.parameters_edit.param('ROI', 'Enable').value()
        self.image_viewer.poly_roi.setVisible(is_checked)
        for label in self.line_labels:
            label.setVisible(is_checked)
        for label in self.angle_labels:
            label.setVisible(is_checked)
        if is_checked:
            self.update_line_labels_and_angles()

    def convert_to_view_coords(self, point) -> pg.ViewBox:
        return self.image_viewer.poly_roi.mapToView(point)  # type: ignore

    def midpoint(self, p1, p2) -> tuple[float, float]:
        view_coords_p1 = self.convert_to_view_coords(p1)
        view_coords_p2 = self.convert_to_view_coords(p2)
        return (
            (view_coords_p1.x() + view_coords_p2.x()) / 2,
            (view_coords_p1.y() + view_coords_p2.y()) / 2,
        )

    def add_line_labels(self) -> None:
        self.line_labels = []
        points = self.image_viewer.poly_roi.getHandles()
        for i in range(len(points)):
            self.create_label(i, points)

    def create_label(self, i: int, points: list) -> None:
        p1 = points[i].pos()
        p2 = points[(i + 1) % len(points)].pos()
        mid = self.midpoint(p1, p2)

        # Calculate the length of the line segment
        length = np.sqrt((p2.x() - p1.x())**2 + (p2.y() - p1.y())**2)

        # Set the length as the text of the label
        label = pg.TextItem("{:.2f}".format(length))

        label.setPos(mid[0], mid[1])
        self.image_viewer.getView().addItem(label)
        self.line_labels.append(label)

    def angle_between_lines(self, p0, p1, p2) -> float:
        angle1 = np.arctan2(p1.y() - p0.y(), p1.x() - p0.x())
        angle2 = np.arctan2(p2.y() - p0.y(), p2.x() - p0.x())
        angle = np.degrees(angle1 - angle2)
        angle = abs(angle) % 360
        if angle > 180:
            angle = 360 - angle
        return angle

    def update_line_labels_and_angles(self) -> None:
        points = self.image_viewer.poly_roi.getHandles()
        n = len(points)

        if not hasattr(self, 'angle_labels'):
            self.angle_labels = []
        if not hasattr(self, 'angles'):
            self.angles = []

        for i in range(len(points)):
            p1 = points[i].pos()
            p2 = points[(i + 1) % len(points)].pos()
            mid = self.midpoint(p1, p2)

            # Recalculate the length of the line segment
            length = np.sqrt((p2.x() - p1.x())**2 + (p2.y() - p1.y())**2)
            is_checked = self.parameters_edit.param('ROI', 'show in mm').value()
            if is_checked:
                length = self.convert_to_mm(length)
            else:
                length = length

            if i < len(self.line_labels):
                self.line_labels[i].setPos(mid[0], mid[1])
                self.line_labels[i].setText("{:.2f}".format(length))
            else:
                self.create_label(i, points)

        while len(self.line_labels) > len(points):
            label = self.line_labels.pop()
            self.image_viewer.getView().removeItem(label)

        for i in range(n):
            p0 = points[i].pos()
            p1 = points[(i - 1) % n].pos()
            p2 = points[(i + 1) % n].pos()

            angle = self.angle_between_lines(p0, p1, p2)

            if i < len(self.angle_labels):
                self.angle_labels[i].setPos(
                    self.convert_to_view_coords(p0).x(),
                    self.convert_to_view_coords(p0).y(),
                )
                self.angle_labels[i].setText(f"{angle:.2f}°")
            else:
                angle_label = pg.TextItem(f"{angle:.2f}°")
                angle_label.setPos(p0.x(), p0.y())
                self.image_viewer.getView().addItem(angle_label)
                self.angle_labels.append(angle_label)

            self.angles.append(angle)

        while len(self.angle_labels) > n:
            label = self.angle_labels.pop()
            self.image_viewer.getView().removeItem(label)

    def convert_to_mm(self, value_in_pixels: int) -> float:
        conv_px = float(self.parameters_edit.param('ROI', 'Pixels').value())
        conv_mm = float(self.parameters_edit.param('ROI', 'in mm').value())
        value_in_mm = value_in_pixels * conv_mm / conv_px
        return value_in_mm
