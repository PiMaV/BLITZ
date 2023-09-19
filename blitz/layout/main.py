import os
import sys
from timeit import default_timer as clock
from typing import Any, Optional

import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QIcon, QMouseEvent
from PyQt5.QtWidgets import (QApplication, QFileDialog, QHeaderView, QLabel,
                             QMainWindow, QPushButton, QVBoxLayout, QWidget)
from pyqtgraph import RectROI, mkPen
from pyqtgraph.dockarea import Dock, DockArea
from pyqtgraph.parametertree import Parameter, ParameterTree

from ..data.load import from_file, from_tof
from ..data.tools import smoothen
from ..stats import Stats
from ..tools import (format_pixel_value, get_available_ram, insert_line_breaks,
                     wrap_text)
from .parameters import EDIT_PARAMS, FILE_PARAMS
from .widgets import (DragDropButton, DraggableCrosshair, LoadingDialog,
                      LoggingTextEdit, PlotterWindow)

TITLE = (
    "BLITZ V 1.5.2 : Bulk Loading & Interactive Time-series Zonal-analysis "
    "from INP Greifswald"
)


class MainWindow(QMainWindow):

    def __init__(self) -> None:
        super().__init__()

        area = DockArea()
        self.setCentralWidget(area)
        self.setWindowTitle(TITLE)
        self.set_default_geometry()

        self.window_ratio = .77

        self.crosshair_pen = mkPen(
            color=(200, 200, 200, 140),
            style=Qt.PenStyle.DashDotDotLine,
            width=1,
        )

        self.setup_docks(area)
        self.setup_parameter_tree()
        self.setup_file_info_dock()
        self.setup_image_and_line_viewers()

        script_dir = os.path.dirname(os.path.abspath(__file__))
        icon_path = os.path.join(script_dir, 'BLITZ.ico')
        self.setWindowIcon(QIcon(icon_path))
        self.crosshair = DraggableCrosshair(self)
        self.default_filepath = script_dir
        self.rect_roi_positions = {}
        # self.x_, self.y_ = 0, 0
        pg.setConfigOptions(useNumba=True)
        self.last_rotate = False
        self.last_flip_x = False
        self.last_flip_y = False
        self.stats = Stats()
        self.pixel_value: Optional[np.ndarray] = None
        self.message_count = 0
        self.tof_window = 3
        self.line_labels = []
        self.angle_labels = []
        self.setup_connections()
        sys.stdout = LoggingTextEdit(self.history)
        self.setup_mouse_events()
        self.add_line_labels()
        self.toggle_roi()

    def setup_docks(self, area: DockArea) -> None:
        image_viewer_width = int(self.window_ratio* self.width())
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
        self.paramTreeEdit, self.rootParamEdit = self.create_parameter_tree(
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

        self.paramTreeFile, self.rootParamFile = self.create_parameter_tree(
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

        view = pg.PlotItem()
        view.showGrid(x=True, y=True, alpha=0.4)
        self.imageView = pg.ImageView(view=view)
        self.imageView.ui.graphicsView.setBackground(pg.mkBrush(20, 20, 20))
        self.dock_image_viewer.addWidget(self.imageView)

        self.imageView.ui.roiBtn.setChecked(True)
        self.imageView.roiClicked()
        self.imageView.roi.setPen(self.crosshair_pen)
        self.imageView.ui.histogram.setMinimumWidth(220)
        self.imageView.ui.roiPlot.setFixedHeight(self.common_size)
        self.imageView.ui.roiPlot.plotItem.showGrid(  # type: ignore
            x=True, y=True, alpha=0.6,
        )
        self.imageView.ui.histogram.gradient.loadPreset('greyclip')

        self.poly_roi = pg.PolyLineROI([[0,0], [0,20], [10, 10]], closed=True)
        self.poly_roi.setPen(color=(128, 128, 0, 100),width = 3)
        self.imageView.getView().addItem(self.poly_roi)
        self.v_line.setYLink(self.imageView.getView())
        self.h_line.setXLink(self.imageView.getView())

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
        self.imageView.timeLine.sigPositionChanged.connect(
            self.crosshair.update_plots
        )
        self.reset_button.clicked.connect(self.reset_view)
        self.poly_roi.sigRegionChanged.connect(
            self.update_line_labels_and_angles
        )

        self.rootParamFile.param(
            'Load Additional Data', 'Browse'
        ).sigActivated.connect(self.browse_add_file)

        self.rootParamEdit.param(
            'Calculations', 'Operation'
        ).sigValueChanged.connect(self.on_operation_changed)
        self.rootParamFile.param(
            'Load Additional Data', 'Show'
        ).sigValueChanged.connect(self.plot_additional_data)
        self.rootParamEdit.param(
            'Manipulations', 'Rotate CCW'
        ).sigValueChanged.connect(self.checkbox_changed)
        self.rootParamEdit.param(
            'Manipulations', 'Flip X'
        ).sigValueChanged.connect(self.checkbox_changed)
        self.rootParamEdit.param(
            'Manipulations', 'Flip Y'
        ).sigValueChanged.connect(self.checkbox_changed)
        self.rootParamEdit.param(
            'Mask', 'Mask'
        ).sigValueChanged.connect(self.toggle_mask)
        self.rootParamEdit.param(
            'Mask','Apply Mask'
        ).sigActivated.connect(self.apply_mask)
        self.rootParamEdit.param(
            'ROI','Enable'
        ).sigValueChanged.connect(self.toggle_roi)
        self.rootParamEdit.param(
            'ROI','show in mm'
        ).sigValueChanged.connect(self.update_line_labels_and_angles)
        self.rootParamEdit.param(
            'ROI','Pixels'
        ).sigValueChanged.connect(self.update_line_labels_and_angles)
        self.rootParamEdit.param(
            'Visualisations', 'Crosshair'
        ).sigValueChanged.connect(self.crosshair.toggle_crosshair)

        self.rootParamFile.param(
            'Loading Pars', 'max. RAM (GB)'
        ).setLimits((0.1, .8*get_available_ram()))

    def toggle_mask(self) -> None:
        if self.rootParamEdit.param('Mask', 'Mask').value():
            img = self.imageView.getImageItem().image
            width, height = img.shape[0], img.shape[1]  # type: ignore
            rectRoi = RectROI([0, 0], [width, height], pen=(0,9))
            rectRoi.addScaleHandle([0, 0], [1, 1])
            rectRoi.addScaleHandle([1, 1], [0, 0])
            rectRoi.addScaleHandle([0, 1], [1, 0])
            rectRoi.addScaleHandle([1, 0], [0, 1])
            self.rect_roi_positions['rect_roi'] = rectRoi
            self.imageView.addItem(self.rect_roi_positions['rect_roi'])
        else:
            self.imageView.removeItem(self.rect_roi_positions['rect_roi'])
            self.masked_data = self.data
            self.last_rotate = False
            self.last_flip_x = False
            self.last_flip_y = False
            self.update_data_manipulations()
            self.show_image(self.masked_data, autoRange=True)

    def apply_mask(self) -> None:
        self.mask_data(self.rect_roi_positions['rect_roi'])
        self.imageView.removeItem(self.rect_roi_positions['rect_roi'])
        self.show_image(self.masked_data, autoRange=True)
        self.stats.clear()

    def mask_data(self, roi: pg.ROI) -> None:
        pos = roi.pos()
        size = roi.size()
        data_shape = self.masked_data.shape
        x_start = max(0, int(pos[0]))
        y_start = max(0, int(pos[1]))
        x_end = min(data_shape[1], int(pos[0] + size[0]))
        y_end = min(data_shape[2], int(pos[1] + size[1]))
        self.masked_data = self.masked_data[:, x_start:x_end, y_start:y_end]
        # Mask calculations:
        # self.compute_statistics_and_normalizations(self.masked_data)
        # Update roi and crosshair:
        self.init_roi_and_crosshair()
        # set Operation to Original:
        self.rootParamEdit.param('Calculations', 'Operation').setValue('Org')

    def browse_add_file(self) -> None:
        self.add_filepath, _ = QFileDialog.getOpenFileName()
        if self.add_filepath:
            self.load_add_data()
            self.rootParamFile.param(
                'Load Additional Data', 'Show'
            ).setOpts(enabled=True, value=True)
            self.plot_additional_data()

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
            self.rootParamEdit.param(
                'Calculations', 'Operation'
            ).setValue('Org')
            self.data, self.metadata = from_file(
                filepath,
                size=self.rootParamFile.param(
                    'Loading Pars', 'Size ratio'
                ).value(),
                ratio=self.rootParamFile.param(
                    'Loading Pars', 'Subset ratio'
                ).value(),
                convert_to_8_bit=self.rootParamFile.param(
                    'Loading Pars', 'Load as 8 bit'
                ).value(),
                ram_size=self.rootParamFile.param(
                    'Loading Pars', 'max. RAM (GB)'
                ).value(),
            )
        else:
            # If filepath is not provided, load random data
            self.data, self.metadata = from_file()

        print(f"Available RAM: {get_available_ram():.2f} GB")
        self.masked_data = self.data
        self.rootParamEdit.param(
            'Mask', 'Mask'
        ).setOpts(enabled=True, value=False)

        self.init_roi_and_crosshair()
        self.stats.min_val = None
        self.stats.max_val = None
        self.stats.mean = None
        self.stats.std = None
        self.filepath = filepath
        self.close_loading_dialog(dialog)
        print(f"Time taken to load_data: {clock() - start_time:.2f} seconds")

        self.show_image(
            self.masked_data,
            autoRange=True,
            autoLevels=True,
            autoHistogramRange=True,
        )
        self.imageView.ui.roiPlot.plotItem.vb.autoRange()

    def load_add_data(self) -> None:
        self.add_data = from_tof(self.add_filepath)
        self.sync_tof_to_video()

    def sync_tof_to_video(self) -> None:
        # Calculate the duration of each frame in the ORIGINAL video in ms
        original_frame_duration_ms = 1000 / self.metadata[0]['fps']

        # Calculate the duration of each frame in the RESAMPLED video
        resampled_frame_duration_ms = (
            original_frame_duration_ms
            * self.metadata[0]['frame_count']
            / self.data.shape[0]
        )

        # Generate the times for each frame in the resampled video
        resampled_frame_times = np.arange(
            0,
            self.data.shape[0] * resampled_frame_duration_ms,
            resampled_frame_duration_ms,
        )

        # Create an array for synced data (frame number + sensor values)
        synced_data = np.zeros((self.data.shape[0], self.add_data.shape[1]))

        # Fill the first column with the frame numbers
        synced_data[:, 0] = np.arange(self.data.shape[0])

        # perform linear interpolation for the TOF data onto the
        # resampled video's frame times
        # TODO: This can be one off! Check if the first and last frame
        # times are the same or such
        for col in range(1, self.add_data.shape[1]):
            synced_data[:, col] = np.interp(
                resampled_frame_times,
                self.add_data[:, 0],
                self.add_data[:, col],
            )

        # store the synced data
        self.synced_data = synced_data

        min_tof = np.min(self.synced_data[:, 1])
        max_tof = np.max(self.synced_data[:, 1])

        self.synced_data[:, 1] = 200 - 200 * (
            (self.synced_data[:, 1] - min_tof) / (max_tof - min_tof)
        )

    def compute_statistics_and_normalizations(self, data: np.ndarray) -> None:
        dialog = self.show_loading_dialog(message="Calculating statistics...")
        print(f"Available RAM: {get_available_ram():.2f} GB")
        start_time = clock()

        self.stats.mean = np.mean(data, axis=0)
        self.stats.std = np.std(data, axis=0)
        self.stats.min_val = np.min(data, axis=0)
        self.stats.max_val = np.max(data, axis=0)

        self.close_loading_dialog(dialog)
        print(f"Time taken to calculate: {clock() - start_time:.2f} seconds")
        print(f"Available RAM: {get_available_ram():.2f} GB")

    def on_operation_changed(self) -> None:
        operation = self.rootParamEdit.param(
            'Calculations', 'Operation'
        ).value()

        if self.stats.mean is None or self.stats.std is None:
            self.compute_statistics_and_normalizations(self.masked_data)

        if operation == 'Min':
            result = self.stats.min_val
        elif operation == 'Max':
            result = self.stats.max_val
        elif operation == 'Mean':
            result = self.stats.mean
        elif operation == 'STD':
            result = self.stats.std
        elif operation == 'Org':
            result = self.masked_data
        else:
            result = None
            print("Operation not implemented")

        self.show_image(
            result,
            autoRange=True,
            autoLevels=True,
            autoHistogramRange=True,
        )

    def show_image(
        self,
        data,
        autoRange=False,
        autoLevels=False,
        autoHistogramRange=False,
    ) -> None:
        self.imageView.setImage(
            data,
            autoRange=autoRange,
            autoLevels=autoLevels,
            autoHistogramRange=autoHistogramRange,
        )

    def plot_additional_data(self) -> None:
        show_data_state = self.rootParamFile.param(
            'Load Additional Data', 'Show'
        ).value()
        roi_plot = self.imageView.getRoiPlot()

        if show_data_state:
            self.plot_or_update_roi(roi_plot)
            self.plot_or_update_additional_window()
        else:
            self.remove_roi_plot(roi_plot)
            self.close_additional_window()

    def plot_or_update_roi(self, roi_plot: pg.PlotWidget) -> None:
        x_sync = self.synced_data[:, 0]
        y_sync = self.synced_data[:, 1]
        x_smooth_sync, y_smooth_sync = smoothen(
            x_sync,
            y_sync,
            window_size=self.tof_window,
        )

        # smoothed data for synced data
        if not hasattr(self, 'smoothed_line'):
            self.smoothed_line = roi_plot.plot(
                x_smooth_sync,
                y_smooth_sync,
                pen="pink",
                label=f"Smoothed Synced Data ({self.tof_window})",
                width=2,
            )
        else:
            self.smoothed_line.setData(x_smooth_sync, y_smooth_sync)

    def remove_roi_plot(self, roi_plot: pg.PlotWidget) -> None:
        if hasattr(self, 'smoothed_line'):
            roi_plot.removeItem(self.smoothed_line)
            del self.smoothed_line

    def plot_or_update_additional_window(self) -> None:
        x_data = self.add_data[:, 0]
        y_data = self.add_data[:, 1]
        y_data = y_data.max() - y_data

        if not hasattr(self, 'additional_plot_window'):
            self.additional_plot_window = PlotterWindow(self)
            self.plot_data_in_window(x_data, y_data)
            self.additional_plot_window.show()

    def plot_data_in_window(
        self,
        x_data: np.ndarray,
        y_data: np.ndarray,
    ) -> None:
        self.additional_plot_window.plot_data(
            x_data,
            y_data,
            pen_color="gray",
            label="Raw",
        )
        x_smooth, y_smooth = smoothen(
            x_data,
            y_data,
            window_size=self.tof_window,
        )
        self.additional_plot_window.plot_data(
            x_smooth,
            y_smooth,
            pen_color="green" if self.tof_window == 3 else "red",
            label=f"Smoothed ({self.tof_window})",
        )

    def close_additional_window(self) -> None:
        if hasattr(self, 'additional_plot_window'):
            self.additional_plot_window.window.close()
            del self.additional_plot_window

    def checkbox_changed(self) -> None:
        self.update_data_manipulations()
        self.show_image(self.masked_data)

    def update_data_manipulations(self) -> None:
        # Check the checkboxes for the data manipulations:
        rotate = self.rootParamEdit.param(
            'Manipulations', 'Rotate CCW'
        ).value()
        flip_x = self.rootParamEdit.param('Manipulations', 'Flip X').value()
        flip_y = self.rootParamEdit.param('Manipulations', 'Flip Y').value()

        if rotate != self.last_rotate:
            self.masked_data = np.swapaxes(self.masked_data, 1, 2)
        if flip_x != self.last_flip_x:
            self.masked_data = np.flip(self.masked_data, axis=2)
        if flip_y != self.last_flip_y:
            self.masked_data = np.flip(self.masked_data, axis=1)

        self.last_rotate = rotate
        self.last_flip_x = flip_x
        self.last_flip_y = flip_y

    def reset_view(self) -> None:
        self.masked_data = self.data
        self.imageView.autoRange()
        self.imageView.autoLevels()
        self.imageView.autoHistogramRange()
        self.rootParamEdit.param('Manipulations', 'Rotate CCW').setValue(False)
        self.rootParamEdit.param('Manipulations', 'Flip X').setValue(False)
        self.rootParamEdit.param('Manipulations', 'Flip Y').setValue(False)
        self.rootParamEdit.param('Mask', 'Mask').setValue(False)
        self.last_rotate = False
        self.last_flip_x = False
        self.last_flip_y = False
        self.init_roi_and_crosshair()
        self.stats.clear()
        if self.rect_roi_positions.get('rect_roi'):
            self.imageView.removeItem(self.rect_roi_positions['rect_roi'])

    def setup_mouse_events(self) -> None:
        self.imageView.scene.sigMouseMoved.connect(self.mouseMovedScene)
        self.imageView.timeLine.sigPositionChanged.connect(
            self.mouse_moved_timeline
        )
        self.imageView.ui.roiPlot.mousePressEvent = self.on_roi_plot_clicked

    def mouseMovedScene(self, pos: tuple[float, float]) -> None:
        if not hasattr(self, 'masked_data'):
            return
        img_coords = self.imageView.getView().vb.mapSceneToView(pos)
        self.x_, self.y_ = int(img_coords.x()), int(img_coords.y())

        image_data: np.ndarray = self.imageView.imageItem.image  # type: ignore

        if (0 <= self.x_ < image_data.shape[0]
                and 0 <= self.y_ < image_data.shape[1]):
            self.pixel_value = image_data[self.x_, self.y_]
        else:
            self.pixel_value = None

        self.update_position_label()

    def on_roi_plot_clicked(self, event) -> None:
        if isinstance(event, QMouseEvent):
            if event.button() == Qt.MouseButton.MiddleButton:
                x_pos = self.imageView.ui.roiPlot.plotItem.vb.mapSceneToView(
                    event.pos()
                ).x()
                index = int(x_pos)
                index = max(0, min(index, self.masked_data.shape[0]-1))
                self.imageView.setCurrentIndex(index)
            else:
                pg.PlotWidget.mousePressEvent(self.imageView.ui.roiPlot, event)

    def mouse_moved_timeline(self) -> None:
        if not hasattr(self, 'masked_data'):
            return
        self.update_position_label()

    def init_roi_and_crosshair(self) -> None:
        height = self.masked_data.shape[2]
        width = self.masked_data.shape[1]
        percentage = 0.02
        roi_width = max(int(percentage * width),1)
        roi_height = max(int(percentage * height),1)
        x_pos = (width - roi_width) / 2
        y_pos = (height - roi_height) / 2
        self.imageView.roi.setPos([x_pos, y_pos])
        self.imageView.roi.setSize([roi_width, roi_height])
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

    def update_position_label(self) -> None:
        self.current_image = int(self.imageView.currentIndex)

        if (hasattr(self, 'metadata')
                and 'file_name' in self.metadata[self.current_image]):
            self.current_image_name = self.metadata[self.current_image].get(
                'file_name',
                str(self.current_image),
            )
        else:
            self.current_image_name = str(self.current_image)


        pixel_text = format_pixel_value(self.pixel_value)
        current_image_name_wrapped = wrap_text(self.current_image_name, 40)

        text = (
            f"|X:{self.x_:4d} Y:{self.y_:4d}|\n"
            f"|{pixel_text}|\n"
            f"|Frame:{self.current_image:4d}"
            f"/{self.masked_data.shape[0]-1:4d}|\n"
            f"|Name: {current_image_name_wrapped}|"
        )

        self.info_label.setText(text)

    def toggle_roi(self) -> None:
        is_checked = self.rootParamEdit.param('ROI', 'Enable').value()
        self.poly_roi.setVisible(is_checked)
        for label in self.line_labels:
            label.setVisible(is_checked)
        for label in self.angle_labels:
            label.setVisible(is_checked)
        if is_checked:
            self.update_line_labels_and_angles()

    def convert_to_view_coords(self, point) -> pg.ViewBox:
        return self.poly_roi.mapToView(point)  # type: ignore

    def midpoint(self, p1, p2) -> tuple[float, float]:
        view_coords_p1 = self.convert_to_view_coords(p1)
        view_coords_p2 = self.convert_to_view_coords(p2)
        return (
            (view_coords_p1.x() + view_coords_p2.x()) / 2,
            (view_coords_p1.y() + view_coords_p2.y()) / 2,
        )

    def add_line_labels(self) -> None:
        self.line_labels = []
        points = self.poly_roi.getHandles()
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
        self.imageView.getView().addItem(label)
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
        points = self.poly_roi.getHandles()
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
            is_checked = self.rootParamEdit.param('ROI', 'show in mm').value()
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
            self.imageView.getView().removeItem(label)

        for i in range(n):
            p0 = points[i].pos()
            p1 = points[(i - 1) % n].pos()
            p2 = points[(i + 1) % n].pos()

            angle = self.angle_between_lines(p0, p1, p2)

            if i < len(self.angle_labels):
                self.angle_labels[i].setPos(((self.convert_to_view_coords(p0)).x()), (self.convert_to_view_coords(p0)).y())
                self.angle_labels[i].setText(f"{angle:.2f}°")
            else:
                angle_label = pg.TextItem(f"{angle:.2f}°")
                angle_label.setPos(p0.x(), p0.y())
                self.imageView.getView().addItem(angle_label)
                self.angle_labels.append(angle_label)

            self.angles.append(angle)

        while len(self.angle_labels) > n:
            label = self.angle_labels.pop()
            self.imageView.getView().removeItem(label)

    def convert_to_mm(self, value_in_pixels: int) -> float:
        conv_px = float(self.rootParamEdit.param('ROI', 'Pixels').value())
        conv_mm = float(self.rootParamEdit.param('ROI', 'in mm').value())
        value_in_mm = value_in_pixels * conv_mm / conv_px
        return value_in_mm
