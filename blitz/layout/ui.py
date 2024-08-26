import json

import pyqtgraph as pg
from PyQt5.QtCore import QFile, Qt
from PyQt5.QtGui import QFont, QIcon
from PyQt5.QtWidgets import (QApplication, QCheckBox, QComboBox,
                             QDoubleSpinBox, QFrame, QGridLayout, QHBoxLayout,
                             QLabel, QLayout, QLineEdit, QMenu, QMenuBar,
                             QPushButton, QScrollArea, QSpinBox, QStatusBar,
                             QStyle, QTabWidget, QVBoxLayout, QWidget, QSizePolicy)
from pyqtgraph.dockarea import Dock, DockArea

from .. import __version__, resources, settings
from ..data.ops import ReduceOperation
from ..tools import LoggingTextEdit, get_available_ram, setup_logger
from .viewer import ImageViewer
from .widgets import ExtractionPlot, MeasureROI, TimePlot

TITLE = (
    "BLITZ: (B)ulk (L)oading & (I)nteractive (T)ime series (Z)onal analysis "
    f"- INP Greifswald [{__version__}]"
)


class UI_MainWindow(QWidget):

    def setup_UI(self, form: QWidget) -> None:
        super().__init__()
        form.setWindowTitle(TITLE)
        form.setWindowIcon(QIcon(":/icon/blitz.ico"))

        screen_geometry = QApplication.primaryScreen().availableGeometry()
        relative_size = settings.get("window/relative_size")
        width = int(screen_geometry.width() * relative_size)
        height = int(screen_geometry.height() * relative_size)
        form.setGeometry(
            (screen_geometry.width() - width) // 2,
            (screen_geometry.height() - height) // 2,
            width,
            height,
        )
        if relative_size == 1.0:
            form.showMaximized()

        self.dock_area = DockArea()
        self.setup_docks()
        form.setCentralWidget(self.dock_area)

        self.setup_logger()
        self.setup_image_and_line_viewers()
        self.setup_menu_and_status_bar()
        form.setMenuBar(self.menubar)
        form.setStatusBar(self.statusbar)

        self.setup_lut_dock()
        self.setup_option_dock()
        self.assign_tooltips()

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

    def setup_logger(self) -> None:
        logger = LoggingTextEdit()
        logger.setReadOnly(True)

        file_info_widget = QWidget()
        layout = QVBoxLayout()
        layout.addWidget(logger)
        file_info_widget.setLayout(layout)
        self.dock_status.addWidget(file_info_widget)
        setup_logger(logger)

    def setup_image_and_line_viewers(self) -> None:
        self.image_viewer = ImageViewer()
        self.dock_viewer.addWidget(self.image_viewer)

        self.h_plot = ExtractionPlot(self.image_viewer)
        self.dock_h_plot.addWidget(self.h_plot)
        self.v_plot = ExtractionPlot(self.image_viewer, vertical=True)
        self.dock_v_plot.addWidget(self.v_plot)
        self.v_plot.couple(self.h_plot)
        self.h_plot.couple(self.v_plot)

        self.roi_plot = TimePlot(
            self.dock_t_line,
            self.image_viewer,
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

        self.v_plot.setYLink(self.image_viewer.getView())
        self.h_plot.setXLink(self.image_viewer.getView())

    def setup_menu_and_status_bar(self) -> None:
        self.menubar = QMenuBar()

        file_menu = QMenu("File", self)
        file_menu.setToolTipsVisible(True)
        self.action_open_file = file_menu.addAction("Open File")
        self.action_open_folder = file_menu.addAction("Open Folder")
        self.action_load_tof = file_menu.addAction("Load TOF")
        file_menu.addSeparator()
        self.action_export = file_menu.addAction("Export")
        file_menu.addSeparator()
        self.action_restart = file_menu.addAction("Restart")
        self.menubar.addMenu(file_menu)

        project_menu = QMenu("Project", self)
        project_menu.setToolTipsVisible(True)
        self.action_project_save = project_menu.addAction("Save")
        self.action_project_open = project_menu.addAction("Open")
        self.menubar.addMenu(project_menu)

        about_menu = QMenu("About", self)
        about_menu.setToolTipsVisible(True)
        self.action_link_inp = about_menu.addAction("INP Greifswald")
        self.action_link_github = about_menu.addAction("GitHub")
        self.menubar.addMenu(about_menu)

        font_status = QFont()
        font_status.setPointSize(settings.get("viewer/font_size_status_bar"))

        self.statusbar = QStatusBar()
        self.position_label = QLabel("")
        self.position_label.setFont(font_status)
        self.statusbar.addPermanentWidget(self.position_label)
        self.frame_label = QLabel("")
        self.frame_label.setFont(font_status)
        self.statusbar.addWidget(self.frame_label)
        self.file_label = QLabel("")
        self.file_label.setFont(font_status)
        self.statusbar.addWidget(self.file_label)
        self.loading_label = QLabel("")
        self.loading_label.setFont(font_status)
        self.statusbar.addWidget(self.loading_label)
        self.ram_label = QLabel("")
        self.ram_label.setFont(font_status)
        self.statusbar.addWidget(self.ram_label)

    def setup_lut_dock(self) -> None:
        self.image_viewer.ui.histogram.setParent(None)
        self.dock_lookup.addWidget(self.image_viewer.ui.histogram)
        self.checkbox_autofit = QCheckBox("Auto-Fit")
        self.checkbox_autofit.setChecked(True)
        lut_button_container = QWidget(self)
        self.button_load_lut = QPushButton("Load")
        self.button_export_lut = QPushButton("Export")
        lut_button_layout = QGridLayout()
        lut_button_layout.addWidget(self.checkbox_autofit, 0, 0)
        lut_button_layout.addWidget(self.button_load_lut, 0, 1)
        lut_button_layout.addWidget(self.button_export_lut, 0, 2)
        lut_button_container.setLayout(lut_button_layout)
        self.dock_lookup.addWidget(lut_button_container)

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

    def setup_option_dock(self) -> None:
        self.option_tabwidget = QTabWidget()
        self.dock_option.addWidget(self.option_tabwidget)

        style_heading = """QLabel {
            background-color: rgb(50, 50, 78);
            qproperty-alignment: AlignCenter;
            border-bottom: 5px solid rgb(13, 0, 26);
            font-size: 14pt;
            font: bold;
        }"""

        style_options = (
            "font-size: 9pt;"
        )

        self.option_tabwidget.setStyleSheet(style_options)

        # --- File ---
        file_layout = QVBoxLayout()
        load_label = QLabel("Loading")
        load_label.setStyleSheet(style_heading)
        file_layout.addWidget(load_label)
        load_hlay = QHBoxLayout()
        self.checkbox_load_8bit = QCheckBox("8 bit")
        load_hlay.addWidget(self.checkbox_load_8bit)
        self.checkbox_load_grayscale = QCheckBox("Grayscale")
        self.checkbox_load_grayscale.setChecked(True)
        load_hlay.addWidget(self.checkbox_load_grayscale)
        file_layout.addLayout(load_hlay)
        self.spinbox_load_size = QDoubleSpinBox()
        self.spinbox_load_size.setRange(0, 1)
        self.spinbox_load_size.setValue(1)
        self.spinbox_load_size.setSingleStep(0.1)
        self.spinbox_load_size.setPrefix("Size-ratio: ")
        file_layout.addWidget(self.spinbox_load_size)
        self.spinbox_load_subset = QDoubleSpinBox()
        self.spinbox_load_subset.setRange(0, 1)
        self.spinbox_load_subset.setValue(1)
        self.spinbox_load_subset.setSingleStep(0.1)
        self.spinbox_load_subset.setPrefix("Subset-ratio: ")
        file_layout.addWidget(self.spinbox_load_subset)
        self.spinbox_max_ram = QDoubleSpinBox()
        self.spinbox_max_ram.setRange(.1, .8 * get_available_ram())
        self.spinbox_max_ram.setValue(settings.get("data/max_ram"))
        self.spinbox_max_ram.setSingleStep(0.1)
        self.spinbox_max_ram.setPrefix("Max. RAM: ")
        file_layout.addWidget(self.spinbox_max_ram)
        load_btn_lay = QHBoxLayout()
        self.button_open_file = QPushButton("Open File")
        load_btn_lay.addWidget(self.button_open_file)
        self.button_open_folder = QPushButton("Open Folder")
        load_btn_lay.addWidget(self.button_open_folder)
        file_layout.addLayout(load_btn_lay)
        connect_label = QLabel("Network")
        connect_label.setStyleSheet(style_heading)
        file_layout.addWidget(connect_label)
        address_label = QLabel("Address:")
        self.address_edit = QLineEdit()
        token_label = QLabel("Token:")
        self.token_edit = QLineEdit()
        self.button_connect = QPushButton("Connect")
        self.button_disconnect = QPushButton("Disconnect")
        self.button_disconnect.setEnabled(False)
        connect_lay = QGridLayout()
        connect_lay.addWidget(address_label, 0, 0, 1, 1)
        connect_lay.addWidget(self.address_edit, 0, 1, 1, 1)
        connect_lay.addWidget(token_label, 1, 0, 1, 1)
        connect_lay.addWidget(token_label, 1, 0, 1, 1)
        connect_lay.addWidget(self.token_edit, 1, 1, 1, 1)
        connect_lay.addWidget(self.button_connect, 2, 0, 2, 1)
        connect_lay.addWidget(self.button_disconnect, 2, 1, 2, 1)
        file_layout.addLayout(connect_lay)
        file_layout.addStretch()
        self.create_option_tab(file_layout, "File")

        # --- View ---
        view_layout = QVBoxLayout()
        mask_label = QLabel("View")
        mask_label.setStyleSheet(style_heading)
        view_layout.addWidget(mask_label)
        viewchange_layout = QHBoxLayout()
        self.checkbox_flipx = QCheckBox("Flip x")
        viewchange_layout.addWidget(self.checkbox_flipx)
        self.checkbox_flipy = QCheckBox("Flip y")
        viewchange_layout.addWidget(self.checkbox_flipy)
        self.checkbox_transpose = QCheckBox("Transpose")
        viewchange_layout.addWidget(self.checkbox_transpose)
        view_layout.addLayout(viewchange_layout)
        mask_label = QLabel("Mask")
        mask_label.setStyleSheet(style_heading)
        view_layout.addWidget(mask_label)
        self.checkbox_mask = QCheckBox("Show")
        self.button_apply_mask = QPushButton("Apply")
        self.button_reset_mask = QPushButton("Reset")
        self.button_reset_mask.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding,
        )
        self.button_image_mask = QPushButton("Load binary image")
        mask_holay = QGridLayout()
        mask_holay.addWidget(self.checkbox_mask, 0, 0, 1, 1)
        mask_holay.addWidget(self.button_apply_mask, 0, 1, 1, 1)
        mask_holay.addWidget(self.button_reset_mask, 0, 2, 2, 1)
        mask_holay.addWidget(self.button_image_mask, 1, 0, 1, 2)
        view_layout.addLayout(mask_holay)
        crosshair_label = QLabel("Crosshair")
        crosshair_label.setStyleSheet(style_heading)
        view_layout.addWidget(crosshair_label)
        self.checkbox_crosshair = QCheckBox("Show")
        self.checkbox_crosshair.setChecked(True)
        self.checkbox_crosshair_marking = QCheckBox("Show Markings")
        self.checkbox_crosshair_marking.setChecked(True)
        crosshair_hlay = QHBoxLayout()
        crosshair_hlay.addWidget(self.checkbox_crosshair)
        crosshair_hlay.addWidget(self.checkbox_crosshair_marking)
        view_layout.addLayout(crosshair_hlay)
        width_holay = QHBoxLayout()
        width_label = QLabel("Line width:")
        width_holay.addWidget(width_label)
        self.spinbox_width_h = QSpinBox()
        self.spinbox_width_h.setRange(0, 1)
        self.spinbox_width_h.setValue(0)
        self.spinbox_width_h.setPrefix("H: ")
        width_holay.addWidget(self.spinbox_width_h)
        self.spinbox_width_v = QSpinBox()
        self.spinbox_width_v.setRange(0, 1)
        self.spinbox_width_v.setValue(0)
        self.spinbox_width_v.setPrefix("V: ")
        width_holay.addWidget(self.spinbox_width_v)
        view_layout.addLayout(width_holay)

        roi_label = QLabel("Timeline Plot")
        roi_label.setStyleSheet(style_heading)
        view_layout.addWidget(roi_label)
        self.image_viewer.ui.roiBtn.setParent(None)
        self.image_viewer.ui.roiBtn = QCheckBox("ROI")
        self.checkbox_roi = self.image_viewer.ui.roiBtn
        # NOTE: the order of connection here matters
        # roiClicked shows the plot again
        self.checkbox_roi.setChecked(True)
        self.combobox_roi = QComboBox()
        self.combobox_roi.addItem("Rectangular")
        self.combobox_roi.addItem("Polygon")
        self.combobox_roi.setCurrentIndex(0)
        self.checkbox_tof = QCheckBox("TOF")
        self.checkbox_tof.setEnabled(False)
        self.checkbox_roi_drop = QCheckBox("Update on drop")
        roi_layout = QGridLayout()
        roi_layout.addWidget(self.checkbox_roi, 0, 0, 1, 1)
        roi_layout.addWidget(self.combobox_roi, 0, 1, 1, 2)
        roi_layout.addWidget(self.checkbox_roi_drop, 2, 1, 1, 1)
        roi_layout.addWidget(self.checkbox_tof, 3, 0, 1, 1)
        view_layout.addLayout(roi_layout)
        view_layout.addStretch()
        self.create_option_tab(view_layout, "View")

        # --- Timeline Operation ---
        timeop_layout = QVBoxLayout()
        self.label_crop = QLabel("Timeline Cropping")
        self.label_crop.setStyleSheet(style_heading)
        timeop_layout.addWidget(self.label_crop)
        crop_range_label_to = QLabel("-")
        self.spinbox_crop_range_start = QSpinBox()
        self.spinbox_crop_range_start.setMinimum(0)
        self.spinbox_crop_range_end = QSpinBox()
        self.spinbox_crop_range_end.setMinimum(1)
        self.checkbox_crop_show_range = QCheckBox("")
        map_ = getattr(QStyle, "SP_DesktopIcon")
        self.checkbox_crop_show_range.setIcon(
            self.checkbox_crop_show_range.style().standardIcon(map_)
        )
        self.button_crop = QPushButton("Crop")
        self.button_crop_undo = QPushButton("Undo")
        self.checkbox_crop_keep = QCheckBox("Keep in RAM")
        range_crop_layout = QGridLayout()
        range_crop_layout.addWidget(self.spinbox_crop_range_start, 0, 0, 1, 2)
        range_crop_layout.addWidget(crop_range_label_to, 0, 2, 1, 1)
        range_crop_layout.addWidget(self.spinbox_crop_range_end, 0, 3, 1, 2)
        range_crop_layout.addWidget(self.checkbox_crop_show_range, 0, 5, 1, 2)
        range_crop_layout.addWidget(self.button_crop, 1, 0, 1, 3)
        range_crop_layout.addWidget(self.button_crop_undo, 1, 3, 1, 2)
        range_crop_layout.addWidget(self.checkbox_crop_keep, 1, 5, 1, 2)
        timeop_layout.addLayout(range_crop_layout)

        self.label_reduce = QLabel("Image Aggregation")
        self.label_reduce.setStyleSheet(style_heading)
        timeop_layout.addWidget(self.label_reduce)
        self.combobox_reduce = QComboBox()
        self.combobox_reduce.addItem("-")
        for op in ReduceOperation:
            self.combobox_reduce.addItem(op.name)
        timeop_layout.addWidget(self.combobox_reduce)
        norm_label = QLabel("Normalization")
        norm_label.setStyleSheet(style_heading)
        timeop_layout.addWidget(norm_label)
        self.checkbox_norm_range = QCheckBox("Range:")
        self.checkbox_norm_range.setChecked(True)
        norm_label_op = QLabel("use:")
        self.combobox_norm = QComboBox()
        for op in ReduceOperation:
            self.combobox_norm.addItem(op.name)
        norm_range_label_to = QLabel("-")
        self.spinbox_norm_range_start = QSpinBox()
        self.spinbox_norm_range_start.setMinimum(0)
        self.spinbox_norm_range_end = QSpinBox()
        self.spinbox_norm_range_end.setMinimum(1)
        self.checkbox_norm_show_range = QCheckBox("")
        map_ = getattr(QStyle, "SP_DesktopIcon")
        self.checkbox_norm_show_range.setIcon(
            self.checkbox_norm_show_range.style().standardIcon(map_)
        )
        range_layout = QGridLayout()
        range_layout.addWidget(self.checkbox_norm_range, 0, 0,  1, 1)
        range_layout.addWidget(self.spinbox_norm_range_start, 0, 1, 1, 1)
        range_layout.addWidget(norm_range_label_to, 0, 2, 1, 1)
        range_layout.addWidget(self.spinbox_norm_range_end, 0, 3, 1, 1)
        range_layout.addWidget(self.checkbox_norm_show_range, 0, 4, 1, 1)
        range_layout.addWidget(norm_label_op, 1, 1, 1, 3)
        range_layout.addWidget(self.combobox_norm, 1, 2, 1, 3)
        timeop_layout.addLayout(range_layout)
        timeop_layout.addSpacing(10)
        self.checkbox_norm_bg = QCheckBox("Background:")
        self.checkbox_norm_bg.setEnabled(False)
        self.button_bg_input = QPushButton("[Select]")
        bg_layout = QHBoxLayout()
        bg_layout.addWidget(self.checkbox_norm_bg)
        bg_layout.addWidget(self.button_bg_input)
        timeop_layout.addLayout(bg_layout)
        self.checkbox_norm_lag = QCheckBox("Sliding:")
        self.spinbox_norm_window = QSpinBox()
        self.spinbox_norm_window.setPrefix("Window: ")
        self.spinbox_norm_window.setValue(1)
        self.spinbox_norm_window.setMinimum(1)
        self.spinbox_norm_lag = QSpinBox()
        self.spinbox_norm_lag.setPrefix("Lag: ")
        lag_layout = QHBoxLayout()
        lag_layout.addWidget(self.checkbox_norm_lag)
        lag_layout.addWidget(self.spinbox_norm_window)
        lag_layout.addWidget(self.spinbox_norm_lag)
        timeop_layout.addLayout(lag_layout)
        self.spinbox_norm_beta = QSpinBox()
        self.spinbox_norm_beta.setSuffix("%")
        self.spinbox_norm_beta.setMinimum(0)
        self.spinbox_norm_beta.setMaximum(100)
        self.spinbox_norm_beta.setValue(100)
        self.checkbox_norm_subtract = QCheckBox("Subtract")
        self.checkbox_norm_divide= QCheckBox("Divide")
        norm_layout = QHBoxLayout()
        norm_layout.addWidget(self.spinbox_norm_beta)
        norm_layout.addWidget(self.checkbox_norm_subtract)
        norm_layout.addWidget(self.checkbox_norm_divide)
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
        self.checkbox_measure_roi = QCheckBox("Show")
        tools_layout.addWidget(self.checkbox_measure_roi)
        self.checkbox_mm = QCheckBox("Display in au")
        tools_layout.addWidget(self.checkbox_mm)
        self.spinbox_pixel = QSpinBox()
        self.spinbox_pixel.setPrefix("Pixels: ")
        self.spinbox_pixel.setMinimum(1)
        self.spinbox_pixel.setMaximum(99_999)
        self.spinbox_mm = QDoubleSpinBox()
        self.spinbox_mm.setPrefix("in au: ")
        self.spinbox_mm.setMinimum(0.000001)
        self.spinbox_mm.setMaximum(999_999.0)
        self.spinbox_mm.setValue(1.0)
        self.spinbox_mm.setSingleStep(0.01)
        self.spinbox_mm.setDecimals(5)
        self.textbox_area = QLineEdit("Area:")
        self.textbox_area.setDisabled(True)
        self.textbox_circ = QLineEdit("Circ:")
        self.textbox_circ.setDisabled(True)
        self.textbox_bounding_rect = QLineEdit("HxW:")
        self.textbox_bounding_rect.setDisabled(True)
        self.checkbox_show_bounding_rect = QCheckBox("Show Bounding Rect")
        self.checkbox_show_bounding_rect.setChecked(False)
        tools_layout.addWidget(self.spinbox_pixel)
        tools_layout.addWidget(self.spinbox_mm)
        hline = QFrame()
        hline.setFrameShape(QFrame.Shape.HLine)
        hline.setFrameShadow(QFrame.Shadow.Sunken)
        tools_layout.addWidget(hline)
        tools_layout.addWidget(self.textbox_circ)
        tools_layout.addWidget(self.textbox_area)
        tools_layout.addWidget(self.textbox_bounding_rect)
        tools_layout.addWidget(self.checkbox_show_bounding_rect)
        tools_layout.addStretch()
        self.create_option_tab(tools_layout, "Tools")
        self.measure_roi = MeasureROI(
            self.image_viewer,
            self.textbox_circ,
            self.textbox_area,
            self.textbox_bounding_rect,
        )

        # --- ROSEE ---
        rosee_layout = QVBoxLayout()
        label_rosee = QLabel("RoSEE")
        label_rosee.setStyleSheet(style_heading)
        rosee_layout.addWidget(label_rosee)
        self.checkbox_rosee_local_extrema = QCheckBox("Use local extrema")
        self.checkbox_rosee_local_extrema.setChecked(True)
        rosee_layout.addWidget(self.checkbox_rosee_local_extrema)
        self.spinbox_rosee_smoothing = QSpinBox()
        self.spinbox_rosee_smoothing.setPrefix("Smoothing: ")
        rosee_layout.addWidget(self.spinbox_rosee_smoothing)
        self.checkbox_rosee_normalize = QCheckBox("Normalize values")
        rosee_layout.addWidget(self.checkbox_rosee_normalize)
        hline = QFrame()
        hline.setFrameShape(QFrame.Shape.HLine)
        hline.setFrameShadow(QFrame.Shadow.Sunken)
        rosee_layout.addWidget(hline)
        self.checkbox_rosee_h = QCheckBox("horizontal")
        self.checkbox_rosee_v = QCheckBox("vertical")
        rosee_hlay = QHBoxLayout()
        rosee_hlay.addWidget(self.checkbox_rosee_h)
        rosee_hlay.addWidget(self.checkbox_rosee_v)
        rosee_layout.addLayout(rosee_hlay)
        hline = QFrame()
        hline.setFrameShape(QFrame.Shape.HLine)
        hline.setFrameShadow(QFrame.Shadow.Sunken)
        rosee_layout.addWidget(hline)
        self.textbox_rosee_interval_h = QLineEdit()
        self.textbox_rosee_interval_h.setEnabled(False)
        self.textbox_rosee_interval_v = QLineEdit()
        self.textbox_rosee_interval_v.setEnabled(False)
        self.textbox_rosee_slope_h = QLineEdit("eye: ")
        self.textbox_rosee_slope_h.setEnabled(False)
        self.textbox_rosee_slope_v = QLineEdit("eye: ")
        self.textbox_rosee_slope_v.setEnabled(False)
        self.checkbox_rosee_show_indices = QCheckBox("Show Indices")
        rosee_index_vlayh = QVBoxLayout()
        rosee_index_vlayh.addWidget(self.textbox_rosee_interval_h)
        rosee_index_vlayh.addWidget(self.textbox_rosee_slope_h)
        rosee_index_vlayv = QVBoxLayout()
        rosee_index_vlayv.addWidget(self.textbox_rosee_interval_v)
        rosee_index_vlayv.addWidget(self.textbox_rosee_slope_v)
        rosee_index_hlay = QHBoxLayout()
        rosee_index_hlay.addLayout(rosee_index_vlayh)
        rosee_index_hlay.addLayout(rosee_index_vlayv)
        rosee_layout.addLayout(rosee_index_hlay)
        rosee_layout.addWidget(self.checkbox_rosee_show_indices)
        rosee_layout.addStretch()
        self.create_option_tab(rosee_layout, "RoSEE")

    def assign_tooltips(self) -> None:
        file = QFile(":/docs/tooltips.json")
        file.open(QFile.OpenModeFlag.ReadOnly)
        tips = json.loads(bytes(file.readAll()))

        for widget, tip in tips.items():
            try:
                self.__getattribute__(widget).setStyleSheet(
                    self.__getattribute__(widget).styleSheet()
                    + """QToolTip {
                        border: 2px solid green;
                        padding: 2px;
                        border-radius: 3px;
                        font-size: 12pt;
                        opacity: 200;
                        color: rgb(20, 20, 20);
                        background-color: rgb(200, 200, 200);
                    }"""
                )
            except:
                pass
            self.__getattribute__(widget).setToolTip(tip)
