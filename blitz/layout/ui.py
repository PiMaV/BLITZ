import pyqtgraph as pg
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QIcon
from PyQt5.QtWidgets import (QApplication, QCheckBox, QComboBox,
                             QDoubleSpinBox, QFrame, QGridLayout, QHBoxLayout,
                             QLabel, QLayout, QLineEdit, QMenu, QMenuBar,
                             QPushButton, QScrollArea, QSpinBox, QStatusBar,
                             QStyle, QTabWidget, QVBoxLayout, QWidget)
from pyqtgraph.dockarea import Dock, DockArea

from .. import __version__, resources, settings
from ..data.ops import ReduceOperation
from ..tools import LoggingTextEdit, get_available_ram, setup_logger
from .tof import TOFAdapter
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

        self.measure_roi = MeasureROI(self.image_viewer)

        # create a new timeline replacing roiPlot
        self.norm_range = pg.LinearRegionItem()
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

        self.v_plot.setYLink(self.image_viewer.getView())
        self.h_plot.setXLink(self.image_viewer.getView())

    def setup_menu_and_status_bar(self) -> None:
        self.menubar = QMenuBar()

        file_menu = QMenu("File", self)
        self.action_open = file_menu.addAction("Open...")
        self.action_load = file_menu.addAction("Load TOF")
        self.action_export = file_menu.addAction("Export")
        file_menu.addSeparator()
        self.action_write_ini = file_menu.addAction("Write .ini")
        self.action_select_ini = file_menu.addAction("Select .ini")
        file_menu.addSeparator()
        self.action_restart = file_menu.addAction("Restart")
        self.menubar.addMenu(file_menu)

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

    def setup_lut_dock(self) -> None:
        self._lut_file: str = ""
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
        self.max_ram_spinbox.setValue(settings.get("data/max_ram"))
        self.max_ram_spinbox.setSingleStep(0.1)
        self.max_ram_spinbox.setPrefix("Max. RAM: ")
        file_layout.addWidget(self.max_ram_spinbox)
        load_btn_lay = QHBoxLayout()
        self.button_open_file = QPushButton("Open File")
        load_btn_lay.addWidget(self.button_open_file)
        self.button_open_folder = QPushButton("Open Folder")
        load_btn_lay.addWidget(self.button_open_folder)
        file_layout.addLayout(load_btn_lay)
        connect_label = QLabel("Web Connection")
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
        mask_holay = QHBoxLayout()
        self.checkbox_mask = QCheckBox("Show")
        mask_holay.addWidget(self.checkbox_mask)
        self.button_apply_mask = QPushButton("Apply")
        mask_holay.addWidget(self.button_apply_mask)
        self.button_reset_mask = QPushButton("Reset")
        mask_holay.addWidget(self.button_reset_mask)
        view_layout.addLayout(mask_holay)
        crosshair_label = QLabel("Crosshair")
        crosshair_label.setStyleSheet(style_heading)
        view_layout.addWidget(crosshair_label)
        self.checkbox_crosshair = QCheckBox("Show")
        self.checkbox_crosshair.setChecked(True)
        view_layout.addWidget(self.checkbox_crosshair)
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
        self.checkbox_roi.setChecked(True)
        self.checkbox_tof = QCheckBox("TOF")
        # NOTE: the order of connection here matters
        # roiClicked shows the plot again
        self.checkbox_tof.setEnabled(False)
        checkbox_layout = QHBoxLayout()
        checkbox_layout.addStretch()
        checkbox_layout.addWidget(self.checkbox_roi)
        checkbox_layout.addStretch()
        checkbox_layout.addWidget(self.checkbox_tof)
        checkbox_layout.addStretch()
        view_layout.addLayout(checkbox_layout)
        self.checkbox_roi_drop = QCheckBox("Update ROI only on Drop")
        view_layout.addWidget(self.checkbox_roi_drop)
        view_layout.addStretch()
        self.create_option_tab(view_layout, "View")

        # --- Timeline Operation ---
        timeop_layout = QVBoxLayout()
        timeline_label = QLabel("Reduction")
        timeline_label.setStyleSheet(style_heading)
        timeop_layout.addWidget(timeline_label)
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
        self.checkbox_mm = QCheckBox("Display in mm")
        tools_layout.addWidget(self.checkbox_mm)
        self.spinbox_pixel = QSpinBox()
        self.spinbox_pixel.setPrefix("Pixels: ")
        self.spinbox_pixel.setMinimum(1)
        self.spinbox_pixel.setMaximum(1000)
        converter_layout = QHBoxLayout()
        converter_layout.addWidget(self.spinbox_pixel)
        self.spinbox_mm = QDoubleSpinBox()
        self.spinbox_mm.setPrefix("in mm: ")
        self.spinbox_mm.setMinimum(1.0)
        self.spinbox_mm.setMaximum(100_000.0)
        self.spinbox_mm.setValue(1.0)
        converter_layout.addWidget(self.spinbox_mm)
        tools_layout.addLayout(converter_layout)

        tools_layout.addStretch()
        self.create_option_tab(tools_layout, "Tools")
