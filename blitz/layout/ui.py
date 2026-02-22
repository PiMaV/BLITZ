import json

import pyqtgraph as pg
from PyQt6.QtCore import QFile, Qt, QTimer
from PyQt6.QtGui import QFont, QIcon
from PyQt6.QtWidgets import (QButtonGroup, QCheckBox, QComboBox, QDoubleSpinBox,
                             QFrame, QGridLayout, QGroupBox, QHBoxLayout,
                             QLabel, QLayout, QLineEdit, QMenu, QMenuBar,
                             QPushButton, QRadioButton, QScrollArea,
                             QSizePolicy, QSlider, QSplitter, QSpinBox,
                             QStatusBar, QTabWidget, QVBoxLayout,
                             QWidget)
from pyqtgraph.dockarea import Dock, DockArea

from .. import __version__, settings
from .. import resources  # noqa: F401  (import registers Qt resources)
from ..data.ops import ReduceOperation
from ..theme import get_style
from ..tools import LoggingTextEdit, get_available_ram, setup_logger
from .bench_sparklines import BenchSparklines
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
        self.timeline_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.timeline_splitter.addWidget(self.roi_plot)
        self.timeline_splitter.setStretchFactor(0, 1)
        timeline_container = QWidget()
        timeline_vbox = QVBoxLayout(timeline_container)
        timeline_vbox.setContentsMargins(0, 0, 0, 0)
        timeline_vbox.setSpacing(0)
        timeline_vbox.addWidget(self.timeline_splitter, 1)
        self.dock_t_line.addWidget(timeline_container)
        self.image_viewer.ui.menuBtn.setParent(None)

        self.v_plot.setYLink(self.image_viewer.getView())
        self.h_plot.setXLink(self.image_viewer.getView())
        # Disable autoRange for unlinked axis to prevent cumulative zoom-out
        # (pyqtgraph bug with linked views: unlinked axis keeps expanding on refresh)
        self.h_plot.getViewBox().enableAutoRange(y=False)
        self.v_plot.getViewBox().enableAutoRange(x=False)

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

        theme_menu = QMenu("Theme", self)
        theme_menu.setToolTipsVisible(True)
        self.action_theme_dark = theme_menu.addAction("Dark (Tokyo Night)")
        self.action_theme_light = theme_menu.addAction("Light (Tokyo Day)")
        theme_menu.setToolTip("Restart to apply")
        _theme = settings.get("app/theme")
        self.action_theme_dark.setCheckable(True)
        self.action_theme_light.setCheckable(True)
        self.action_theme_dark.setChecked(_theme == "dark")
        self.action_theme_light.setChecked(_theme == "light")
        self.menubar.addMenu(theme_menu)

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
        self.ram_label = QLabel("")
        self.ram_label.setFont(font_status)
        self.statusbar.addWidget(self.ram_label)

    def setup_lut_dock(self) -> None:
        self.image_viewer.ui.histogram.setParent(None)
        lut_left = QWidget()
        lut_left_vbox = QVBoxLayout(lut_left)
        lut_left_vbox.setContentsMargins(0, 0, 0, 0)
        lut_left_vbox.setSpacing(0)
        lut_left_vbox.addWidget(self.image_viewer.ui.histogram)
        self.button_autofit = QPushButton("Fit")
        self.checkbox_auto_colormap = QCheckBox("Auto colormap")
        self.checkbox_auto_colormap.setChecked(True)
        lut_button_container = QWidget(self)
        self.button_load_lut = QPushButton("Load")
        self.button_export_lut = QPushButton("Export")
        self.button_load_lut.setVisible(False)
        self.button_export_lut.setVisible(False)
        lut_button_layout = QVBoxLayout()
        lut_button_layout.addWidget(self.button_autofit)
        lut_button_layout.addWidget(self.checkbox_auto_colormap)
        lut_button_layout.addWidget(self.button_load_lut)
        lut_button_layout.addWidget(self.button_export_lut)
        lut_button_container.setLayout(lut_button_layout)
        lut_left_vbox.addWidget(lut_button_container)
        self.blocking_status = QLabel("IDLE")
        self.blocking_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.blocking_status.setMinimumWidth(64)
        self.blocking_status.setMinimumHeight(42)
        self.blocking_status.setFrameShape(QFrame.Shape.StyledPanel)
        self.blocking_status.setFrameShadow(QFrame.Shadow.Sunken)
        self.blocking_status.setStyleSheet(get_style("idle"))
        self.blocking_status.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        lut_splitter = QSplitter(Qt.Orientation.Horizontal)
        lut_splitter.addWidget(lut_left)
        lut_splitter.addWidget(self.blocking_status)
        lut_splitter.setStretchFactor(0, 1)
        lut_splitter.setStretchFactor(1, 0)
        lut_splitter.setSizes([999, 64])
        self.dock_lookup.addWidget(lut_splitter)

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
        self.option_tabwidget.setStyleSheet(
            "QTabWidget::pane { border: 1px solid #3b4261; border-radius: 4px; } "
            "QTabBar::tab:selected { color: #7aa2f7; font-weight: bold; } "
            "font-size: 9pt;"
        )

        # --- File ---
        file_layout = QVBoxLayout()

        self.checkbox_video_dialog_always = QCheckBox("Show load options dialog")
        self.checkbox_video_dialog_always.setChecked(
            bool(settings.get("default/show_load_dialog"))
        )
        self.checkbox_video_dialog_always.setToolTip(
            "When checked: load options dialog opens for every load. When unchecked: last used settings are applied without asking (e.g. for repeated drag-and-drop)."
        )
        self.checkbox_video_dialog_always.setStyleSheet(get_style("toggle_switch"))
        file_layout.addWidget(self.checkbox_video_dialog_always)

        import_group = QGroupBox("Import Settings")
        import_layout = QVBoxLayout()
        import_group.setLayout(import_layout)

        load_hlay = QHBoxLayout()
        self.checkbox_load_8bit = QCheckBox("8 bit")
        load_hlay.addWidget(self.checkbox_load_8bit)
        self.checkbox_load_grayscale = QCheckBox("grayscale")
        self.checkbox_load_grayscale.setChecked(True)
        load_hlay.addWidget(self.checkbox_load_grayscale)
        import_layout.addLayout(load_hlay)
        self.spinbox_load_size = QDoubleSpinBox()
        self.spinbox_load_size.setRange(0, 1)
        self.spinbox_load_size.setValue(1)
        self.spinbox_load_size.setSingleStep(0.1)
        self.spinbox_load_size.setPrefix("size ratio: ")
        import_layout.addWidget(self.spinbox_load_size)
        self.spinbox_load_subset = QDoubleSpinBox()
        self.spinbox_load_subset.setRange(0, 1)
        self.spinbox_load_subset.setValue(1)
        self.spinbox_load_subset.setSingleStep(0.1)
        self.spinbox_load_subset.setPrefix("subset ratio: ")
        import_layout.addWidget(self.spinbox_load_subset)
        self.spinbox_max_ram = QDoubleSpinBox()
        self.spinbox_max_ram.setSingleStep(0.1)
        self.spinbox_max_ram.setPrefix("max. RAM: ")
        self.spinbox_max_ram.setRange(.1, .8 * get_available_ram())
        import_layout.addWidget(self.spinbox_max_ram)
        self.checkbox_sync_file = QCheckBox("load/save project file")
        self.checkbox_sync_file.setChecked(False)
        self.checkbox_sync_file.setStyleSheet(
            f"QCheckBox:checked {{ color: {get_style('color_red')}; }}"
        )
        import_layout.addWidget(self.checkbox_sync_file)
        file_layout.addWidget(import_group)

        self.crop_section_widget = QWidget()
        crop_section_layout = QVBoxLayout(self.crop_section_widget)
        crop_label = QLabel("Crop")
        crop_label.setStyleSheet(get_style("heading"))
        crop_section_layout.addWidget(crop_label)
        self.button_crop = QPushButton("Apply Crop")
        crop_section_layout.addWidget(self.button_crop)
        file_layout.addWidget(self.crop_section_widget)
        self.crop_section_widget.setVisible(False)
        file_layout.addStretch()
        self.create_option_tab(file_layout, "File")

        # --- View ---
        view_layout = QVBoxLayout()
        mask_label = QLabel("View")
        mask_label.setStyleSheet(get_style("heading"))
        view_layout.addWidget(mask_label)
        viewchange_layout = QHBoxLayout()
        self.checkbox_flipx = QCheckBox("Flip x")
        viewchange_layout.addWidget(self.checkbox_flipx)
        self.checkbox_flipy = QCheckBox("Flip y")
        viewchange_layout.addWidget(self.checkbox_flipy)
        self.checkbox_transpose = QCheckBox("Transpose")
        viewchange_layout.addWidget(self.checkbox_transpose)
        view_layout.addLayout(viewchange_layout)
        mask_label = QLabel("Display Mask")
        mask_label.setStyleSheet(get_style("heading"))
        mask_label.setToolTip("Post-load mask: draw ROI or load binary image to exclude regions from display/analysis")
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
        crosshair_label.setStyleSheet(get_style("heading"))
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

        extract_label = QLabel("Extraction plots")
        extract_label.setStyleSheet(get_style("heading_small"))
        view_layout.addWidget(extract_label)
        self.checkbox_minmax_per_image = QCheckBox("Min/Max per image")
        self.checkbox_minmax_per_image.setChecked(False)
        self.checkbox_envelope_per_crosshair = QCheckBox("Envelope per crosshair")
        self.checkbox_envelope_per_crosshair.setChecked(False)
        self.checkbox_envelope_per_dataset = QCheckBox("Envelope per position (dataset)")
        self.checkbox_envelope_per_dataset.setChecked(False)
        self.spinbox_envelope_pct = QSpinBox()
        self.spinbox_envelope_pct.setPrefix("Envelope: ")
        self.spinbox_envelope_pct.setSuffix("%")
        self.spinbox_envelope_pct.setRange(0, 49)
        self.spinbox_envelope_pct.setValue(0)
        self.spinbox_envelope_pct.setSpecialValueText("Min/Max")
        extract_layout = QVBoxLayout()
        extract_layout.addWidget(self.checkbox_minmax_per_image)
        extract_layout.addWidget(self.checkbox_envelope_per_crosshair)
        extract_layout.addWidget(self.checkbox_envelope_per_dataset)
        extract_layout.addWidget(self.spinbox_envelope_pct)
        view_layout.addLayout(extract_layout)
        view_layout.addSpacing(10)

        roi_label = QLabel("Timeline Plot")
        roi_label.setStyleSheet(get_style("heading"))
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

        # --- Ops: Subtract/Divide, Source Aggregate|File, Amount sliders ---
        ops_layout = QVBoxLayout()
        ops_label = QLabel("Ops")
        ops_label.setStyleSheet(get_style("heading"))
        ops_layout.addWidget(ops_label)
        self.button_ops_open_aggregate = QPushButton("Open Aggregate")
        self.button_ops_open_aggregate.setToolTip(
            "Configure range and reduce method in Aggregate tab"
        )
        ops_layout.addWidget(self.button_ops_open_aggregate)
        sub_grp = QGroupBox("1. Subtract")
        sub_lay = QVBoxLayout()
        sub_src_row = QHBoxLayout()
        sub_src_row.addWidget(QLabel("Source:"))
        self.combobox_ops_subtract_src = QComboBox()
        self.combobox_ops_subtract_src.addItem("Off", "off")
        self.combobox_ops_subtract_src.addItem("Aggregate", "aggregate")
        self.combobox_ops_subtract_src.addItem("File", "file")
        sub_src_row.addWidget(self.combobox_ops_subtract_src)
        sub_lay.addLayout(sub_src_row)
        sub_amt_row = QHBoxLayout()
        sub_amt_row.addWidget(QLabel("Amount:"))
        self.slider_ops_subtract = QSlider(Qt.Orientation.Horizontal)
        self.slider_ops_subtract.setRange(0, 100)
        self.slider_ops_subtract.setValue(100)
        self.label_ops_subtract = QLabel("100%")
        sub_amt_row.addWidget(self.slider_ops_subtract)
        sub_amt_row.addWidget(self.label_ops_subtract)
        sub_lay.addLayout(sub_amt_row)
        sub_grp.setLayout(sub_lay)
        ops_layout.addWidget(sub_grp)
        div_grp = QGroupBox("2. Divide")
        div_lay = QVBoxLayout()
        div_src_row = QHBoxLayout()
        div_src_row.addWidget(QLabel("Source:"))
        self.combobox_ops_divide_src = QComboBox()
        self.combobox_ops_divide_src.addItem("Off", "off")
        self.combobox_ops_divide_src.addItem("Aggregate", "aggregate")
        self.combobox_ops_divide_src.addItem("File", "file")
        div_src_row.addWidget(self.combobox_ops_divide_src)
        div_lay.addLayout(div_src_row)
        div_amt_row = QHBoxLayout()
        div_amt_row.addWidget(QLabel("Amount:"))
        self.slider_ops_divide = QSlider(Qt.Orientation.Horizontal)
        self.slider_ops_divide.setRange(0, 100)
        self.slider_ops_divide.setValue(0)
        self.label_ops_divide = QLabel("0%")
        div_amt_row.addWidget(self.slider_ops_divide)
        div_amt_row.addWidget(self.label_ops_divide)
        div_lay.addLayout(div_amt_row)
        div_grp.setLayout(div_lay)
        ops_layout.addWidget(div_grp)
        self.button_ops_load_file = QPushButton("Load reference image")
        self.ops_file_widget = QWidget()
        self.ops_file_widget.setLayout(QHBoxLayout())
        self.ops_file_widget.layout().addWidget(self.button_ops_load_file)
        ops_layout.addWidget(self.ops_file_widget)
        self.ops_file_widget.setVisible(False)
        ops_layout.addStretch()
        self.create_option_tab(ops_layout, "Ops")

        # --- Timeline Panel: 2 Tabs Frame | Agg (Tab-Wechsel = Modus) ---
        self.spinbox_current_frame = QSpinBox()
        self.spinbox_current_frame.setMinimum(0)
        self.spinbox_current_frame.setPrefix("Idx: ")
        self.spinbox_current_frame.setMinimumWidth(72)
        self.spinbox_crop_range_start = QSpinBox()
        self.spinbox_crop_range_start.setMinimum(0)
        self.spinbox_crop_range_start.setMinimumWidth(72)
        self.spinbox_crop_range_end = QSpinBox()
        self.spinbox_crop_range_end.setMinimum(0)
        self.spinbox_crop_range_end.setMinimumWidth(72)
        self.spinbox_selection_window = QSpinBox()
        self.spinbox_selection_window.setPrefix("Win: ")
        self.spinbox_selection_window.setMinimum(1)
        self.spinbox_selection_window.setMinimumWidth(52)
        self.checkbox_window_const = QCheckBox("Win const.")
        self.button_reset_range = QPushButton("Full Range")

        self.combobox_reduce = QComboBox()
        self.combobox_reduce.addItem("None - current frame")
        for op in ReduceOperation:
            self.combobox_reduce.addItem(op.name)

        self.radio_time_series = QRadioButton()
        self.radio_time_series.setChecked(True)
        self.radio_aggregated = QRadioButton()
        self.view_mode_group = QButtonGroup()
        self.view_mode_group.addButton(self.radio_time_series)
        self.view_mode_group.addButton(self.radio_aggregated)

        frame_tab = QWidget()
        frame_layout = QVBoxLayout(frame_tab)
        frame_layout.setContentsMargins(4, 2, 4, 2)
        frame_layout.addWidget(self.spinbox_current_frame)
        self.checkbox_timeline_bands = QCheckBox("Upper/lower band")
        self.checkbox_timeline_bands.setChecked(False)
        self.checkbox_timeline_bands.setToolTip(
            "Show min/max envelope in the timeline plot"
        )
        frame_layout.addWidget(self.checkbox_timeline_bands)
        timeline_agg_row = QHBoxLayout()
        timeline_agg_row.addWidget(QLabel("Curve:"))
        self.combobox_timeline_aggregation = QComboBox()
        self.combobox_timeline_aggregation.addItem("Mean", "mean")
        self.combobox_timeline_aggregation.addItem("Median", "median")
        self.combobox_timeline_aggregation.setToolTip(
            "Aggregation within ROI per frame for the timeline curve"
        )
        timeline_agg_row.addWidget(self.combobox_timeline_aggregation)
        frame_layout.addLayout(timeline_agg_row)
        frame_layout.addStretch()

        self.range_section_widget = QWidget()
        range_layout = QVBoxLayout(self.range_section_widget)
        range_layout.setContentsMargins(0, 0, 0, 0)
        sel_row1 = QHBoxLayout()
        sel_row1.addWidget(QLabel("Start:"))
        sel_row1.addWidget(self.spinbox_crop_range_start)
        sel_row1.addWidget(QLabel("-"))
        sel_row1.addWidget(QLabel("End:"))
        sel_row1.addWidget(self.spinbox_crop_range_end)
        range_layout.addLayout(sel_row1)
        sel_win_row = QHBoxLayout()
        sel_win_row.addWidget(self.spinbox_selection_window)
        sel_win_row.addWidget(self.checkbox_window_const)
        sel_win_row.addWidget(self.button_reset_range)
        range_layout.addLayout(sel_win_row)

        self.checkbox_agg_update_on_drag = QCheckBox("Update on drag")
        self.checkbox_agg_update_on_drag.setChecked(False)
        self.checkbox_agg_update_on_drag.setToolTip(
            "Aggregate updates live while dragging the range (off = update on drop)"
        )
        agg_tab = QWidget()
        agg_layout = QVBoxLayout(agg_tab)
        agg_layout.setContentsMargins(4, 2, 4, 2)
        agg_layout.addWidget(QLabel("Reduce:"))
        agg_layout.addWidget(self.combobox_reduce)
        agg_layout.addWidget(self.range_section_widget)
        agg_layout.addWidget(self.checkbox_agg_update_on_drag)
        agg_layout.addStretch()

        self.timeline_tabwidget = QTabWidget()
        self.timeline_tabwidget.addTab(frame_tab, "Frame")
        self.timeline_tabwidget.addTab(agg_tab, "Aggregate")
        self.timeline_tabwidget.setStyleSheet(
            "QTabBar::tab { min-width: 72px; padding: 6px 12px; font-size: 10pt; } "
            "QTabBar::tab:selected { font-weight: bold; background: #3b4261; "
            "border-radius: 3px 3px 0 0; } "
        )

        self.selection_panel = QWidget()
        sel_layout = QVBoxLayout(self.selection_panel)
        sel_layout.setContentsMargins(0, 0, 0, 0)
        sel_layout.addWidget(self.timeline_tabwidget)
        w = max(800, self.width())
        border_size = int(0.25 * w / 2)
        self.selection_panel.setMinimumWidth(max(120, border_size))
        self.timeline_splitter.addWidget(self.selection_panel)

        def _set_timeline_splitter_sizes():
            try:
                lut_w = self.dock_lookup.width()
            except Exception:
                lut_w = int(0.25 * max(800, self.width()) / 2)
            if lut_w < 100:
                lut_w = int(0.25 * max(800, self.width()) / 2)
            total = self.timeline_splitter.width()
            if total > 100:
                self.timeline_splitter.setSizes([max(200, total - lut_w), lut_w])
            else:
                w = max(800, self.width())
                bw = int(0.25 * w / 2)
                iv = int(0.75 * w)
                self.timeline_splitter.setSizes([max(200, iv - bw), bw])

        QTimer.singleShot(150, _set_timeline_splitter_sizes)

        # --- Tools ---
        tools_layout = QVBoxLayout()
        roi_label = QLabel("Measure Tool")
        roi_label.setStyleSheet(get_style("heading"))
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
        label_rosee.setStyleSheet(get_style("heading"))
        rosee_layout.addWidget(label_rosee)
        self.checkbox_rosee_active = QCheckBox("Show RoSEE")
        self.checkbox_rosee_active.setChecked(False)
        rosee_layout.addWidget(self.checkbox_rosee_active)
        hline = QFrame()
        hline.setFrameShape(QFrame.Shape.HLine)
        hline.setFrameShadow(QFrame.Shadow.Sunken)
        rosee_layout.addWidget(hline)
        self.checkbox_rosee_local_extrema = QCheckBox("Use local extrema")
        self.checkbox_rosee_local_extrema.setChecked(False)
        rosee_layout.addWidget(self.checkbox_rosee_local_extrema)
        self.spinbox_rosee_smoothing = QSpinBox()
        self.spinbox_rosee_smoothing.setPrefix("Smoothing: ")
        rosee_layout.addWidget(self.spinbox_rosee_smoothing)
        self.label_rosee_plots = QLabel("Plots")
        self.label_rosee_plots.setStyleSheet(get_style("heading_small"))
        rosee_layout.addWidget(self.label_rosee_plots)
        self.checkbox_rosee_h = QCheckBox("horizontal")
        self.checkbox_rosee_v = QCheckBox("vertical")
        self.checkbox_rosee_h.setChecked(True)
        self.checkbox_rosee_v.setChecked(True)
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
        hline = QFrame()
        hline.setFrameShape(QFrame.Shape.HLine)
        hline.setFrameShadow(QFrame.Shadow.Sunken)
        rosee_layout.addWidget(hline)
        self.checkbox_rosee_normalize = QCheckBox("Normalize values")
        rosee_layout.addWidget(self.checkbox_rosee_normalize)
        self.checkbox_rosee_show_indices = QCheckBox("Show Indices")
        self.checkbox_rosee_show_lines = QCheckBox("Show Lines")
        rosee_lines_hlay = QHBoxLayout()
        rosee_lines_hlay.addWidget(self.checkbox_rosee_show_indices)
        rosee_lines_hlay.addWidget(self.checkbox_rosee_show_lines)
        rosee_layout.addLayout(rosee_lines_hlay)
        self.label_rosee_image = QLabel("Image")
        self.label_rosee_image.setStyleSheet(get_style("heading_small"))
        rosee_layout.addWidget(self.label_rosee_image)
        rosee_hlay2 = QHBoxLayout()
        self.checkbox_rosee_in_image_h = QCheckBox("horizontal")
        self.checkbox_rosee_in_image_v = QCheckBox("vertical")
        rosee_hlay2.addWidget(self.checkbox_rosee_in_image_h)
        rosee_hlay2.addWidget(self.checkbox_rosee_in_image_v)
        rosee_layout.addLayout(rosee_hlay2)
        self.checkbox_show_isocurve = QCheckBox("Isocurves")
        self.checkbox_show_isocurve.setChecked(False)
        self.spinbox_isocurves = QSpinBox()
        self.spinbox_isocurves.setMinimum(1)
        self.spinbox_isocurves.setValue(1)
        iso_hlay = QHBoxLayout()
        iso_hlay.addWidget(self.checkbox_show_isocurve)
        iso_hlay.addWidget(self.spinbox_isocurves)
        rosee_layout.addLayout(iso_hlay)
        self.spinbox_iso_smoothing = QSpinBox()
        self.spinbox_iso_smoothing.setPrefix("Smoothing: ")
        self.spinbox_iso_smoothing.setMinimum(0)
        self.spinbox_iso_smoothing.setValue(3)
        rosee_layout.addWidget(self.spinbox_iso_smoothing)
        rosee_layout.addStretch()
        self.create_option_tab(rosee_layout, "RoSEE")

        # --- PCA ---
        pca_layout = QVBoxLayout()
        pca_heading = QLabel("PCA")
        pca_heading.setStyleSheet(get_style("heading"))
        pca_layout.addWidget(pca_heading)

        pca_opt_layout = QHBoxLayout()
        self.spinbox_pcacomp_target = QSpinBox()
        self.spinbox_pcacomp_target.setPrefix("Target Comp: ")
        self.spinbox_pcacomp_target.setMinimum(1)
        self.spinbox_pcacomp_target.setMaximum(500)
        self.spinbox_pcacomp_target.setValue(20)
        self.spinbox_pcacomp_target.setToolTip("Number of components to calculate")
        pca_opt_layout.addWidget(self.spinbox_pcacomp_target)

        self.checkbox_pca_exact = QCheckBox("Exact (Slow)")
        self.checkbox_pca_exact.setToolTip("Use full SVD (Exact but slow/memory hungry). Uncheck for Approximate (Randomized) SVD.")
        pca_opt_layout.addWidget(self.checkbox_pca_exact)
        pca_layout.addLayout(pca_opt_layout)

        self.button_pca_calc = QPushButton("Calculate PCA")
        self.button_pca_calc.setToolTip("Compute Principal Component Analysis (SVD). May take time.")
        pca_layout.addWidget(self.button_pca_calc)

        self.label_pca_status = QLabel("Not calculated")
        self.label_pca_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        pca_layout.addWidget(self.label_pca_status)

        hline = QFrame()
        hline.setFrameShape(QFrame.Shape.HLine)
        hline.setFrameShadow(QFrame.Shadow.Sunken)
        pca_layout.addWidget(hline)

        pca_view_layout = QHBoxLayout()
        self.spinbox_pcacomp = QSpinBox()
        self.spinbox_pcacomp.setPrefix("View Comp: ")
        self.spinbox_pcacomp.setMinimum(1)
        self.spinbox_pcacomp.setMaximum(100)
        self.spinbox_pcacomp.setEnabled(False)
        pca_view_layout.addWidget(self.spinbox_pcacomp)

        self.combobox_pca = QComboBox()
        self.combobox_pca.addItem("Reconstruction")
        self.combobox_pca.addItem("Components")
        self.combobox_pca.setEnabled(False)
        pca_view_layout.addWidget(self.combobox_pca)
        pca_layout.addLayout(pca_view_layout)

        self.button_pca_show = QPushButton("Show")
        self.button_pca_show.setCheckable(True)
        self.button_pca_show.setEnabled(False)
        pca_layout.addWidget(self.button_pca_show)

        pca_layout.addStretch()
        self.create_option_tab(pca_layout, "PCA")

        # --- Bench ---
        bench_layout = QVBoxLayout()
        bench_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        bench_label = QLabel("Bench")
        bench_label.setStyleSheet(get_style("heading"))
        bench_label.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        bench_layout.addWidget(bench_label)
        self.bench_sparklines = BenchSparklines()
        bench_layout.addWidget(self.bench_sparklines, 0, Qt.AlignmentFlag.AlignTop)
        self.label_bench_raw = QLabel("Raw matrix: —")
        bench_layout.addWidget(self.label_bench_raw)
        self.label_bench_result = QLabel("Result matrix: —")
        bench_layout.addWidget(self.label_bench_result)
        self.label_bench_mode = QLabel("View mode: —")
        bench_layout.addWidget(self.label_bench_mode)
        self.label_bench_cache = QLabel("Result cache: —")
        bench_layout.addWidget(self.label_bench_cache)
        self.label_bench_live = QLabel("")
        self.label_bench_live.setStyleSheet(
            "color: #9ece6a; font-weight: bold;"
        )
        bench_layout.addWidget(self.label_bench_live)
        bench_layout.addStretch()
        self.create_option_tab(bench_layout, "Bench")
        self.bench_tab_index = self.option_tabwidget.count() - 1

        # --- Stream (sources + network) ---
        stream_layout = QVBoxLayout()
        stream_heading = QLabel("Stream")
        stream_heading.setStyleSheet(get_style("heading"))
        stream_layout.addWidget(stream_heading)
        self.button_mock_live = QPushButton("Cam Mock")
        self.button_mock_live.setToolTip("Simulated camera (Lissajous, Lightning). No real device.")
        stream_layout.addWidget(self.button_mock_live)
        self.button_real_camera = QPushButton("Webcam")
        self.button_real_camera.setToolTip("Real camera (USB webcam)")
        stream_layout.addWidget(self.button_real_camera)
        connect_label = QLabel("Network")
        connect_label.setStyleSheet(get_style("heading"))
        stream_layout.addWidget(connect_label)
        address_label = QLabel("Address:")
        self.address_edit = QLineEdit()
        token_label = QLabel("Token:")
        self.token_edit = QLineEdit()
        self.button_connect = QPushButton("Connect")
        self.button_connect.setStyleSheet(get_style("button_primary"))
        self.button_disconnect = QPushButton("Disconnect")
        self.button_disconnect.setEnabled(False)
        connect_lay = QGridLayout()
        connect_lay.addWidget(address_label, 0, 0, 1, 1)
        connect_lay.addWidget(self.address_edit, 0, 1, 1, 1)
        connect_lay.addWidget(token_label, 1, 0, 1, 1)
        connect_lay.addWidget(self.token_edit, 1, 1, 1, 1)
        connect_lay.addWidget(self.button_connect, 2, 0, 2, 1)
        connect_lay.addWidget(self.button_disconnect, 2, 1, 2, 1)
        stream_layout.addLayout(connect_lay)
        stream_layout.addStretch()
        self.create_option_tab(stream_layout, "Stream")

    def assign_tooltips(self) -> None:
        file = QFile(":/docs/tooltips.json")
        file.open(QFile.OpenModeFlag.ReadOnly)
        tips = json.loads(bytes(file.readAll()))

        for widget, tip in tips.items():
            try:
                w = self.__getattribute__(widget)
                w.setStyleSheet(
                    w.styleSheet()
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
                w.setToolTip(tip)
            except AttributeError:
                pass
