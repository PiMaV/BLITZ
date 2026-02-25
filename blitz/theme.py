"""
Tokyo Night inspired colors for BLITZ GUI.
Supports dark (default) and light theme. Matches boot_bench splash.
"""
_current_theme = "dark"


def set_theme(theme: str) -> None:
    global _current_theme
    _current_theme = "light" if theme == "light" else "dark"


def get_stylesheet() -> str:
    from . import settings as _st
    base = STYLESHEET_LIGHT if _current_theme == "light" else STYLESHEET_FULL
    w = _st.get("viewer/splitter_handle_width")
    return base.replace("__SPLITTER_W__", str(w))


def get_style(name: str) -> str:
    return _STYLES[_current_theme][name]


def get_plot_bg() -> tuple[int, int, int]:
    return PLOT_VIEWER_BG_LIGHT if _current_theme == "light" else PLOT_VIEWER_BG


def get_agg_band_bg() -> tuple[int, int, int]:
    """Green-tinted background for range band (clear separation from frame timeline)."""
    return _AGG_BAND_BG_LIGHT if _current_theme == "light" else _AGG_BAND_BG


def get_agg_section_stylesheet() -> str:
    """Stylesheet for Reduce/range section: subtle green border, ties to agg band."""
    return _AGG_SECTION_STYLE_LIGHT if _current_theme == "light" else _AGG_SECTION_STYLE


def get_agg_groupbox_stylesheet() -> str:
    """Stylesheet for Range QGroupBox: title in green border, same content styling as agg_section."""
    hc = get_agg_heading_color()
    if _current_theme == "light":
        return (
            "QGroupBox#agg_section { "
            "background-color: rgba(200, 235, 210, 120); "
            "border: 2px solid rgba(100, 160, 110, 200); border-radius: 4px; "
            "margin: 4px 0 0 0; padding: 6px 6px 6px 6px; padding-top: 18px; } "
            "QGroupBox#agg_section::title { "
            f"subcontrol-origin: margin; left: 8px; padding: 0 4px; "
            f"color: {hc}; font-weight: bold; font-size: 10pt; }} "
            "QGroupBox#agg_section QComboBox { "
            "background-color: white; border: 1px solid #a0a8c0; "
            "border-radius: 3px; min-height: 20px; padding: 2px 6px; } "
            "QGroupBox#agg_section QSpinBox { "
            "background-color: white; border: 1px solid #a0a8c0; "
            "border-radius: 2px; padding: 2px 4px; } "
            "QGroupBox#agg_section QLabel { color: #1a1b26; } "
            "QGroupBox#agg_section QCheckBox { color: #1a1b26; } "
            "QGroupBox#agg_section QPushButton { "
            "background-color: #e0e2e8; border: 1px solid #a0a8c0; "
            "border-radius: 3px; padding: 4px 10px; } "
            "QGroupBox#agg_section QPushButton:hover { background-color: #c8cad3; } "
        )
    return (
        "QGroupBox#agg_section { "
        "background-color: rgba(30, 45, 38, 100); "
        "border: 2px solid rgba(158, 206, 106, 140); border-radius: 4px; "
        "margin: 4px 0 0 0; padding: 6px 6px 6px 6px; padding-top: 18px; } "
        "QGroupBox#agg_section::title { "
        f"subcontrol-origin: margin; left: 8px; padding: 0 4px; "
        f"color: {hc}; font-weight: bold; font-size: 10pt; }} "
        "QGroupBox#agg_section QComboBox { "
        "background-color: #1a1b26; border: 1px solid #3b4261; "
        "border-radius: 3px; min-height: 20px; padding: 2px 6px; } "
        "QGroupBox#agg_section QSpinBox { "
        "background-color: #1a1b26; border: 1px solid #3b4261; "
        "border-radius: 2px; padding: 2px 4px; } "
        "QGroupBox#agg_section QLabel { color: #a9b1d6; } "
        "QGroupBox#agg_section QCheckBox { color: #a9b1d6; } "
        "QGroupBox#agg_section QPushButton { "
        "background-color: #3b4261; border: 1px solid #565f89; "
        "border-radius: 3px; padding: 4px 10px; } "
        "QGroupBox#agg_section QPushButton:hover { background-color: #414868; } "
    )


def get_agg_heading_color() -> str:
    """Color for 'Range' heading inside agg section."""
    return "#598249" if _current_theme == "light" else "#9ece6a"


def get_agg_separator_stylesheet() -> str:
    """Tokyo green separator bar (between timelines, before Range section). 4px like Trenner."""
    return f"background-color: {COLOR_GREEN}; border: none; margin: 0;"


def get_viewer_bg() -> tuple[int, int, int]:
    return VIEWER_BG_LIGHT if _current_theme == "light" else VIEWER_BG


def get_dialog_preview_bg() -> tuple[int, int, int]:
    return DIALOG_PREVIEW_BG_LIGHT if _current_theme == "light" else DIALOG_PREVIEW_BG


def get_load_data_color() -> tuple[int, int, int]:
    return LOAD_DATA_COLOR_LIGHT if _current_theme == "light" else LOAD_DATA_COLOR


# Same color as timeline vertical line (cursor) and timeline curve in both themes
TIMELINE_LINE_RGB = (122, 162, 247)  # Tokyo blue


def get_timeline_line_color() -> tuple[int, int, int]:
    """Color for timeline cursor (vert line) and timeline curve. Same in both themes."""
    return TIMELINE_LINE_RGB


def get_timeline_curve_color() -> tuple[int, int, int]:
    """Main timeline curve: same as timeline vert line (Tokyo blue)."""
    return get_timeline_line_color()


def get_timeline_curve_colors_rgbw() -> tuple:
    """Four colors for multi-channel timeline (r,g,b,w). Fourth = same as timeline line."""
    if _current_theme == "light":
        return ((200, 80, 80), (80, 160, 80), (122, 162, 247), TIMELINE_LINE_RGB)
    return ("r", "g", "b", TIMELINE_LINE_RGB)


# Tokyo Night Dark palette (hex)
COLOR_BLUE = "#7aa2f7"
COLOR_GREEN = "#9ece6a"
COLOR_RED = "#f7768e"
COLOR_ORANGE = "#e0af68"
COLOR_YELLOW = "#e0af68"
COLOR_FG = "#a9b1d6"
COLOR_BG_DARK = "#2d2e3a"
COLOR_BG_DARKER = "#1a1b26"

# Status styles (for blocking_status label)
STYLE_IDLE = f"background-color: {COLOR_BG_DARK}; color: {COLOR_FG};"
STYLE_SCAN = f"background-color: {COLOR_ORANGE}; color: {COLOR_BG_DARKER};"
STYLE_BUSY = f"background-color: {COLOR_RED}; color: white;"

# Statusbar background (idle = subtle, busy = strong orange for visibility)
STYLE_STATUSBAR_IDLE = f"QStatusBar {{ background-color: {COLOR_BG_DARK}; }}"
STYLE_STATUSBAR_BUSY = f"QStatusBar {{ background-color: {COLOR_ORANGE}; color: {COLOR_BG_DARKER}; }}"

# Option tab section headings
STYLE_HEADING = f"""
QLabel {{
    background-color: {COLOR_BG_DARK};
    qproperty-alignment: AlignCenter;
    border-bottom: 4px solid {COLOR_BLUE};
    font-size: 13pt;
    font-weight: bold;
    color: {COLOR_BLUE};
}}
"""

STYLE_HEADING_SMALL = f"""
QLabel {{
    background-color: {COLOR_BG_DARK};
    qproperty-alignment: AlignCenter;
    font-size: 10pt;
    font-weight: bold;
    color: {COLOR_GREEN};
}}
"""

# Primary action button (e.g. Open File, Open Folder)
STYLE_BUTTON_PRIMARY = f"""
QPushButton {{
    background-color: {COLOR_BLUE};
    color: {COLOR_BG_DARKER};
    font-weight: bold;
    border-radius: 4px;
}}
QPushButton:hover {{
    background-color: #8fb4ff;
}}
QPushButton:pressed {{
    background-color: #5a8aeb;
}}
"""

# Plot backgrounds (for pyqtgraph, use rgb tuples)
PLOT_BG = (45, 46, 58)  # ~COLOR_BG_DARK
PLOT_CPU_COLOR = (122, 162, 247)   # blue
PLOT_RAM_COLOR = (158, 206, 106)   # green
PLOT_DISK_COLOR = (247, 118, 142)  # red

# Dialog preview background (pg.PlotWidget)
DIALOG_PREVIEW_BG = (45, 46, 58)

# Viewer/plot background (warmer than pure black)
VIEWER_BG = (26, 27, 38)  # ~COLOR_BG_DARKER
PLOT_VIEWER_BG = (26, 27, 38)

# Range band: green-tinted (clear separation from frame timeline)
_AGG_BAND_BG = (22, 36, 30)
_AGG_BAND_BG_LIGHT = (220, 242, 230)

# Range section (Reduce, range controls): green border, no transparency on inputs
_AGG_SECTION_STYLE = (
    "QFrame#agg_section { "
    "background-color: rgba(30, 45, 38, 100); "
    "border: 2px solid rgba(158, 206, 106, 140); border-radius: 4px; "
    "margin: 4px 0 0 0; padding: 6px; } "
    "QFrame#agg_section QComboBox { "
    "background-color: #1a1b26; border: 1px solid #3b4261; "
    "border-radius: 3px; min-height: 20px; padding: 2px 6px; } "
    "QFrame#agg_section QSpinBox { "
    "background-color: #1a1b26; border: 1px solid #3b4261; "
    "border-radius: 2px; padding: 2px 4px; } "
    "QFrame#agg_section QLabel { color: #a9b1d6; } "
    "QFrame#agg_section QCheckBox { color: #a9b1d6; } "
    "QFrame#agg_section QPushButton { "
    "background-color: #3b4261; border: 1px solid #565f89; "
    "border-radius: 3px; padding: 4px 10px; } "
    "QFrame#agg_section QPushButton:hover { background-color: #414868; } "
)
_AGG_SECTION_STYLE_LIGHT = (
    "QFrame#agg_section { "
    "background-color: rgba(200, 235, 210, 120); "
    "border: 2px solid rgba(100, 160, 110, 200); border-radius: 4px; "
    "margin: 4px 0 0 0; padding: 6px; } "
    "QFrame#agg_section QComboBox { "
    "background-color: white; border: 1px solid #a0a8c0; "
    "border-radius: 3px; min-height: 20px; padding: 2px 6px; } "
    "QFrame#agg_section QSpinBox { "
    "background-color: white; border: 1px solid #a0a8c0; "
    "border-radius: 2px; padding: 2px 4px; } "
    "QFrame#agg_section QLabel { color: #1a1b26; } "
    "QFrame#agg_section QCheckBox { color: #1a1b26; } "
    "QFrame#agg_section QPushButton { "
    "background-color: #e0e2e8; border: 1px solid #a0a8c0; "
    "border-radius: 3px; padding: 4px 10px; } "
    "QFrame#agg_section QPushButton:hover { background-color: #c8cad3; } "
)

# Load data placeholder text (BGR for cv2.putText)
LOAD_DATA_COLOR = (247, 162, 122)  # Tokyo blue #7aa2f7 in BGR

# Light theme variants
VIEWER_BG_LIGHT = (230, 230, 235)
PLOT_VIEWER_BG_LIGHT = (230, 230, 235)
DIALOG_PREVIEW_BG_LIGHT = (220, 220, 230)
LOAD_DATA_COLOR_LIGHT = (114, 97, 55)  # Tokyo blue on light, BGR

STYLE_IDLE_LIGHT = "background-color: #e0e2e8; color: #565f89;"
STYLE_SCAN_LIGHT = "background-color: #e0af68; color: #1a1b26;"
STYLE_BUSY_LIGHT = "background-color: #f7768e; color: white;"

STYLE_HEADING_LIGHT = """
QLabel {
    background-color: #e0e2e8;
    qproperty-alignment: AlignCenter;
    border-bottom: 4px solid #7aa2f7;
    font-size: 13pt;
    font-weight: bold;
    color: #7aa2f7;
}
"""

STYLE_HEADING_SMALL_LIGHT = """
QLabel {
    background-color: #e0e2e8;
    qproperty-alignment: AlignCenter;
    font-size: 10pt;
    font-weight: bold;
    color: #598249;
}
"""

STYLE_BUTTON_PRIMARY_LIGHT = """
QPushButton {
    background-color: #7aa2f7;
    color: white;
    font-weight: bold;
    border-radius: 4px;
}
QPushButton:hover {
    background-color: #8fb4ff;
}
QPushButton:pressed {
    background-color: #5a8aeb;
}
"""

STYLE_TOGGLE_SWITCH = """
QCheckBox {
    font-size: 14pt;
    font-weight: bold;
    color: #7aa2f7;
    spacing: 12px;
    padding: 12px;
    border: 1px solid #3b4261;
    border-radius: 8px;
    background-color: #1f2335;
    margin-bottom: 8px;
}
QCheckBox::indicator {
    width: 24px;
    height: 24px;
}
QCheckBox:hover {
    border-color: #7aa2f7;
    background-color: #24283b;
}
"""

STYLE_TOGGLE_SWITCH_LIGHT = """
QCheckBox {
    font-size: 14pt;
    font-weight: bold;
    color: #565f89;
    spacing: 12px;
    padding: 12px;
    border: 1px solid #9aa5ce;
    border-radius: 8px;
    background-color: #e0e2e8;
    margin-bottom: 8px;
}
QCheckBox::indicator {
    width: 24px;
    height: 24px;
}
QCheckBox:hover {
    border-color: #7aa2f7;
    background-color: #c0caf5;
}
"""

_STYLES = {
    "dark": {
        "idle": STYLE_IDLE,
        "scan": STYLE_SCAN,
        "busy": STYLE_BUSY,
        "statusbar_idle": STYLE_STATUSBAR_IDLE,
        "statusbar_busy": STYLE_STATUSBAR_BUSY,
        "heading": STYLE_HEADING,
        "heading_small": STYLE_HEADING_SMALL,
        "button_primary": STYLE_BUTTON_PRIMARY,
        "toggle_switch": STYLE_TOGGLE_SWITCH,
        "color_red": COLOR_RED,
    },
    "light": {
        "idle": STYLE_IDLE_LIGHT,
        "scan": STYLE_SCAN_LIGHT,
        "busy": STYLE_BUSY_LIGHT,
        "statusbar_idle": "QStatusBar { background-color: #e0e2e8; }",
        "statusbar_busy": "QStatusBar { background-color: #e0af68; color: #1a1b26; }",
        "heading": STYLE_HEADING_LIGHT,
        "heading_small": STYLE_HEADING_SMALL_LIGHT,
        "button_primary": STYLE_BUTTON_PRIMARY_LIGHT,
        "toggle_switch": STYLE_TOGGLE_SWITCH_LIGHT,
        "color_red": "#c45c5c",
    },
}

# Full Tokyo Night Dark stylesheet
STYLESHEET_FULL = """
QWidget { background-color: #1a1b26; color: #a9b1d6; }
QMainWindow { background-color: #1a1b26; }

QPushButton {
    background-color: #24283b;
    color: #a9b1d6;
    border: 1px solid #3b4261;
    border-radius: 4px;
    padding: 6px 12px;
    min-width: 60px;
}
QPushButton:hover { background-color: #2d2e3a; border-color: #7aa2f7; }
QPushButton:pressed { background-color: #1f2335; }
QPushButton:disabled { color: #565f89; background-color: #1f2335; }
QPushButton:focus { border-color: #7aa2f7; }

QLineEdit, QSpinBox, QDoubleSpinBox, QPlainTextEdit {
    background-color: #1e2030;
    color: #a9b1d6;
    border: 1px solid #3b4261;
    border-radius: 4px;
    padding: 4px 8px;
    selection-background-color: #7aa2f7;
    selection-color: #1a1b26;
}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {
    border-color: #7aa2f7;
}

QComboBox {
    background-color: #1e2030;
    color: #a9b1d6;
    border: 1px solid #3b4261;
    border-radius: 4px;
    padding: 4px 8px;
    min-width: 80px;
}
QComboBox:hover { border-color: #7aa2f7; }
QComboBox::drop-down {
    border: none;
    background-color: #24283b;
    width: 20px;
}
QComboBox::down-arrow { border-color: #a9b1d6; }
QComboBox QAbstractItemView {
    background-color: #1e2030;
    color: #a9b1d6;
    selection-background-color: #7aa2f7;
    selection-color: #1a1b26;
}

QCheckBox, QRadioButton { spacing: 6px; color: #a9b1d6; }
QCheckBox::indicator {
    width: 18px;
    height: 18px;
    image: url(:/icon/check_off.svg);
}
QCheckBox::indicator:checked {
    image: url(:/icon/check_on.svg);
}
QRadioButton::indicator {
    width: 16px;
    height: 16px;
    background-color: #1e2030;
    border: 1px solid #3b4261;
    border-radius: 8px;
}
QRadioButton::indicator:checked { background-color: #7aa2f7; border-color: #7aa2f7; }

QTabWidget::pane {
    border: 1px solid #3b4261;
    border-radius: 4px;
    background-color: #1a1b26;
    top: -1px;
}
QTabBar::tab {
    background-color: #1f2335;
    color: #565f89;
    padding: 8px 16px;
    margin-right: 2px;
}
QTabBar::tab:selected { color: #7aa2f7; font-weight: bold; background-color: #1a1b26; border: 1px solid #3b4261; border-bottom: none; }
QTabBar::tab:hover:!selected { color: #a9b1d6; }

QMenuBar { background-color: #1a1b26; color: #a9b1d6; border-bottom: 1px solid #3b4261; }
QMenuBar::item:selected { background-color: #24283b; color: #7aa2f7; }
QMenu { background-color: #1e2030; color: #a9b1d6; border: 1px solid #3b4261; }
QMenu::item:selected { background-color: #7aa2f7; color: #1a1b26; }

QStatusBar { background-color: #1a1b26; color: #a9b1d6; border-top: 1px solid #7aa2f7; }

QScrollBar:vertical {
    background-color: #1a1b26;
    width: 12px;
    border-radius: 6px;
    margin: 2px;
}
QScrollBar::handle:vertical {
    background-color: #3b4261;
    border-radius: 6px;
    min-height: 24px;
}
QScrollBar::handle:vertical:hover { background-color: #565f89; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QScrollBar:horizontal {
    background-color: #1a1b26;
    height: 12px;
    border-radius: 6px;
    margin: 2px;
}
QScrollBar::handle:horizontal {
    background-color: #3b4261;
    border-radius: 6px;
    min-width: 24px;
}
QScrollBar::handle:horizontal:hover { background-color: #565f89; }
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0; }

QSlider::groove:horizontal { height: 6px; background-color: #24283b; border-radius: 3px; }
QSlider::handle:horizontal { width: 16px; margin: -5px 0; background-color: #7aa2f7; border-radius: 8px; }
QSlider::handle:horizontal:hover { background-color: #8fb4ff; }

QGroupBox {
    font-weight: bold;
    color: #7aa2f7;
    border: 1px solid #3b4261;
    border-radius: 4px;
    margin-top: 12px;
    padding-top: 12px;
}
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }

QFrame { background-color: transparent; color: #a9b1d6; }
QSplitter::handle { background-color: #3b4261; width: __SPLITTER_W__px; height: __SPLITTER_W__px; min-width: __SPLITTER_W__px; min-height: __SPLITTER_W__px; }

QProgressBar {
    background-color: #1e2030;
    border: 1px solid #3b4261;
    border-radius: 4px;
    text-align: center;
}
QProgressBar::chunk { background-color: #7aa2f7; border-radius: 3px; }

QTextEdit {
    background-color: #1e2030;
    color: #a9b1d6;
    border: 1px solid #3b4261;
    border-radius: 4px;
    selection-background-color: #7aa2f7;
    selection-color: #1a1b26;
}

QDialog { background-color: #1a1b26; }

QToolTip {
    background-color: #24283b;
    color: #a9b1d6;
    border: 1px solid #7aa2f7;
    padding: 4px 8px;
    border-radius: 4px;
}

QHeaderView::section {
    background-color: #1f2335;
    color: #7aa2f7;
    padding: 6px 8px;
    border: none;
    border-right: 1px solid #3b4261;
}
QHeaderView::section:hover { background-color: #24283b; }

QScrollArea { border: none; background-color: transparent; }
QDockWidget { color: #a9b1d6; titlebar-close-icon: none; }
"""

# Tokyo Night Light stylesheet
STYLESHEET_LIGHT = """
QWidget { background-color: #d5d6db; color: #565f89; }
QMainWindow { background-color: #d5d6db; }

QPushButton {
    background-color: #e0e2e8;
    color: #565f89;
    border: 1px solid #9aa5ce;
    border-radius: 4px;
    padding: 6px 12px;
    min-width: 60px;
}
QPushButton:hover { background-color: #c0caf5; border-color: #7aa2f7; }
QPushButton:pressed { background-color: #a9b1d6; }
QPushButton:disabled { color: #9aa5ce; background-color: #e0e2e8; }
QPushButton:focus { border-color: #7aa2f7; }

QLineEdit, QSpinBox, QDoubleSpinBox, QPlainTextEdit {
    background-color: #ffffff;
    color: #565f89;
    border: 1px solid #9aa5ce;
    border-radius: 4px;
    padding: 4px 8px;
    selection-background-color: #7aa2f7;
    selection-color: white;
}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus { border-color: #7aa2f7; }

QComboBox {
    background-color: #ffffff;
    color: #565f89;
    border: 1px solid #9aa5ce;
    border-radius: 4px;
    padding: 4px 8px;
    min-width: 80px;
}
QComboBox:hover { border-color: #7aa2f7; }
QComboBox::drop-down { border: none; background-color: #e0e2e8; width: 20px; }
QComboBox::down-arrow { border-color: #565f89; }
QComboBox QAbstractItemView {
    background-color: #ffffff;
    color: #565f89;
    selection-background-color: #7aa2f7;
    selection-color: white;
}

QCheckBox, QRadioButton { spacing: 6px; color: #565f89; }
QCheckBox::indicator { width: 18px; height: 18px; image: url(:/icon/check_off.svg); }
QCheckBox::indicator:checked { image: url(:/icon/check_on.svg); }
QRadioButton::indicator {
    width: 16px; height: 16px;
    background-color: #ffffff;
    border: 1px solid #9aa5ce;
    border-radius: 8px;
}
QRadioButton::indicator:checked { background-color: #7aa2f7; border-color: #7aa2f7; }

QTabWidget::pane {
    border: 1px solid #9aa5ce;
    border-radius: 4px;
    background-color: #d5d6db;
    top: -1px;
}
QTabBar::tab {
    background-color: #e0e2e8;
    color: #565f89;
    padding: 8px 16px;
    margin-right: 2px;
}
QTabBar::tab:selected { color: #7aa2f7; font-weight: bold; background-color: #d5d6db; border: 1px solid #9aa5ce; border-bottom: none; }
QTabBar::tab:hover:!selected { color: #1a1b26; }

QMenuBar { background-color: #d5d6db; color: #565f89; border-bottom: 1px solid #9aa5ce; }
QMenuBar::item:selected { background-color: #e0e2e8; color: #7aa2f7; }
QMenu { background-color: #ffffff; color: #565f89; border: 1px solid #9aa5ce; }
QMenu::item:selected { background-color: #7aa2f7; color: white; }

QStatusBar { background-color: #d5d6db; color: #565f89; border-top: 1px solid #7aa2f7; }

QScrollBar:vertical { background-color: #d5d6db; width: 12px; border-radius: 6px; margin: 2px; }
QScrollBar::handle:vertical { background-color: #9aa5ce; border-radius: 6px; min-height: 24px; }
QScrollBar::handle:vertical:hover { background-color: #565f89; }
QScrollBar:horizontal { background-color: #d5d6db; height: 12px; border-radius: 6px; margin: 2px; }
QScrollBar::handle:horizontal { background-color: #9aa5ce; border-radius: 6px; min-width: 24px; }
QScrollBar::handle:horizontal:hover { background-color: #565f89; }

QSlider::groove:horizontal { height: 6px; background-color: #e0e2e8; border-radius: 3px; }
QSlider::handle:horizontal { width: 16px; margin: -5px 0; background-color: #7aa2f7; border-radius: 8px; }

QGroupBox { font-weight: bold; color: #7aa2f7; border: 1px solid #9aa5ce; border-radius: 4px; margin-top: 12px; padding-top: 12px; }
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }

QFrame { background-color: transparent; color: #565f89; }
QSplitter::handle { background-color: #9aa5ce; width: __SPLITTER_W__px; height: __SPLITTER_W__px; min-width: __SPLITTER_W__px; min-height: __SPLITTER_W__px; }

QProgressBar { background-color: #e0e2e8; border: 1px solid #9aa5ce; border-radius: 4px; text-align: center; }
QProgressBar::chunk { background-color: #7aa2f7; border-radius: 3px; }

QTextEdit {
    background-color: #ffffff;
    color: #565f89;
    border: 1px solid #9aa5ce;
    border-radius: 4px;
    selection-background-color: #7aa2f7;
    selection-color: white;
}

QDialog { background-color: #d5d6db; }

QToolTip {
    background-color: #e0e2e8;
    color: #565f89;
    border: 1px solid #7aa2f7;
    padding: 4px 8px;
    border-radius: 4px;
}

QHeaderView::section {
    background-color: #e0e2e8;
    color: #7aa2f7;
    padding: 6px 8px;
    border: none;
    border-right: 1px solid #9aa5ce;
}
QHeaderView::section:hover { background-color: #c0caf5; }

QScrollArea { border: none; background-color: transparent; }
QDockWidget { color: #565f89; titlebar-close-icon: none; }
"""
