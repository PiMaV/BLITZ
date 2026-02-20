import multiprocessing
import sys

import pyqtgraph as pg
from PyQt6.QtCore import QCoreApplication
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import (QApplication, QDialog, QLabel, QPushButton,
                              QVBoxLayout)

# PyQt6 + pyqtgraph: when a ViewBox is destroyed, forgetView() calls updateAllViewLists()
# on every remaining ViewBox; some menus' QComboBox may already be deleted -> RuntimeError.
# Guard setViewList so teardown does not crash.
try:
    from pyqtgraph.graphicsItems.ViewBox.ViewBoxMenu import ViewBoxMenu
    _orig_set_view_list = ViewBoxMenu.setViewList

    def _safe_set_view_list(self, views):
        try:
            _orig_set_view_list(self, views)
        except RuntimeError as e:
            if "has been deleted" in str(e):
                return
            raise

    ViewBoxMenu.setViewList = _safe_set_view_list
except Exception:  # noqa: S110
    pass

from . import resources  # noqa: F401  (registers Qt resources for icon; needed before dialogs)
from . import settings
from .layout.main import MainWindow


from .theme import get_stylesheet, set_theme


def _show_first_start_welcome(app: QApplication) -> None:
    """First start: welcome. Reserved for important info later."""
    dlg = QDialog()
    dlg.setWindowIcon(QIcon(":/icon/blitz.ico"))
    dlg.setWindowTitle("Welcome to BLITZ")
    dlg.setMinimumWidth(420)
    layout = QVBoxLayout(dlg)
    layout.addWidget(QLabel("<b>Hallo und viel Spass!</b>"))
    btn = QPushButton("Got it")
    btn.setStyleSheet("background-color: #7aa2f7; color: #1a1b26;")
    btn.clicked.connect(dlg.accept)
    layout.addWidget(btn)
    dlg.exec()


def run() -> int:
    multiprocessing.freeze_support()
    pg.setConfigOptions(useNumba=False)
    exit_code = 0
    restart_exit_code = settings.get("app/restart_exit_code")
    app = QApplication(sys.argv)
    theme = settings.get("app/theme")
    set_theme("light" if theme == "light" else "dark")
    app.setStyleSheet(get_stylesheet())

    # Migration: boot_bench_done -> first_start_welcome_shown
    shown = settings.get("default/first_start_welcome_shown")
    if not shown:
        try:
            from .settings import SETTINGS
            if SETTINGS is not None and SETTINGS.settings.contains("default/boot_bench_done"):
                if SETTINGS.settings.value("default/boot_bench_done", False, type=bool):
                    shown = True
                    settings.set("default/first_start_welcome_shown", True)
        except Exception:
            pass
    if not shown:
        _show_first_start_welcome(app)
        settings.set("default/first_start_welcome_shown", True)

    while True:
        app = QCoreApplication.instance()
        theme = settings.get("app/theme")
        set_theme("light" if theme == "light" else "dark")
        app.setStyleSheet(get_stylesheet())
        main_window = MainWindow()
        main_window.show()
        exit_code = app.exec()
        if exit_code != restart_exit_code:
            break
        main_window.deleteLater()
    return exit_code
