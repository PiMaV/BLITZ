import multiprocessing
import sys

import pyqtgraph as pg
from PyQt6.QtCore import QCoreApplication
from PyQt6.QtWidgets import QApplication

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


def run() -> int:
    multiprocessing.freeze_support()
    pg.setConfigOptions(useNumba=False)
    exit_code = 0
    restart_exit_code = settings.get("app/restart_exit_code")
    app = QApplication(sys.argv)
    theme = settings.get("app/theme")
    set_theme("light" if theme == "light" else "dark")
    app.setStyleSheet(get_stylesheet())

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
