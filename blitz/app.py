import multiprocessing
import sys

import pyqtgraph as pg
import qdarkstyle
from PyQt5.QtCore import QCoreApplication
from PyQt5.QtWidgets import QApplication

from . import settings
from .layout.main import MainWindow


def run() -> int:
    multiprocessing.freeze_support()
    pg.setConfigOptions(useNumba=True)
    exit_code = 0
    restart_exit_code = settings.get("app/restart_exit_code")
    app = QApplication(sys.argv)
    app.setStyleSheet(qdarkstyle.load_stylesheet(qt_api='pyqt5'))
    while True:
        app = QCoreApplication.instance()
        main_window = MainWindow()
        main_window.show()
        exit_code = app.exec_()
        if exit_code != restart_exit_code:
            break
        main_window.deleteLater()
    return exit_code
