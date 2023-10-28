import multiprocessing
import sys

import pyqtgraph as pg
import qdarkstyle
from PyQt5.QtWidgets import QApplication

from .layout.main import MainWindow


def run() -> int:
    multiprocessing.freeze_support()
    pg.setConfigOptions(useNumba=True)
    app = QApplication(sys.argv)
    app.setStyleSheet(qdarkstyle.load_stylesheet(qt_api='pyqt5'))
    main_window = MainWindow()
    main_window.load_images()
    main_window.show()
    return app.exec()
