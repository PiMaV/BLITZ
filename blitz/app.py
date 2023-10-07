import sys

import qdarkstyle
from PyQt5.QtWidgets import QApplication

from .layout.main import MainWindow


def run() -> int:
    app = QApplication(sys.argv)
    app.setStyleSheet(qdarkstyle.load_stylesheet(qt_api='pyqt5'))
    main_window = MainWindow()
    main_window.load_images()
    main_window.show()
    return app.exec()
