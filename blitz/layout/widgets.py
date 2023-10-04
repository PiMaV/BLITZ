from typing import Any, Optional

import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtGui import QDropEvent
from PyQt5.QtWidgets import (QDialog, QLabel, QMainWindow, QPushButton,
                             QTextEdit, QVBoxLayout, QWidget)


class LoadingDialog:
    def __init__(self, main_window: QMainWindow, message: str) -> None:
        self.main_window = main_window
        self.dialog = QDialog()
        self.dialog.setWindowTitle(message)
        layout = QVBoxLayout()
        label = QLabel(message)
        layout.addWidget(label)
        self.dialog.setLayout(layout)
        self.dialog.setModal(True)


class WindowedPlot(QMainWindow):
    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)

        self.plot_item  = pg.PlotItem()
        self.plot_widget = pg.PlotWidget(plotItem=self.plot_item)
        self.legend = self.plot_item.addLegend()

        self.setCentralWidget(self.plot_widget)

    def plot_data(
        self,
        x_data: np.ndarray,
        y_data: np.ndarray,
        pen_color: str,
        label: Optional[str] = None,
    ) -> None:
        plot_data = self.plot_item.plot(x_data, y_data, pen=pen_color, width=1)
        if label:
            self.legend.addItem(plot_data, label)


class DragDropButton(QPushButton):
    file_dropped = pyqtSignal(str)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setAcceptDrops(True)

    def dragEnterEvent(self, e: QDropEvent):
        if e.mimeData().hasUrls():
            e.accept()
        else:
            e.ignore()

    def dropEvent(self, e: QDropEvent) -> None:
        file_path = e.mimeData().urls()[0].toLocalFile()
        self.file_dropped.emit(file_path)


class LoggingTextEdit(QTextEdit):
    def write(self, message: Any):
        cursor = self.textCursor()
        cursor.movePosition(cursor.End)
        cursor.insertText(str(message))
        self.setTextCursor(cursor)
        self.ensureCursorVisible()

    def flush(self):
        pass
