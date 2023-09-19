from typing import Any, Optional

import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtGui import QDropEvent
from PyQt5.QtWidgets import (QDialog, QLabel, QMainWindow, QPushButton,
                             QVBoxLayout, QTextEdit)


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


class PlotterWindow:
    def __init__(self, main_window: QMainWindow) -> None:
        self.mainWindow = main_window

        self.window = QMainWindow()
        self.window.setWindowTitle("Additional Plot")
        self.window.resize(500, 400)

        # Create a PlotWidget and set data
        self.plot_item  = pg.PlotItem()
        self.plot_widget = pg.PlotWidget(plotItem=self.plot_item)
        self.legend = self.plot_item.addLegend()

        self.window.setCentralWidget(self.plot_widget)

    def plot_data(
        self,
        x_data,
        y_data,
        pen_color,
        label: Optional[str] = None,
    ) -> None:
        plot_data = self.plot_item.plot(x_data, y_data, pen=pen_color, width=1)
        if label:
            self.legend.addItem(plot_data, label)

    def show(self):
        self.window.show()


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


class DraggableCrosshair:
    def __init__(self, main_window: QMainWindow) -> None:
        self.main_window = main_window
        self.v_line_itself = pg.InfiniteLine(angle=90, movable=True)
        self.h_line_itself = pg.InfiniteLine(angle=0, movable=True)
        self.main_window.imageView.addItem(self.v_line_itself)
        self.main_window.imageView.addItem(self.h_line_itself)
        self.setup_connections()
        self.toggle_crosshair()

    def setup_connections(self) -> None:
        self.v_line_itself.sigPositionChanged.connect(self.update_plots)
        self.h_line_itself.sigPositionChanged.connect(self.update_plots)
        self.items_added = False

    def toggle_crosshair(self):
        crosshair_state = self.main_window.rootParamEdit.param(
            'Visualisations', 'Crosshair'
        ).value()

        if crosshair_state:
            if self.v_line_itself not in self.main_window.imageView.view.items:
                self.main_window.imageView.addItem(self.v_line_itself)

            if self.h_line_itself not in self.main_window.imageView.view.items:
                self.main_window.imageView.addItem(self.h_line_itself)

            self.v_line_itself.setPen(self.main_window.crosshair_pen)
            self.h_line_itself.setPen(self.main_window.crosshair_pen)

            self.v_line_itself.setMovable(True)
            self.h_line_itself.setMovable(True)
        else:
            self.v_line_itself.setMovable(False)
            self.h_line_itself.setMovable(False)

            self.main_window.imageView.removeItem(self.h_line_itself)
            self.main_window.imageView.removeItem(self.v_line_itself)

    def update_plots(self):
        x = int(self.v_line_itself.getPos()[0])
        y = int(self.h_line_itself.getPos()[1])
        frame_idx = int(self.main_window.imageView.timeLine.value())

        self.main_window.v_line.clear()
        self.main_window.h_line.clear()
        self.plot_data(frame_idx, x, y)

    def plot_data(self, frame_idx: int, x: int, y: int) -> None:
        frame_max = self.main_window.masked_data.shape[0] - 1
        x_max = self.main_window.masked_data.shape[1] - 1
        y_max = self.main_window.masked_data.shape[2] - 1
        if frame_idx > frame_max:
            print("Frame index out of bounds, skipping plotting.")
            return

        x_constrained = min(max(0, x), x_max)
        y_constrained = min(max(0, y), y_max)

        if x_constrained != x or y_constrained != y:
            print("Indices out of bounds, skipping plotting.")
            return

        x_values = np.arange(self.main_window.masked_data.shape[2])

        if (len(self.main_window.masked_data.shape) == 4
                and self.main_window.masked_data.shape[3] == 3):
            # colored image
            r_data_v = self.main_window.masked_data[frame_idx, x, :, 0]
            g_data_v = self.main_window.masked_data[frame_idx, x, :, 1]
            b_data_v = self.main_window.masked_data[frame_idx, x, :, 2]

            r_data_h = self.main_window.masked_data[frame_idx, :, y, 0]
            g_data_h = self.main_window.masked_data[frame_idx, :, y, 1]
            b_data_h = self.main_window.masked_data[frame_idx, :, y, 2]

            self.main_window.v_line.plot(r_data_v, x_values, pen='r')
            self.main_window.v_line.plot(g_data_v, x_values, pen='g')
            self.main_window.v_line.plot(b_data_v, x_values, pen='b')

            self.main_window.h_line.plot(r_data_h, pen='r')
            self.main_window.h_line.plot(g_data_h, pen='g')
            self.main_window.h_line.plot(b_data_h, pen='b')
        else:
            # grayscale image
            v_data = self.main_window.masked_data[frame_idx, x, :]
            h_data = self.main_window.masked_data[frame_idx, :, y]
            self.main_window.v_line.plot(v_data, x_values, pen='gray')
            self.main_window.h_line.plot(h_data, pen='gray')
