from timeit import default_timer as clock
from types import TracebackType
from typing import Any, Optional, Self, Sequence

import numpy as np
import psutil
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QColor, QFont, QTextCharFormat
from PyQt5.QtWidgets import (QApplication, QDialog, QLabel, QProgressBar,
                             QTextEdit, QVBoxLayout)

from . import settings

PROGRESS_DIALOG_DELAY_MS = 1000

LOGGER: Any = None


def log(message: str, color: str | tuple[int, int, int] = "white") -> None:
    if LOGGER is None:
        print(message)
    else:
        LOGGER.log(message, color=color)


def setup_logger(logger: Any) -> None:
    global LOGGER
    LOGGER = logger


class LoadingDialog(QDialog):

    def __init__(self, parent, message: str = "Loading ...") -> None:
        super().__init__(parent)
        self.setWindowTitle(message)
        layout = QVBoxLayout()
        self._label = QLabel(message)
        layout.addWidget(self._label)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)
        self.setLayout(layout)
        self.setModal(True)

    def set_progress(self, value: int) -> None:
        self.progress_bar.setValue(value)

    def set_message(self, message: str) -> None:
        self._label.setText(message)
        self.setWindowTitle(message)


class LoggingTextEdit(QTextEdit):

    COLOR_WARNING = QColor(255, 0, 0)
    COLOR_TEXT = QColor(255, 255, 255)

    def log(
        self,
        message: Any,
        color: str | tuple[int, int, int] = "white",
    ) -> None:
        cursor = self.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        format = QTextCharFormat()
        format.setFont(QFont("Courier New"))
        format.setForeground(QColor(color))
        format.setFontPointSize(settings.get("viewer/font_size_log"))
        cursor.mergeCharFormat(format)
        if message != "\n":
            cursor.insertText(f"> {message}\n")
        self.setTextCursor(cursor)
        self.ensureCursorVisible()


class LoadingManager:

    def __init__(
        self,
        parent,
        text: str = "Loading ...",
        delay_ms: int = PROGRESS_DIALOG_DELAY_MS,
    ) -> None:
        self.text = text
        self.parent = parent
        self._delay_ms = delay_ms
        self._start_time = 0
        self._time_needed = 0
        self._dialog: Optional[LoadingDialog] = None
        self._timer: Optional[QTimer] = None
        self._dialog_shown = False

    @property
    def time(self) -> float:
        return clock() - self._start_time

    @property
    def duration(self) -> float:
        return self._time_needed

    def _show_dialog(self) -> None:
        if not self._dialog_shown and self._dialog is not None:
            self._dialog.show()
            QApplication.processEvents()
            self._dialog_shown = True

    def __enter__(self) -> Self:
        self._start_time = clock()
        self._dialog = LoadingDialog(self.parent, self.text)
        self._timer = QTimer()
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._show_dialog)
        self._timer.start(self._delay_ms)
        return self

    def set_progress(self, value: int) -> None:
        if self._dialog_shown and self._dialog is not None:
            self._dialog.set_progress(value)
            QApplication.processEvents()

    def set_message(self, message: str) -> None:
        if self._dialog_shown and self._dialog is not None:
            self._dialog.set_message(message)
            QApplication.processEvents()

    def __exit__(
        self,
        exctype: Optional[type[BaseException]],
        excinst: Optional[BaseException],
        exctb: Optional[TracebackType],
    ) -> bool:
        if self._timer is not None:
            self._timer.stop()
            self._timer = None
        if self._dialog is not None:
            if self._dialog_shown:
                self._dialog.accept()
            del self._dialog
            self._dialog = None
        self._time_needed = clock() - self._start_time
        return False


def get_available_ram() -> float:
    available_ram = psutil.virtual_memory().available / (1024**3)
    return available_ram


def fit_text(text: str, max_length: int) -> str:
    if len(text) > max_length:
        text = "[...]" + text[-max_length+5:]
    return text


def format_pixel_value(
    value: str | Sequence[float | int] | np.ndarray | None,
) -> str:
    if isinstance(value, str) or value is None:
        return f"{value}"
    elif isinstance(value, (list, tuple, np.ndarray)) and len(value) == 3:
        return f"({value[0]:.3f}, {value[1]:.3f}, {value[2]:.3f})"
    else:
        if int(value[0]) == value[0]:
            return f"{int(value[0]):4d}"
        return f"{value[0]:4f}"
