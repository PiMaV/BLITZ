from timeit import default_timer as clock
from types import TracebackType
from typing import Any, Optional, Self, Sequence

import numpy as np
import psutil
from PyQt5.QtGui import QColor, QTextCharFormat
from PyQt5.QtWidgets import (QApplication, QDialog, QLabel, QTextEdit,
                             QVBoxLayout)

from . import settings

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
        label = QLabel(message)
        layout.addWidget(label)
        self.setLayout(layout)
        self.setModal(True)


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
        format.setForeground(QColor(color))
        format.setFontPointSize(settings.get("viewer/font_size_log"))
        cursor.mergeCharFormat(format)
        if message != "\n":
            cursor.insertText(f"> {message}\n")
        self.setTextCursor(cursor)
        self.ensureCursorVisible()


class LoadingManager:

    def __init__(self, parent, text: str = "Loading ...") -> None:
        self.text = text
        self.parent = parent
        self._start_time = 0
        self._time_needed = 0

    @property
    def time(self) -> float:
        return clock() - self._start_time

    @property
    def duration(self) -> float:
        return self._time_needed

    def __enter__(self) -> Self:
        self._dialog = LoadingDialog(self.parent, self.text)
        self._dialog.show()
        QApplication.processEvents()
        self._start_time = clock()
        return self

    def __exit__(
        self,
        exctype: Optional[type[BaseException]],
        excinst: Optional[BaseException],
        exctb: Optional[TracebackType],
    ) -> bool:
        self._dialog.accept()
        del self._dialog
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
        try:
            return f"({value[0]:3d}, {value[1]:3d}, {value[2]:3d})"
        except (ValueError, TypeError):
            return f"({value[0]:.3f}, {value[1]:.3f}, {value[2]:.3f})"
    else:
        try:
            return f"{value:4d}"
        except (ValueError, TypeError):
            return f"{value:.4f}"
