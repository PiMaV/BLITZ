import textwrap
from timeit import default_timer as clock
from types import TracebackType
from typing import Any, Optional, Self, Sequence

import numpy as np
import psutil
from PyQt5.QtWidgets import (QApplication, QDialog, QLabel, QTextEdit,
                             QVBoxLayout)


LOGGER: Any = None


def log(message: str) -> None:
    if LOGGER is None:
        print(message)
    else:
        LOGGER.log(message)


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

    def log(self, message: Any):
        cursor = self.textCursor()
        cursor.movePosition(cursor.End)
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


def wrap_text(text: str, max_length: int) -> str:
    return '\n'.join(textwrap.wrap(text, max_length))


def format_pixel_value(
    value: str | Sequence[float | int] | np.ndarray | None,
) -> str:
    if isinstance(value, str) or value is None:
        return f"{value}"
    elif isinstance(value, (list, tuple, np.ndarray)) and len(value) == 3:
        return f"({int(value[0]):3d}, {int(value[1]):3d}, {int(value[2]):3d})"
    else:
        try:
            return f"{int(value):4d}"  # type: ignore
        except (ValueError, TypeError):
            return "Invalid value"
