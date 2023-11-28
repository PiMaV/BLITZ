import textwrap
from typing import Any, Sequence, Self
from timeit import default_timer as clock

from PyQt5.QtWidgets import QApplication

import numpy as np
import psutil

LOGGER: Any = None
LOADING_LABEL: Any = None
SPINNER: Any = None


def log(message: str) -> None:
    if LOGGER is None:
        print(message)
    else:
        LOGGER.log(message)


def setup_logger(logger: Any) -> None:
    global LOGGER
    LOGGER = logger


def setup_loading_widgets(label: Any, spinner: Any) -> None:
    global LOADING_LABEL, SPINNER
    LOADING_LABEL = label
    SPINNER = spinner


class LoadingManager:

    def __init__(self, text: str = "Loading...") -> None:
        self.text = text
        self._start_time = 0
        self._time_needed = 0

    @property
    def time(self) -> float:
        return clock() - self._start_time

    @property
    def duration(self) -> float:
        return self._time_needed

    def __enter__(self) -> Self:
        LOADING_LABEL.setText(self.text)
        SPINNER.start()
        self._start_time = clock()
        return self

    def __exit__(self, type, value, traceback) -> None:
        LOADING_LABEL.setText("")
        SPINNER.stop()
        self._time_needed = clock() - self._start_time


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
