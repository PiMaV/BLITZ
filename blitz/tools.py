import sys
from timeit import default_timer as clock
from types import TracebackType
from typing import Any, Optional, Self, Sequence

import numpy as np
import psutil
from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QColor, QFont, QTextCharFormat
from PyQt6.QtWidgets import (QApplication, QDialog, QLabel, QProgressBar,
                             QStatusBar, QTextEdit, QVBoxLayout)

from . import settings
from .theme import get_style

PROGRESS_DIALOG_DELAY_MS = 1000
BLOCKING_BAR_DELAY_MS = 500

LOGGER: Any = None
_LOG_ONE_LINER: Optional[QLabel] = None


def log(message: str, color: str | tuple[int, int, int] = "white") -> None:
    if LOGGER is None:
        print(message)
    else:
        LOGGER.log(message, color=color)
    if _LOG_ONE_LINER is not None and message != "\n":
        txt = str(message).strip()[:120]
        _LOG_ONE_LINER.setText(txt)
        _LOG_ONE_LINER.setToolTip(str(message).strip())


def setup_logger(logger: Any, one_liner_label: Optional[QLabel] = None) -> None:
    global LOGGER, _LOG_ONE_LINER
    LOGGER = logger
    _LOG_ONE_LINER = one_liner_label


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
        qc = QColor(*color) if isinstance(color, tuple) else QColor(color)
        format.setForeground(qc)
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
        status_label: Optional[QLabel] = None,
        blocking_label: Optional[QLabel] = None,
        statusbar: Optional[QStatusBar] = None,
        blocking_delay_ms: Optional[int] = BLOCKING_BAR_DELAY_MS,
    ) -> None:
        self.text = text
        self.parent = parent
        self._delay_ms = delay_ms
        self._status_label = status_label
        self._blocking_label = blocking_label
        self._statusbar = statusbar
        self._blocking_delay_ms = blocking_delay_ms
        self._start_time = 0
        self._time_needed = 0
        self._dialog: Optional[LoadingDialog] = None
        self._timer: Optional[QTimer] = None
        self._blocking_timer: Optional[QTimer] = None
        self._blocking_shown = False
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

    def _show_blocking_indicator(self) -> None:
        if self._blocking_shown:
            return
        self._blocking_shown = True
        if self._blocking_label is not None:
            self._blocking_label.setText("BUSY")
            self._blocking_label.setStyleSheet(get_style("busy"))
            if self._blocking_label.parent():
                self._blocking_label.parent().update()
        if self._statusbar is not None:
            self._statusbar.setStyleSheet(get_style("statusbar_busy"))
        for _ in range(3):
            QApplication.processEvents()

    def __enter__(self) -> Self:
        self._start_time = clock()
        if self._status_label is not None:
            self._status_label.setText(self.text)
            QApplication.processEvents()
        if self._blocking_label is not None or self._statusbar is not None:
            if self._blocking_delay_ms is None or self._blocking_delay_ms <= 0:
                self._show_blocking_indicator()
            else:
                self._blocking_timer = QTimer()
                self._blocking_timer.setSingleShot(True)
                self._blocking_timer.timeout.connect(self._show_blocking_indicator)
                self._blocking_timer.start(self._blocking_delay_ms)
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
        if self._blocking_timer is not None:
            self._blocking_timer.stop()
            self._blocking_timer = None
        if self._timer is not None:
            self._timer.stop()
            self._timer = None
        if self._dialog is not None:
            if self._dialog_shown:
                self._dialog.accept()
            del self._dialog
            self._dialog = None
        if self._status_label is not None:
            self._status_label.setText("")
        if self._blocking_shown:
            if self._blocking_label is not None:
                self._blocking_label.setText("IDLE")
                self._blocking_label.setStyleSheet(get_style("idle"))
            if self._statusbar is not None:
                self._statusbar.setStyleSheet(get_style("statusbar_idle"))
        self._time_needed = clock() - self._start_time
        return False


def get_available_ram() -> float:
    available_ram = psutil.virtual_memory().available / (1024**3)
    return available_ram


def get_used_ram() -> float:
    """Used RAM in GB."""
    return psutil.virtual_memory().used / (1024**3)


def get_cpu_percent() -> float:
    """CPU usage in percent (0-100). Non-blocking."""
    return psutil.cpu_percent(interval=None)


def get_cpu_percore() -> list[float]:
    """CPU usage per core (0-100). Non-blocking."""
    return psutil.cpu_percent(interval=None, percpu=True)


def get_cpu_freq_mhz() -> Optional[float]:
    """Current CPU frequency in MHz, or None. On Windows often nominal only."""
    try:
        cf = psutil.cpu_freq()
        return float(cf.current) if cf and cf.current else None
    except Exception:
        return None


def get_sensors_temperatures() -> dict[str, list[Any]]:
    """Hardware temperatures by sensor name. Empty dict on Windows (not supported by psutil)."""
    if sys.platform == "win32":
        return {}
    try:
        return psutil.sensors_temperatures() or {}
    except Exception:
        return {}


def get_sensors_fans() -> dict[str, list[Any]]:
    """Hardware fan speeds (RPM) by sensor name. Empty dict on Windows (not supported by psutil)."""
    if sys.platform == "win32":
        return {}
    try:
        return psutil.sensors_fans() or {}
    except Exception:
        return {}


def get_disk_io_mbs() -> tuple[float, float]:
    """Disk I/O in MB/s (read, write). Cached; call each tick for rate."""
    now = psutil.disk_io_counters()
    if now is None:
        return 0.0, 0.0
    prev = getattr(get_disk_io_mbs, "_prev", None)
    if prev is None:
        get_disk_io_mbs._prev = (now.read_bytes, now.write_bytes)
        return 0.0, 0.0
    read_mbs = (now.read_bytes - prev[0]) / (1024**2)
    write_mbs = (now.write_bytes - prev[1]) / (1024**2)
    get_disk_io_mbs._prev = (now.read_bytes, now.write_bytes)
    return max(0.0, read_mbs), max(0.0, write_mbs)


def format_size_mb(nbytes: int | float) -> str:
    """Format bytes as XX.X MB."""
    return f"{nbytes / (1024**2):.1f} MB"


def fit_text(text: str, max_length: int) -> str:
    if len(text) > max_length:
        text = "[...]" + text[-max_length+5:]
    return text


def format_pixel_value(
    value: str | Sequence[float | int] | np.ndarray | None,
) -> str:
    """Format for display: scalar float (max 2 decimals), int, or RGB tuple."""
    if isinstance(value, str) or value is None:
        return f"{value}"
    elif isinstance(value, (list, tuple, np.ndarray)) and len(value) == 3:
        arr = np.asarray(value, dtype=float)
        if np.allclose(arr, np.round(arr)):
            return f"({int(round(arr[0]))}, {int(round(arr[1]))}, {int(round(arr[2]))})"
        return f"({arr[0]:.2f}, {arr[1]:.2f}, {arr[2]:.2f})"
    else:
        v = np.asarray(value).flat[0]
        if int(v) == v:
            return f"{int(v)}"
        return f"{v:.2f}"


def format_pixel_value_fixed(
    value: np.ndarray | Sequence[float | int] | None,
    bits: int,
    is_rgb: bool,
) -> str:
    """Fixed-width pixel value for stable layout. Width depends on bits (8→3, 16→5, 32→10)."""
    width = {8: 3, 16: 5, 32: 10}.get(bits, 5)
    if value is None:
        return "—" if not is_rgb else "(—, —, —)"
    arr = np.asarray(value)
    if is_rgb and arr.size >= 3:
        parts = arr.flatten()[:3]
        vals = [int(round(float(p))) for p in parts]
        return f"({vals[0]:{width}d},{vals[1]:{width}d},{vals[2]:{width}d})"
    v = int(round(float(arr.flat[0])))
    return f"{v:{width}d}"
