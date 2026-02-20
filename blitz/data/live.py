"""LiveView handler: Mock mode - Lissajous visualizer.

Software simulation: Lissajous curves rendered to image. No video file.
Output: grayscale (plasma-ready) or RGB (e.g. green on black).
"""

from typing import Optional

import cv2
import numpy as np
from PyQt6.QtCore import QObject, QThread, pyqtSignal

from ..tools import log
from .image import ImageData, MetaData


def _lissajous_curve_points(t: float, width: int, height: int, n_pts: int = 400) -> np.ndarray:
    """Single Lissajous curve (one per frame). Returns (N+1, 2) int32 array."""
    w, h = width, height
    cx, cy = w / 2.0, h / 2.0
    margin = min(w, h) // 8
    rx = (w - 2 * margin) / 2.0
    ry = (h - 2 * margin) / 2.0
    pts = []
    for i in range(n_pts + 1):
        s = i / n_pts
        x = cx + rx * (
            np.sin(3 * s * 2 * np.pi + t) * 0.4
            + np.sin(s * 2 * np.pi + t * 2) * 0.6
        )
        y = cy + ry * (
            np.sin(5 * s * 2 * np.pi + t * 1.3) * 0.3
            + np.cos(s * 2 * np.pi * 2 + t * 1.4) * 0.7
        )
        pts.append((x, y))
    return np.array(pts, dtype=np.int32)


def _lissajous_frame(t: float, width: int, height: int, grayscale: bool) -> np.ndarray:
    """One frame: black background, single Lissajous curve. No trail, no shadow."""
    w, h = width, height
    line_thick = max(1, min(w, h) // 256)
    pts_arr = _lissajous_curve_points(t, w, h)
    if grayscale:
        out = np.zeros((h, w), dtype=np.uint8)
        cv2.polylines(out, [pts_arr], isClosed=False, color=255, thickness=line_thick)
    else:
        out = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.polylines(out, [pts_arr], isClosed=False, color=(0, 255, 0), thickness=line_thick)
    return out


def _frames_to_imagedata(frames: np.ndarray, grayscale: bool) -> ImageData:
    """Wrap frames (T,H,W) or (T,H,W,3) as ImageData. ensure_4d in ImageData adds C if needed."""
    arr = np.asarray(frames, dtype=np.uint8)
    t, h, w = arr.shape[0], arr.shape[1], arr.shape[2]
    meta = MetaData(
        file_name="lissajous_viz",
        file_size_MB=0.0,
        size=(w, h),
        dtype=np.uint8,
        bit_depth=8,
        color_model="grayscale" if grayscale else "rgb",
    )
    return ImageData(image=arr, metadata=[meta] * t)


class _LissajousWorker(QObject):
    """Lissajous generator: one curve per frame, no trail."""

    frame_ready = pyqtSignal(object)
    stopped = pyqtSignal()

    def __init__(self, width: int, height: int, fps: float,
                 buffer_size: int, grayscale: bool):
        super().__init__()
        self._w = max(64, width)
        self._h = max(64, height)
        self._fps = max(1, min(60, fps))
        self._buffer_size = max(1, min(1024, buffer_size))
        self._grayscale = grayscale
        self._running = True
        self._t = 0.0
        self._buffer: list[np.ndarray] = []

    def run(self) -> None:
        interval_ms = max(1, int(1000 / self._fps))
        log(f"[LIVE] Lissajous: {self._w}x{self._h} @ {self._fps:.0f} FPS, "
            f"buffer={self._buffer_size}, gray={self._grayscale}")
        while self._running:
            frame = _lissajous_frame(self._t, self._w, self._h, self._grayscale)
            self._t += 0.08
            self._buffer.append(frame.copy())
            if len(self._buffer) > self._buffer_size:
                self._buffer.pop(0)
            img = _frames_to_imagedata(np.stack(self._buffer), self._grayscale)
            self.frame_ready.emit(img)
            QThread.msleep(interval_ms)
        self.stopped.emit()

    def stop(self) -> None:
        self._running = False


class MockLiveHandler(QObject):
    """Streams Lissajous viz. Grayscale default for plasma colormap."""

    frame_ready = pyqtSignal(object)
    stopped = pyqtSignal()

    def __init__(self, width: int = 512, height: int = 512, fps: float = 30.0,
                 buffer_size: int = 64, grayscale: bool = True):
        super().__init__()
        self._width = width
        self._height = height
        self._fps = fps
        self._buffer_size = buffer_size
        self._grayscale = grayscale
        self._thread: Optional[QThread] = None
        self._worker: Optional[_LissajousWorker] = None

    def start(self) -> None:
        if self._thread is not None and self._thread.isRunning():
            return
        self._thread = QThread()
        self._worker = _LissajousWorker(
            self._width, self._height, self._fps,
            self._buffer_size, self._grayscale,
        )
        self._worker.moveToThread(self._thread)
        self._worker.frame_ready.connect(self.frame_ready.emit)
        self._worker.stopped.connect(self._on_worker_stopped)
        self._thread.started.connect(self._worker.run)
        self._thread.start()

    def stop(self) -> None:
        if self._worker:
            self._worker.stop()

    def wait_stopped(self, timeout_ms: int = 3000) -> bool:
        from PyQt6.QtCore import QCoreApplication, QElapsedTimer
        if not self._thread or not self._thread.isRunning():
            return True
        timer = QElapsedTimer()
        timer.start()
        while self._thread.isRunning() and timer.elapsed() < timeout_ms:
            QCoreApplication.processEvents()
            self._thread.wait(20)
        return not self._thread.isRunning()

    def _on_worker_stopped(self) -> None:
        if self._thread:
            self._thread.quit()
            self._thread.wait(2000)
        self.stopped.emit()

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.isRunning()
