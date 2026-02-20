"""LiveView handler: Real camera via cv2.VideoCapture.

USB webcams, etc. Exposure, Gain, FPS configurable.
On Windows: uses CAP_DSHOW (DirectShow) for better webcam support.
"""

import sys
from typing import Optional

import cv2
import numpy as np
from PyQt6.QtCore import QObject, QThread, pyqtSignal

from ..tools import log
from .image import ImageData, MetaData

# Backend: Windows webcams often need DirectShow
_CAP_BACKEND = getattr(cv2, "CAP_DSHOW", 0) if sys.platform == "win32" else 0

# cv2 property ids (OpenCV 4+)
_CAP_EXPOSURE = getattr(cv2, "CAP_PROP_EXPOSURE", 15)
_CAP_GAIN = getattr(cv2, "CAP_PROP_GAIN", 14)
_CAP_BRIGHTNESS = getattr(cv2, "CAP_PROP_BRIGHTNESS", 10)
_CAP_CONTRAST = getattr(cv2, "CAP_PROP_CONTRAST", 11)
_CAP_FPS = getattr(cv2, "CAP_PROP_FPS", 5)
_CAP_AUTO_EXPOSURE = getattr(cv2, "CAP_PROP_AUTO_EXPOSURE", 21)


def _frames_to_imagedata(frames: np.ndarray, grayscale: bool) -> ImageData:
    """Wrap frames (T,H,W) or (T,H,W,3) as ImageData. swapaxes wie Video-Load (T,W,H)."""
    arr = np.asarray(frames, dtype=np.uint8)
    arr = np.swapaxes(arr, 1, 2)  # (T,H,W) -> (T,W,H) fuer pyqtgraph
    t, w, h = arr.shape[0], arr.shape[1], arr.shape[2]
    meta = MetaData(
        file_name="camera",
        file_size_MB=0.0,
        size=(w, h),
        dtype=np.uint8,
        bit_depth=8,
        color_model="grayscale" if grayscale else "rgb",
    )
    return ImageData(image=arr, metadata=[meta] * t)


class _CameraWorker(QObject):
    """Reads frames from cv2.VideoCapture, emits at target FPS."""

    frame_ready = pyqtSignal(object)
    stopped = pyqtSignal()

    def __init__(
        self,
        device_id: int,
        fps: float,
        buffer_size: int,
        grayscale: bool,
        exposure: float,
        gain: float,
        brightness: float,
        contrast: float,
        auto_exposure: bool,
    ):
        super().__init__()
        self._device = device_id
        self._fps = max(1, min(60, fps))
        self._buffer_size = max(1, min(1024, buffer_size))
        self._grayscale = grayscale
        self._exposure = exposure
        self._gain = gain
        self._brightness = brightness
        self._contrast = contrast
        self._auto_exposure = 0.75 if auto_exposure else 0.25  # 0.25=manual
        self._running = True
        self._buffer: list[np.ndarray] = []
        self._cap: Optional[cv2.VideoCapture] = None

    def run(self) -> None:
        # Windows: CAP_DSHOW often fixes webcam; fallback to default
        self._cap = cv2.VideoCapture(self._device, _CAP_BACKEND)
        if not self._cap.isOpened() and _CAP_BACKEND:
            self._cap = cv2.VideoCapture(self._device)
        if not self._cap.isOpened():
            log(f"[CAM] Cannot open camera {self._device} (try other device or backend)", color="red")
            self.stopped.emit()
            return
        # Manche Treiber brauchen explizite Aufloesung
        try:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        except Exception:
            pass
        try:
            self._cap.set(_CAP_AUTO_EXPOSURE, self._auto_exposure)
            self._cap.set(_CAP_EXPOSURE, self._exposure)
            self._cap.set(_CAP_GAIN, self._gain)
            self._cap.set(_CAP_BRIGHTNESS, self._brightness)
            self._cap.set(_CAP_CONTRAST, self._contrast)
            self._cap.set(_CAP_FPS, self._fps)
        except Exception:
            pass
        # Einige Webcams liefern erst nach ein paar Reads gueltige Frames
        for _ in range(5):
            self._cap.read()
        cap_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        cap_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        log(f"[CAM] Device {self._device}: {cap_w}x{cap_h}")
        while self._running and self._cap.isOpened():
            ret, frame = self._cap.read()
            if not ret or frame is None:
                continue
            if self._grayscale:
                f = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                f = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self._buffer.append(f.copy())
            if len(self._buffer) > self._buffer_size:
                self._buffer.pop(0)
            out = _frames_to_imagedata(np.stack(self._buffer), self._grayscale)
            self.frame_ready.emit(out)
            # 20-30 Hz: Rolling Buffer, kein echtes Live, angepasste Anzeigerate
            QThread.msleep(max(33, min(50, int(1000 / self._fps))))
        if self._cap:
            self._cap.release()
        self.stopped.emit()

    def stop(self) -> None:
        self._running = False

    def set_exposure(self, v: float) -> None:
        self._exposure = v
        if self._cap and self._cap.isOpened():
            try:
                self._cap.set(_CAP_EXPOSURE, v)
            except Exception:
                pass

    def set_gain(self, v: float) -> None:
        self._gain = v
        if self._cap and self._cap.isOpened():
            try:
                self._cap.set(_CAP_GAIN, v)
            except Exception:
                pass


class RealCameraHandler(QObject):
    """Streams from real camera. Supports exposure, gain, etc. (camera-dependent)."""

    frame_ready = pyqtSignal(object)
    stopped = pyqtSignal()

    def __init__(
        self,
        parent: Optional[QObject] = None,
        device_id: int = 0,
        fps: float = 25.0,
        buffer_size: int = 32,
        grayscale: bool = True,
        exposure: float = 0.5,
        gain: float = 0.5,
        brightness: float = 0.5,
        contrast: float = 0.5,
        auto_exposure: bool = True,
    ):
        super().__init__(parent)
        self._device = device_id
        self._fps = fps
        self._buffer_size = buffer_size
        self._grayscale = grayscale
        self._exposure = exposure
        self._gain = gain
        self._brightness = brightness
        self._contrast = contrast
        self._auto_exposure = auto_exposure
        self._thread: Optional[QThread] = None
        self._worker: Optional[_CameraWorker] = None

    def start(self) -> None:
        if self._thread is not None and self._thread.isRunning():
            return
        self._thread = QThread()
        self._worker = _CameraWorker(
            self._device,
            self._fps,
            self._buffer_size,
            self._grayscale,
            self._exposure,
            self._gain,
            self._brightness,
            self._contrast,
            self._auto_exposure,
        )
        self._worker.moveToThread(self._thread)
        self._worker.frame_ready.connect(self.frame_ready.emit)
        self._worker.stopped.connect(self._on_worker_stopped)
        self._thread.started.connect(self._worker.run)
        self._thread.start()

    def stop(self) -> None:
        if self._worker:
            self._worker.stop()

    def set_exposure(self, v: float) -> None:
        self._exposure = v
        if self._worker:
            self._worker.set_exposure(v)

    def set_gain(self, v: float) -> None:
        self._gain = v
        if self._worker:
            self._worker.set_gain(v)

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
