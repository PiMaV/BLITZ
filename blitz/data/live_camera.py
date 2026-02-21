"""LiveView handler: Real camera via cv2.VideoCapture.

USB webcams, etc. Exposure, Gain, resolution, FPS request configurable.
On Windows: uses CAP_DSHOW (DirectShow) for better webcam support.

Capture runs at camera speed (no FPS sleep in loop): we read as fast as the
camera delivers so the ring buffer fills with the real frame rate. FPS setting
is only sent to the camera (CAP_PROP_FPS); actual rate depends on the driver.
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
    buffer_status = pyqtSignal(int, int)  # current, max
    stopped = pyqtSignal()

    def __init__(
        self,
        device_id: int,
        width: int,
        height: int,
        fps: float,
        buffer_size: int,
        grayscale: bool,
        exposure: float,
        gain: float,
        brightness: float,
        contrast: float,
        auto_exposure: bool,
        send_live_only: bool = True,
    ):
        super().__init__()
        self._device = device_id
        self._width = max(160, min(4096, width))
        self._height = max(120, min(2160, height))
        self._fps = fps
        self._buffer_size = max(1, min(1024, buffer_size))
        self._grayscale = grayscale
        self._exposure = exposure
        self._gain = gain
        self._brightness = brightness
        self._contrast = contrast
        self._auto_exposure = 0.75 if auto_exposure else 0.25  # 0.25=manual
        self._send_live_only = send_live_only
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
        try:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        except Exception:
            pass
        try:
            self._cap.set(_CAP_AUTO_EXPOSURE, self._auto_exposure)
            self._cap.set(_CAP_EXPOSURE, self._exposure)
            self._cap.set(_CAP_GAIN, self._gain)
            self._cap.set(_CAP_BRIGHTNESS, self._brightness)
            self._cap.set(_CAP_CONTRAST, self._contrast)
            if self._fps > 0:
                self._cap.set(_CAP_FPS, min(120.0, self._fps))
        except Exception:
            pass
        for _ in range(5):
            self._cap.read()
        cap_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        cap_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap_fps = self._cap.get(_CAP_FPS)
        log(f"[CAM] Device {self._device}: {cap_w}x{cap_h}, FPS request={self._fps}, actual FPS={cap_fps:.1f}")
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

            self.buffer_status.emit(len(self._buffer), self._buffer_size)

            if self._send_live_only and self._buffer:
                out = _frames_to_imagedata(
                    np.stack([self._buffer[-1]]), self._grayscale
                )
            else:
                out = _frames_to_imagedata(
                    np.stack(self._buffer), self._grayscale
                )
            self.frame_ready.emit(out)
            QThread.msleep(1)
        if self._buffer:
            final = _frames_to_imagedata(np.stack(self._buffer), self._grayscale)
            self.frame_ready.emit(final)
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
    buffer_status = pyqtSignal(int, int)  # current, max
    stopped = pyqtSignal()

    def __init__(
        self,
        parent: Optional[QObject] = None,
        device_id: int = 0,
        width: int = 640,
        height: int = 480,
        fps: float = 25.0,
        buffer_size: int = 32,
        grayscale: bool = True,
        exposure: float = 0.5,
        gain: float = 0.5,
        brightness: float = 0.5,
        contrast: float = 0.5,
        auto_exposure: bool = True,
        send_live_only: bool = True,
    ):
        super().__init__(parent)
        self._device = device_id
        self._width = width
        self._height = height
        self._fps = fps
        self._buffer_size = buffer_size
        self._grayscale = grayscale
        self._exposure = exposure
        self._gain = gain
        self._brightness = brightness
        self._contrast = contrast
        self._auto_exposure = auto_exposure
        self._send_live_only = send_live_only
        self._thread: Optional[QThread] = None
        self._worker: Optional[_CameraWorker] = None

    def start(self) -> None:
        if self._thread is not None and self._thread.isRunning():
            return
        self._thread = QThread()
        self._worker = _CameraWorker(
            self._device,
            self._width,
            self._height,
            self._fps,
            self._buffer_size,
            self._grayscale,
            self._exposure,
            self._gain,
            self._brightness,
            self._contrast,
            self._auto_exposure,
            self._send_live_only,
        )
        self._worker.moveToThread(self._thread)
        self._worker.frame_ready.connect(self.frame_ready.emit)
        self._worker.buffer_status.connect(self.buffer_status.emit)
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
