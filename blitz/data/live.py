import time
from collections import deque
from typing import Optional

import cv2
import numpy as np
from PyQt5.QtCore import QObject, QThread, pyqtSignal

from ..tools import log
from .image import ImageData, MetaData, VideoMetaData


class CamWatcher(QObject):
    """
    Worker class that captures frames from a camera in a separate thread.
    Maintains a ring buffer of frames.
    """

    on_next_frame = pyqtSignal(np.ndarray)  # Emits (W, H) or (W, H, C) for display
    finished = pyqtSignal()

    def __init__(
        self,
        cam_id: int,
        buffer_size: int,
        frame_interval_ms: int,
        grayscale: bool,
        downsample: float,
    ) -> None:
        super().__init__()
        self.cam_id = cam_id
        self.buffer_size = max(1, buffer_size)
        self.frame_interval_s = frame_interval_ms / 1000.0
        self.grayscale = grayscale
        self.downsample = downsample

        self.running = False
        self.buffer: Optional[np.ndarray] = None
        self._head = 0
        self._full = False
        self._cam: Optional[cv2.VideoCapture] = None

        # Metadata storage
        self._fps = 0.0
        self._frame_width = 0
        self._frame_height = 0

    def start_watching(self) -> None:
        """Starts the capture loop."""
        self._cam = cv2.VideoCapture(self.cam_id)
        if not self._cam.isOpened():
            log(f"Error: Camera {self.cam_id} could not be opened.", color="red")
            self.finished.emit()
            return

        # Get camera properties
        original_width = int(self._cam.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_height = int(self._cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._fps = self._cam.get(cv2.CAP_PROP_FPS) or 30.0

        # Calculate target size
        width = int(original_width * self.downsample)
        height = int(original_height * self.downsample)

        # Store effective dimensions
        self._frame_width = width
        self._frame_height = height

        # Initialize buffer
        # Shape: (T, H, W) or (T, H, W, 3)
        # We store in H, W format for convenience, swap for display later
        if self.grayscale:
            shape = (self.buffer_size, height, width)
            dtype = np.uint8
        else:
            shape = (self.buffer_size, height, width, 3)
            dtype = np.uint8

        self.buffer = np.zeros(shape, dtype=dtype)
        self._head = 0
        self._full = False
        self.running = True

        log(f"Started camera {self.cam_id}. Buffer: {self.buffer_size} frames. Size: {width}x{height}")

        while self.running:
            start_time = time.time()
            ret, frame = self._cam.read()
            if not ret:
                log("Error: Failed to read frame from camera.", color="red")
                break

            # Process frame
            if self.downsample != 1.0:
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

            if self.grayscale:
                if frame.ndim == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                if frame.ndim == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Store in buffer
            self.buffer[self._head] = frame
            self._head = (self._head + 1) % self.buffer_size
            if self._head == 0:
                self._full = True

            # Emit for display (PyQtGraph expects W, H, so we swap axes)
            # frame is (H, W) or (H, W, 3)
            # display_frame -> (W, H) or (W, H, 3)
            display_frame = np.swapaxes(frame, 0, 1)
            self.on_next_frame.emit(display_frame)

            # Wait for next frame
            elapsed = time.time() - start_time
            sleep_time = self.frame_interval_s - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        if self._cam:
            self._cam.release()
        self.finished.emit()

    def stop(self) -> None:
        """Stops the capture loop."""
        self.running = False

    def get_buffered_data(self) -> ImageData | None:
        """Returns the current buffer content as ImageData."""
        if self.buffer is None:
            return None

        # Reorder buffer to chronological order
        if not self._full:
            # Not full: 0 to head
            frames = self.buffer[:self._head].copy()
        else:
            # Full: head to end + 0 to head
            frames = np.concatenate((self.buffer[self._head:], self.buffer[:self._head]))

        if len(frames) == 0:
            return None

        # frames is (T, H, W) or (T, H, W, 3)
        # ImageData expects (T, W, H) or (T, W, H, 3) [swapped spatial dims]
        # So we swap axes 1 and 2
        frames_swapped = np.swapaxes(frames, 1, 2)

        # Create metadata
        # We need a list of MetaData objects, one for each frame
        metadata_list = []

        # Common metadata
        color_model = "grayscale" if self.grayscale else "rgb"
        # frames is uint8, so bit_depth is 8
        bit_depth = 8

        # We simulate file names with timestamps or indices
        # If we knew the actual time, we could use it. Here we use index.
        for i in range(len(frames)):
            # Create VideoMetaData or standard MetaData?
            # VideoMetaData has fps, frame_count, etc.
            # Let's use VideoMetaData as it fits the stream nature
            meta = VideoMetaData(
                file_name=f"live_{i:04d}",
                file_size_MB=frames[i].nbytes / (1024 * 1024),
                size=(self._frame_height, self._frame_width), # (H, W)
                dtype=np.uint8,
                bit_depth=bit_depth,
                color_model=color_model,
                fps=int(self._fps),
                frame_count=len(frames),
                reduced_frame_count=0,
                codec="LIVE",
            )
            metadata_list.append(meta)

        return ImageData(frames_swapped, metadata_list)


class LiveView(QObject):
    """
    Controller for the live camera view.
    """

    on_frame_ready = pyqtSignal(np.ndarray) # Forwarded from watcher

    def __init__(
        self,
        cam_id: int = 0,
        buffer_size: int = 100,
        frame_interval_ms: int = 33,
        grayscale: bool = False,
        downsample: float = 1.0,
    ) -> None:
        super().__init__()
        self.cam_id = cam_id
        self.buffer_size = buffer_size
        self.frame_interval_ms = frame_interval_ms
        self.grayscale = grayscale
        self.downsample = downsample

        self._thread: Optional[QThread] = None
        self._watcher: Optional[CamWatcher] = None
        self._is_running = False

    def start(self) -> None:
        if self._is_running:
            return

        self._thread = QThread()
        self._watcher = CamWatcher(
            self.cam_id,
            self.buffer_size,
            self.frame_interval_ms,
            self.grayscale,
            self.downsample,
        )

        self._watcher.moveToThread(self._thread)

        # Connect signals
        self._thread.started.connect(self._watcher.start_watching)
        self._watcher.on_next_frame.connect(self.on_frame_ready)
        self._watcher.finished.connect(self._thread.quit)
        self._watcher.finished.connect(self._watcher.deleteLater)
        self._thread.finished.connect(self._thread.deleteLater)

        self._thread.start()
        self._is_running = True

    def stop(self) -> ImageData | None:
        """Stops the live view and returns the captured buffer as ImageData."""
        if not self._is_running or not self._watcher:
            return None

        # Stop the watcher
        self._watcher.stop()

        # Wait for thread to finish
        if self._thread:
            self._thread.quit()
            self._thread.wait()

        # Retrieve data
        data = self._watcher.get_buffered_data()

        self._is_running = False
        self._watcher = None
        self._thread = None

        return data

    def is_running(self) -> bool:
        return self._is_running
