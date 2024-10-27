import cv2
import numpy as np
from PyQt5.QtCore import QObject, QThread, pyqtSignal

from ..tools import log
from .load import DataLoader


class CamWatcher(QObject):

    on_next_image = pyqtSignal(object)

    def __init__(
        self,
        cam: int,
        buffer: int,
        frame_rate: int,
        grayscale: bool,
    ) -> None:
        super().__init__()
        self.cam = cv2.VideoCapture(cam)
        if self.cam is None or not self.cam.isOpened():
            self.is_available = False
            log(f"Error: camera {cam} not found", color="red")
        else:
            self.is_available = True
        width = int(self.cam.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._index = 0
        self.grayscale = grayscale
        self.frame_rate = frame_rate
        if self.grayscale:
            self.output = np.zeros((buffer, width, height))
        else:
            self.output = np.zeros((buffer, width, height, 3))
        self.watching = True

    def watch(self) -> None:
        while self.watching:
            _, frame = self.cam.read()
            self.output[:-1] = self.output[1:]
            if self.grayscale:
                frame: np.ndarray = np.sum(
                    frame * np.array([0.2989, 0.5870, 0.1140]),
                    axis=-1,
                )
            self.output[-1] = frame.swapaxes(0, 1)
            self.on_next_image.emit(DataLoader.from_array(self.output))
            QThread.msleep(self.frame_rate)


class LiveView(QObject):

    on_next_image = pyqtSignal(object)

    def __init__(
        self,
        cam: int,
        buffer: int,
        frame_rate: int,
        grayscale: bool,
    ) -> None:
        super().__init__()
        self._watcher = CamWatcher(cam, buffer, frame_rate, grayscale)
        self._reader_thread = QThread()

    @property
    def available(self) -> bool:
        return self._watcher.is_available

    def send(self, img: np.ndarray) -> None:
        self.on_next_image.emit(img)

    def start(self) -> None:
        self._watcher.moveToThread(self._reader_thread)
        self._reader_thread.started.connect(self._watcher.watch)
        self._watcher.on_next_image.connect(self.send)
        self._reader_thread.start()

    def end(self) -> None:
        self._watcher.watching = False
        self._reader_thread.quit()
        self._reader_thread.wait()
