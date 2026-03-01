import os
import tempfile
from pathlib import Path
from queue import Empty, Queue

import requests
import socketio
from PyQt6.QtCore import QObject, QThread, pyqtSignal
from requests.exceptions import ConnectTimeout, RequestException
from socketio.exceptions import ConnectionError, TimeoutError

from .. import settings
from ..tools import log
from .image import ImageData
from .load import DataLoader


class _WebSocket(QObject):

    message_received = pyqtSignal(object)

    def __init__(self, target_address: str) -> None:
        super().__init__()
        self._target = target_address
        self._listening = True
        self._emit_queue: Queue = Queue()
        self.sio = socketio.SimpleClient()

    @property
    def listening(self) -> bool:
        return self._listening

    @listening.setter
    def listening(self, listening: bool) -> None:
        self._listening = listening

    def queue_emit(self, event: str, data: dict) -> None:
        """Thread-safe: main thread can enqueue viewer->server messages."""
        self._emit_queue.put((event, data))

    def listen(self) -> None:
        log("[NET] Attempting to connect...")
        max_attempts = settings.get("web/connect_attempts")
        attempts = 0
        while not self.sio.connected and attempts < max_attempts:
            try:
                self.sio.connect(
                    self._target.replace("http", "ws"),
                    wait_timeout=settings.get("web/connect_timeout"),
                )
            except ConnectionError:
                log(
                    "[NET] Unable to connect, "
                    f"Attempt {attempts+1}/{max_attempts}",
                    color="red",
                )
                attempts += 1
        if not self.sio.connected:
            log("[NET] Cannot be connected, aborting", color="red")
            self.message_received.emit(None)
            return
        log("[NET] Listening to incoming data", color="green")

        while self.listening:
            while True:
                try:
                    event, data = self._emit_queue.get_nowait()
                    self.sio.emit(event, data)
                except Empty:
                    break
            try:
                message = self.sio.receive(timeout=1)
            except TimeoutError:
                pass
            else:
                if message[0] == "send_file_message":
                    self.message_received.emit(message[1])
                elif message[0] == "Connected successfully":
                    log("[NET] Connected to server", color="green")
                else:
                    log("[NET] Unknown message, aborting", color="red")


class _WebDownloader(QObject):

    download_finished = pyqtSignal(object)

    def __init__(self, target_address: str) -> None:
        super().__init__()
        self._target = target_address

    def download(self) -> None:
        response = None
        attempts = 0
        max_attempts = settings.get('web/download_attempts')
        while attempts < max_attempts:
            try:
                response = requests.get(self._target, timeout=2)
            except (ConnectTimeout, RequestException) as e:
                log(f"[NET] Connection error: {e}, "
                    f"Attempt {attempts+1}/{max_attempts}", color="orange")
                attempts += 1
            else:
                break
        if response is not None and response.status_code == 200:
            with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
                f.write(response.content)
                cache_file = Path(f.name)
            self.download_finished.emit(cache_file)
            return
        elif response is None:
            log("[NET] Cannot be reached, aborting", color="red")
        else:
            log(
                "[NET] No such file found at server: "
                f"{self._target.split('filename=')[1]}",
                color="red",
            )
        self.download_finished.emit(None)


class WebDataLoader(QObject):

    image_received = pyqtSignal(object, object)

    def __init__(self, target_address: str, token: str, **kwargs) -> None:
        super().__init__()
        self._target = target_address
        self._token = token
        self._connect_thread = QThread()
        self._download_thread = QThread()
        self._load_kwargs = kwargs
        self._selection_imagedata: ImageData | None = None
        self._pending_file_name: str | None = None
        self._pending_index: int | None = None

    def _start_listening(self) -> None:
        self._socket = _WebSocket(self._target)
        self._socket.moveToThread(self._connect_thread)
        self._connect_thread.started.connect(self._socket.listen)
        self._socket.message_received.connect(self._finish_connect)
        self._connect_thread.start()

    def _finish_connect(self, payload: dict | None) -> None:
        if payload is None:
            self.image_received.emit(None, None)
            self._connect_thread.quit()
            self._connect_thread.wait()
            return
        file_name = payload.get("file_name")
        index = payload.get("index")
        index = index if isinstance(index, int) else None
        if (
            file_name == "__selection__.npy"
            and index is not None
            and self._selection_imagedata is not None
        ):
            self.image_received.emit(self._selection_imagedata, index)
            return
        self._pending_file_name = file_name
        self._pending_index = index
        self._start_download(file_name)

    def _start_download(self, file_name: str) -> None:
        target = self._target
        if not target.endswith("/"):
            target += "/"
        target += f"{self._token}"
        target += f"?filename={file_name}"

        self._downloader = _WebDownloader(target)
        self._downloader.moveToThread(self._download_thread)
        self._download_thread.started.connect(self._downloader.download)
        self._downloader.download_finished.connect(self._finish_download)
        self._download_thread.start()

    def _finish_download(self, path: Path | None) -> None:
        self._download_thread.quit()
        self._download_thread.wait()
        if path is not None:
            try:
                img = DataLoader(**self._load_kwargs).load(path)
                if self._pending_file_name == "__selection__.npy":
                    self._selection_imagedata = img
                    display_index = (
                        self._pending_index
                        if self._pending_index is not None
                        else 0
                    )
                    self.image_received.emit(img, display_index)
                else:
                    self._selection_imagedata = None
                    self.image_received.emit(img, None)
            except Exception as e:
                log(f"[NET] Error loading downloaded file: {e}", color="red")
                self.image_received.emit(None, None)
            finally:
                try:
                    os.remove(path)
                except OSError:
                    log(f"[NET] Failed to remove temp file: {path}", color="orange")
        else:
            self.image_received.emit(None, None)

    def emit_index(self, index: int) -> None:
        """Tell WOLKE which row index is shown (BLITZ -> WOLKE sync)."""
        if getattr(self, "_socket", None) is not None:
            self._socket.queue_emit("viewer_index", {"index": index})

    def start(self) -> None:
        self._start_listening()

    def stop(self) -> None:
        self._socket.listening = False
        self._connect_thread.quit()
        self._connect_thread.wait()
