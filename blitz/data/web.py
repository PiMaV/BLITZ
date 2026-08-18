import os
import tempfile
from pathlib import Path
from queue import Empty, Queue
from urllib.parse import urlparse

import requests
import socketio
from PyQt6.QtCore import QObject, QThread, pyqtSignal
from requests.exceptions import ConnectTimeout, RequestException
from socketio.exceptions import ConnectionError, TimeoutError

from .. import settings
from ..tools import log
from .image import ImageData
from .load import DataLoader


def _loopback_http_target(url: str) -> bool:
    host = (urlparse(url).hostname or "").lower()
    return host in {"127.0.0.1", "localhost", "::1"}


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
    download_progress = pyqtSignal(int, int)  # bytes_got, bytes_total (0 if unknown)

    def __init__(self, target_address: str) -> None:
        super().__init__()
        self._target = target_address

    def download(self) -> None:
        response = None
        attempts = 0
        max_attempts = settings.get("web/download_attempts")
        timeout = settings.get("web/download_timeout")
        while attempts < max_attempts:
            try:
                # (connect, read) — read must allow large .npy from WOLKE / sidecars
                headers = None
                if _loopback_http_target(self._target):
                    # Same machine: zip+unzip is CPU for no bandwidth. EVT gzip is
                    # opt-in anyway; this keeps WOLKE on localhost uncompressed too.
                    headers = {"Accept-Encoding": "identity"}
                response = requests.get(
                    self._target,
                    timeout=(timeout, timeout),
                    stream=True,
                    headers=headers,
                )
            except (ConnectTimeout, RequestException) as e:
                log(
                    f"[NET] Connection error: {e}, "
                    f"Attempt {attempts+1}/{max_attempts}",
                    color="orange",
                )
                attempts += 1
            else:
                break
        if response is not None and response.status_code == 200:
            total = 0
            try:
                total = int(response.headers.get("Content-Length") or 0)
            except (TypeError, ValueError):
                total = 0
            got = 0
            with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
                for chunk in _iter_response_bytes(response):
                    f.write(chunk)
                    got += len(chunk)
                    self.download_progress.emit(got, total)
                if got == 0:
                    body = response.content or b""
                    f.write(body)
                    got = len(body)
                    self.download_progress.emit(got, total)
                cache_file = Path(f.name)
            raw_mb = got / (1024 * 1024)
            enc = response.headers.get("Content-Encoding", "")
            if total:
                wire_mb = total / (1024 * 1024)
                extra = (
                    f", {enc} {wire_mb:.1f} MB on wire"
                    if enc
                    else f", {wire_mb:.1f} MB Content-Length"
                )
                log(f"[NET] Downloaded {raw_mb:.1f} MB{extra}", color="green")
            else:
                log(f"[NET] Downloaded {raw_mb:.1f} MB", color="green")
            try:
                response.close()
            except Exception:
                pass
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
        if response is not None:
            try:
                response.close()
            except Exception:
                pass
        self.download_finished.emit(None)


def _iter_response_bytes(response, chunk_size: int = 256 * 1024):
    """Yield real byte chunks; ignore non-bytes iterators (unit-test mocks)."""
    iterator = getattr(response, "iter_content", None)
    if not callable(iterator):
        return
    try:
        chunks = iterator(chunk_size=chunk_size)
    except TypeError:
        chunks = iterator()
    if chunks is None:
        return
    try:
        for chunk in chunks:
            if isinstance(chunk, (bytes, bytearray)) and chunk:
                yield bytes(chunk)
            elif not isinstance(chunk, (bytes, bytearray)):
                return
    except TypeError:
        return


class WebDataLoader(QObject):

    image_received = pyqtSignal(object, object)
    ingest_started = pyqtSignal()
    ingest_progress = pyqtSignal(int, int)  # bytes_got, bytes_total (0 if unknown)
    ingest_opening = pyqtSignal()
    ingest_failed = pyqtSignal()

    def __init__(self, target_address: str, token: str, **kwargs) -> None:
        super().__init__()
        self._target = target_address
        self._token = token
        self._connect_thread = QThread()
        self._download_thread: QThread | None = None
        self._downloader: _WebDownloader | None = None
        self._load_kwargs = kwargs
        self._selection_imagedata: ImageData | None = None
        self._pending_file_name: str | None = None
        self._pending_index: int | None = None
        self._download_busy = False

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
            # Same ImageData, new frame. MainWindow must seek, not reload.
            self.image_received.emit(self._selection_imagedata, index)
            return
        self._pending_file_name = file_name
        self._pending_index = index
        self._start_download(file_name)

    def _start_download(self, file_name: str) -> None:
        if self._download_busy:
            log("[NET] Download already in progress, skipping", color="orange")
            return
        target = self._target
        if not target.endswith("/"):
            target += "/"
        target += f"{self._token}"
        target += f"?filename={file_name}"

        # Fresh QThread per download — a finished QThread cannot be restarted
        self.ingest_started.emit()
        self._download_busy = True
        self._download_thread = QThread()
        self._downloader = _WebDownloader(target)
        self._downloader.moveToThread(self._download_thread)
        self._download_thread.started.connect(self._downloader.download)
        self._downloader.download_progress.connect(self.ingest_progress)
        self._downloader.download_finished.connect(self._finish_download)
        self._download_thread.start()

    def _finish_download(self, path: Path | None) -> None:
        thread = self._download_thread
        if thread is not None:
            thread.quit()
            thread.wait()
            thread.deleteLater()
            self._download_thread = None
        if self._downloader is not None:
            self._downloader.deleteLater()
            self._downloader = None
        self._download_busy = False
        if path is not None:
            try:
                self.ingest_opening.emit()
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
                # Keep socket listening; only a None payload from the socket
                # tears down the connection (see end_web_connection).
                self.ingest_failed.emit()
                return
            finally:
                try:
                    os.remove(path)
                except OSError:
                    log(f"[NET] Failed to remove temp file: {path}", color="orange")
        else:
            log("[NET] Download failed — still listening for next push", color="orange")
            self.ingest_failed.emit()

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
