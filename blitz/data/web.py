import os
from pathlib import Path

import requests
import socketio
from PyQt5.QtCore import QObject, QThread, pyqtSignal
from requests.exceptions import ConnectTimeout
from socketio.exceptions import ConnectionError, TimeoutError

from ..settings import get
from ..tools import log
from .image import ImageData
from .load import DataLoader


class _WebSocket(QObject):

    message_received = pyqtSignal(object)

    def __init__(self, target_address: str) -> None:
        super().__init__()
        self._target = target_address

    def listen(self) -> None:
        log("BLITZ listening to incoming data")
        with socketio.SimpleClient() as sio:
            attempts = 0
            while not sio.connected and attempts < get("web/connect_attempts"):
                try:
                    sio.connect(
                        self._target.replace("http", "ws"),
                        wait_timeout=2,
                    )
                except ConnectionError:
                    log(
                        "Unable to connect to server, "
                        f"Attempt {attempts+1}/{get('web/connect_attempts')}"
                    )
                    attempts += 1
            if not sio.connected:
                log("Server cannot be reached, aborting")
                self.message_received.emit(None)
                return
            try:
                message = sio.receive(
                    timeout=get("web/timeout_server_message"),
                )
            except TimeoutError:
                log(
                    "No message received, "
                    f"Attempt {attempts+1}/{get('web/connect_attempts')}"
                )
            else:
                if message[0] == "send_file_message":
                    self.message_received.emit(message[1]["file_name"])
                else:
                    print(message)
                    log("Unknown server message, aborting")
        self.message_received.emit(None)


class _WebDownloader(QObject):

    download_finished = pyqtSignal(object)

    def __init__(self, target_address: str) -> None:
        super().__init__()
        self._target = target_address

    def download(self) -> None:
        print("downloading")
        response = None
        attempts = 0
        while attempts < get("web/download_attempts"):
            try:
                response = requests.get(self._target, timeout=2)
            except ConnectTimeout:
                log(
                    "Unable to connect to server, "
                    f"Attempt {attempts+1}/{get('web/download_attempts')}"
                )
                attempts += 1
            else:
                break
        if response is not None and response.status_code == 200:
            cache_file = Path("cache.npy")
            c = 0
            while cache_file.exists():
                pref = cache_file.stem.split("_")[0]
                cache_file = Path(f"{pref}_{c}{cache_file.suffix}")
                c += 1
            with open(cache_file, 'wb') as f:
                f.write(response.content)
            self.download_finished.emit(cache_file)
        elif response is None:
            log("Server cannot be reached, aborting")
        else:
            log("No such file at server address, aborting")
        self.download_finished.emit(None)


class WebDataLoader(QObject):

    image_received = pyqtSignal(ImageData)

    def __init__(self, target_address: str, token: str) -> None:
        super().__init__()
        self._target = target_address
        self._token = token
        self._connect_thread = QThread()
        self._download_thread = QThread()

    def _start_connect(self) -> None:
        self._socket = _WebSocket(self._target)
        self._socket.moveToThread(self._connect_thread)
        self._connect_thread.started.connect(self._socket.listen)
        self._socket.message_received.connect(self._finish_connect)
        self._connect_thread.start()

    def _finish_connect(self, file_name: str | None) -> None:
        print(f"finished connect: {file_name!r}")
        self._connect_thread.quit()
        self._connect_thread.wait()
        if file_name is not None:
            self._start_download(file_name)

    def _start_download(self, file_name: str):
        target = self._target
        if not target.endswith("/"):
            target += "/"
        target += f"getfile/{file_name}"
        target += f"?token={self._token}"

        self._downloader = _WebDownloader(target)
        self._downloader.moveToThread(self._download_thread)
        self._download_thread.started.connect(self._downloader.download)
        self._downloader.download_finished.connect(self._finish_download)
        self._download_thread.start()

    def _finish_download(self, path: Path | None) -> None:
        self._download_thread.quit()
        self._download_thread.wait()
        if path is not None:
            img = DataLoader().load(path)
            os.remove(path)
            self.image_received.emit(img)

    def start(self) -> None:
        self._start_connect()
