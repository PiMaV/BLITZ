import os
import threading
from pathlib import Path

import requests
import socketio
from requests.exceptions import ConnectTimeout
from socketio.exceptions import ConnectionError, TimeoutError

from ..tools import log
from ..settings import get
from .image import ImageData
from .load import DataLoader


class WebDataLoader:

    MAX_ATTEMPTS = 5

    def __init__(self, target_address: str, token: str) -> None:
        self._target = target_address
        self._token = token
        self._success = False
        self._file = ""

    def update(self) -> bool:
        thread = threading.Thread(target=self._get_file_update)
        self._success = False
        thread.start()
        thread.join()
        return self._success

    def _get_file_update(self) -> None:
        log("BLITZ listening to incoming data")
        with socketio.SimpleClient() as sio:
            attempts = 0
            while not sio.connected and attempts < self.MAX_ATTEMPTS:
                try:
                    sio.connect(
                        self._target.replace("http", "ws"),
                        wait_timeout=2,
                    )
                except ConnectionError:
                    log(
                        "Unable to connect to server, "
                        f"Attempt {attempts+1}/{self.MAX_ATTEMPTS}"
                    )
                    attempts += 1
            if not sio.connected:
                log("Server cannot be reached, aborting")
                return
            try:
                message = sio.receive(
                    timeout=get("web/timeout_server_message"),
                )
            except TimeoutError:
                log(
                    "No message received, "
                    f"Attempt {attempts+1}/{self.MAX_ATTEMPTS}"
                )
            else:
                if message[0] == "send_file_message":
                    self._file = message[1]["file_name"]
                    self._success = True
                else:
                    log("Unknown server message, aborting")

    def _build_file_address(self) -> str:
        target = self._target
        if not target.endswith("/"):
            target += "/"
        target += self._file
        target += f"?token={self._token}"
        return target

    def _get_file_cache_name(self) -> Path:
        name = Path("cache.npy")
        c = 0
        while name.exists():
            pref = name.stem.split("_")[0]
            name = Path(f"{pref}_{c}{name.suffix}")
            c += 1
        return name

    def load(self, **kwargs) -> ImageData:
        if not self.update():
            return DataLoader.from_text(
                "Connecting to server failed",
                color=(255, 0, 0),
            )

        response = None
        attempts = 0
        while attempts < self.MAX_ATTEMPTS:
            try:
                response = requests.get(self._build_file_address(), timeout=2)
            except ConnectTimeout:
                log(
                    "Unable to connect to server, "
                    f"Attempt {attempts+1}/{self.MAX_ATTEMPTS}"
                )
                attempts += 1
            else:
                break
        if response is not None and response.status_code == 200:
            cache_file = self._get_file_cache_name()
            with open(cache_file, 'wb') as f:
                f.write(response.content)
            img = DataLoader(**kwargs).load(cache_file)
            os.remove(cache_file)
            return img
        elif response is None:
            log("Server cannot be reached, aborting")
        else:
            log("No such file at server address, aborting")
        return DataLoader.from_text("Error loading data")
