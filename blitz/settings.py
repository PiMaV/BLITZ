from pathlib import Path
from typing import Any

from PyQt5.QtCore import QSettings
from PyQt5.QtWidgets import QFileDialog

SETTINGS: "_Settings" = None  # type: ignore


_default_settings = {
    "window/relative_size": 0.85,
    "window/docks": {},

    "viewer/ROI_on_drop_threshold": 500_000,
    "viewer/LUT_source": "",
    "viewer/font_size_status_bar": 10,
    "viewer/max_file_name_length": 40,

    "data/multicore_size_threshold": 1.3 * (2**30),
    "data/multicore_files_threshold": 333,
    "data/max_ram": 1.0,

    "web/timeout_server_message": 180,
    "web/connect_attempts": 5,
    "web/download_attempts": 5,

    "app/restart_exit_code": -12341234,
}


class _Settings:

    def __init__(self) -> None:
        self._path = Path.cwd()
        self._file = "settings.ini"
        self.select_ini()

    @property
    def path(self) -> Path:
        return self._path

    @path.setter
    def path(self, path: Path) -> None:
        self._path = path
        self.select_ini()

    @property
    def file(self) -> str:
        return self._file

    @file.setter
    def file(self, file: Path) -> None:
        self._file = file.name
        self.path = file.parent

    def select_ini(self) -> None:
        self.settings = QSettings(
            str(self.path / self._file),
            QSettings.Format.IniFormat,
        )

    def write_all(self) -> None:
        self.settings.sync()
        for key in _default_settings.keys():
            self.settings.setValue(
                key,
                self.settings.value(
                    key,
                    _default_settings[key],
                    type(_default_settings[key]),
                ),
            )

    def __getitem__(self, setting: str) -> Any:
        self.settings.sync()
        return self.settings.value(
            setting,
            _default_settings[setting],
            type(_default_settings[setting]),
        )

    def __setitem__(self, setting: str, value: Any) -> None:
        type_ = type(_default_settings[setting])
        if type(value) != type_:
            raise ValueError(
                f"Setting '{setting}' of type {type_} was given as incorrect "
                f"type {type(value)}"
            )
        self.settings.setValue(setting, value)


def get(setting: str) -> Any:
    global SETTINGS

    if SETTINGS is None:
        SETTINGS = _Settings()

    return SETTINGS[setting]


def set(setting: str, value: Any) -> None:
    global SETTINGS

    if SETTINGS is None:
        SETTINGS = _Settings()

    SETTINGS[setting] = value


def export() -> None:
    global SETTINGS

    if SETTINGS is None:
        SETTINGS = _Settings()

    path = QFileDialog.getExistingDirectory(
        caption="Select Directory",
        directory=str(SETTINGS.path),
    )
    if path is not None:
        SETTINGS.path = Path(path)
        SETTINGS.write_all()


def select() -> None:
    global SETTINGS

    if SETTINGS is None:
        SETTINGS = _Settings()

    file, _ = QFileDialog.getOpenFileName(
        caption="Select Directory",
        directory=str(SETTINGS.path),
    )
    if file is not None:
        file = Path(file)
        SETTINGS.file = file
