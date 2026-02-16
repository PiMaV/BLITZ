from pathlib import Path
from typing import Any, Callable, Optional

from PyQt5.QtCore import QMetaObject, QSettings, pyqtBoundSignal

from .tools import log

_default_core_settings = {
    "window/relative_size": 0.85,
    "window/docks": {},

    "viewer/ROI_on_drop_threshold": 500_000,
    "viewer/font_size_status_bar": 10,
    "viewer/font_size_log": 9,
    "viewer/max_file_name_length": 40,
    "viewer/intersection_point_size": 10,

    "default/multicore_size_threshold": 1.3 * (2**30),
    "default/multicore_files_threshold": 333,
    "default/load_8bit": False,
    "default/load_grayscale": True,
    "default/max_ram": 2.,
    "default/colormap": "greyclip",

    "data/sync": False,

    "web/address": "",
    "web/token": "",
    "web/connect_attempts": 3,
    "web/connect_timeout": 1,
    "web/download_attempts": 3,

    "app/restart_exit_code": -12341234,
}

_default_project_settings = {
    "size_ratio": 1.0,
    "subset_ratio": 1.0,

    "path": "",
    "mask": (),
    "cropped": (),
    "flipped_x": False,
    "flipped_y": False,
    "transposed": False,

    "measure_tool_pixels": 1,
    "measure_tool_au": 1.0,
    "isocurve_smoothing": 3,
}


class _Settings:

    def __init__(self, default: dict, path: Optional[Path] = None) -> None:
        self._path = (Path.cwd() / "settings.blitz") if path is None else path
        self._default = default
        self.settings = QSettings(str(self._path), QSettings.Format.IniFormat)
        self._keep = False
        self._connections: list[
            tuple[pyqtBoundSignal, QMetaObject.Connection]
        ] = []

    def keep(self) -> None:
        self._keep = True

    def write_all(self) -> None:
        self.settings.sync()
        for key in self._default.keys():
            self.settings.setValue(
                key,
                self.settings.value(
                    key,
                    self._default[key],
                    type(self._default[key]),
                ),
            )

    def connect_sync(
        self,
        setting: str,
        signal: pyqtBoundSignal,
        value_getter: Callable,
        value_setter: Optional[Callable] = None,
    ) -> None:
        self._connections.append(
            (signal, signal.connect(
                lambda: self.__setitem__(setting, value_getter())
            ))
        )
        default = self._default[setting]
        if value_setter is not None:
            if (self[setting] != default) or self[setting] != value_getter():
                try:
                    value_setter(self[setting])
                except Exception:
                    log(f"Failed to load setting {setting!r} from "
                        f"{self._path.name}", color="red")

    def __getitem__(self, setting: str) -> Any:
        self.settings.sync()
        return self.settings.value(
            setting,
            self._default[setting],
            type(self._default[setting]),
        )

    def __setitem__(self, setting: str, value: Any) -> None:
        if (def_type := type(self._default[setting])) is not type(value):
            try:
                value = def_type(value)
            except Exception:
                raise ValueError(
                    f"Setting '{setting}' of type {def_type} was given as "
                    f"incorrect type {type(value)}"
                )
        self.settings.setValue(setting, value)

    def remove(self) -> None:
        for signal, connection in self._connections:
            signal.disconnect(connection)
        if not self._keep:
            self._path.unlink()


SETTINGS: _Settings = None  # type: ignore
PROJECT_SETTINGS: _Settings = None  # type: ignore


def create() -> Any:
    global SETTINGS

    if SETTINGS is None:
        SETTINGS = _Settings(_default_core_settings)
        SETTINGS.keep()


def create_project(path: Path) -> Any:
    global PROJECT_SETTINGS
    if PROJECT_SETTINGS is not None:
        PROJECT_SETTINGS.remove()
    PROJECT_SETTINGS = _Settings(_default_project_settings, path)
    PROJECT_SETTINGS.keep()


def get(setting: str) -> Any:
    global SETTINGS
    create()
    return SETTINGS[setting]


def get_project(setting: str) -> Any:
    global PROJECT_SETTINGS
    if PROJECT_SETTINGS is None:
        raise RuntimeError("Project settings are not selected")
    return PROJECT_SETTINGS[setting]


def set(setting: str, value: Any) -> None:
    global SETTINGS
    create()
    SETTINGS[setting] = value


def set_project(setting: str, value: Any) -> None:
    global PROJECT_SETTINGS
    if PROJECT_SETTINGS is None:
        raise RuntimeError("Project settings are not selected")
    PROJECT_SETTINGS[setting] = value


def connect_sync(
    setting: str,
    signal: pyqtBoundSignal,
    value_getter: Callable,
    value_setter: Optional[Callable] = None,
) -> None:
    global SETTINGS
    create()
    SETTINGS.connect_sync(setting, signal, value_getter, value_setter)


def connect_sync_project(
    setting: str,
    signal: pyqtBoundSignal,
    value_getter: Callable,
    value_setter: Optional[Callable] = None,
    manipulator: Optional[Callable] = None,
    manipulator_target = None,
) -> None:
    global PROJECT_SETTINGS
    if PROJECT_SETTINGS is None:
        raise RuntimeError("Project settings are not selected")
    PROJECT_SETTINGS.connect_sync(setting, signal, value_getter, value_setter)
    if manipulator is not None and get_project(setting) == manipulator_target:
        manipulator()
