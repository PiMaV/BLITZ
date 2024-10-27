from pathlib import Path
from typing import Any, Optional, Callable

from PyQt5.QtCore import QSettings
from PyQt5.QtWidgets import QFileDialog

from .tools import log

SETTINGS: "_Settings" = None  # type: ignore


_default_settings = {
    "window/relative_size": 0.85,
    "window/docks": {},

    "viewer/ROI_on_drop_threshold": 500_000,
    "viewer/LUT": {},
    "viewer/font_size_status_bar": 10,
    "viewer/font_size_log": 9,
    "viewer/max_file_name_length": 40,
    "viewer/intersection_point_size": 10,

    "default/multicore_size_threshold": 1.3 * (2**30),
    "default/multicore_files_threshold": 333,
    "default/load_8bit": False,
    "default/load_grayscale": True,
    "default/size_ratio": 1.0,
    "default/subset_ratio": 1.0,
    "default/max_ram": 0.1,
    "default/measure_tool_pixels": 1,
    "default/measure_tool_au": 1.0,
    "default/isocurve_smoothing": 3,

    "data/sync": True,
    "data/path": "",
    "data/mask": (),
    "data/cropped": (),
    "data/flipped_x": False,
    "data/flipped_y": False,
    "data/transposed": False,

    "web/address": "",
    "web/token": "",
    "web/connect_attempts": 3,
    "web/connect_timeout": 1,
    "web/download_attempts": 3,

    "app/restart_exit_code": -12341234,
}


class _Settings:

    def __init__(self) -> None:
        self._path = Path.cwd()
        self._file = "_cache.blitz"
        self.prevent_deletion = False
        self.select_ini()

    @property
    def path(self) -> Path:
        return self._path

    @path.setter
    def path(self, path: Path) -> None:
        self.prevent_deletion = True
        self._path = path
        self.select_ini()

    @property
    def file(self) -> str:
        return self._file

    @file.setter
    def file(self, file: Path) -> None:
        self.prevent_deletion = True
        self._file = file.name
        self.path = file.parent
        self.select_ini()

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
            try:
                value = type_(value)
            except:
                raise ValueError(
                    f"Setting '{setting}' of type {type_} was given as "
                    f"incorrect type {type(value)}"
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


def export(path: Optional[Path] = None) -> None:
    global SETTINGS

    if SETTINGS is None:
        SETTINGS = _Settings()

    if path is None:
        path_, _ = QFileDialog.getSaveFileName(
            caption="Save project file",
            directory=str(SETTINGS.path),
            filter="BLITZ project file (*.blitz)",
        )
        if not path_.endswith(".blitz"):
            path_ += ".blitz"
        path = Path(path_)
    SETTINGS.file = path
    SETTINGS.write_all()


def select() -> bool:
    global SETTINGS
    file, _ = QFileDialog.getOpenFileName(
        caption="Select project file",
        directory=str(SETTINGS.path),
        filter="BLITZ project file (*.blitz)",
    )
    if file:
        new_settings(Path(file))
        return True
    return False


def new_settings(file: Optional[Path] = None) -> None:
    global SETTINGS
    if SETTINGS is None:
        SETTINGS = _Settings()
    elif file is not None:
        clean_up()
    if file is not None:
        SETTINGS.file = file


def clean_up() -> None:
    global SETTINGS

    if (not SETTINGS.prevent_deletion
            and (SETTINGS.path / SETTINGS.file).exists()):
        (SETTINGS.path / SETTINGS.file).unlink()


def connect_sync(
    signal,
    value_getter: Callable,
    value_setter: Callable,
    setting: str,
    manipulator: Optional[Callable] = None,
    manipulator_target = None,
    *,
    rule_out_default: Optional[Any] = None,
) -> None:
    signal.connect(lambda: set(setting, value_getter()))
    if rule_out_default is not None and get(setting) == rule_out_default:
        return
    if ((get(setting) != _default_settings[setting])
            or get(setting) != value_getter()):
        try:
            value_setter(get(setting))
        except:
            log(f"Failed to load setting {setting!r} from the .blitz file",
                color="red")
    if manipulator is not None and get(setting) == manipulator_target:
        manipulator()
