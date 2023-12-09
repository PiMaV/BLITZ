from typing import Any

from PyQt5.QtCore import QSettings

SETTINGS = None


_default_settings = {
    "window/ratio": 0.75,
    "window/relative_size": 0.85,
    "window/LUT_vertical_ratio": 0.6,

    "viewer/ROI_on_drop_threshold": 500_000,
}


def get(setting: str) -> Any:
    global SETTINGS

    if SETTINGS is None:
        SETTINGS = QSettings("./settings.ini", QSettings.Format.IniFormat)

    return SETTINGS.value(
        setting,
        _default_settings[setting],
        type(_default_settings[setting]),
    )


def set(setting: str, value: Any) -> None:
    global SETTINGS

    type_ = type(_default_settings[setting])
    if type(value) != type_:
        raise ValueError(
            f"Setting '{setting}' of type {type_} was given as incorrect type "
            f"{type(value)}"
        )

    if SETTINGS is None:
        SETTINGS = QSettings("./settings.ini", QSettings.Format.IniFormat)

    SETTINGS.setValue(setting, value)
