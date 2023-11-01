import textwrap
from typing import Any, Sequence

import numpy as np
import psutil

LOGGER: Any = None


def log(message: str) -> None:
    if LOGGER is None:
        print(message)
    else:
        LOGGER.log(message)


def set_logger(logger: Any) -> None:
    global LOGGER
    LOGGER = logger


def get_available_ram() -> float:
    available_ram = psutil.virtual_memory().available / (1024**3)
    return available_ram


def wrap_text(text: str, max_length: int) -> str:
    return '\n'.join(textwrap.wrap(text, max_length))


def format_pixel_value(
    value: str | Sequence[float | int] | np.ndarray | None,
) -> str:
    if isinstance(value, str) or value is None:
        return f"{value}"
    elif isinstance(value, (list, tuple, np.ndarray)) and len(value) == 3:
        return f"({int(value[0]):3d}, {int(value[1]):3d}, {int(value[2]):3d})"
    else:
        try:
            return f"{int(value):4d}"  # type: ignore
        except (ValueError, TypeError):
            return "Invalid value"
