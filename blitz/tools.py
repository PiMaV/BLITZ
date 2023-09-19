import textwrap
from typing import Sequence

import numpy as np
import psutil

MAX_LINE_LENGTH = 37


def insert_line_breaks(text: str) -> str:
    lines = []
    line = ""
    length = 0

    for part in text.split('/'):
        if length + len(part) + 1 <= MAX_LINE_LENGTH:  # +1 for the '/'
            line = f"{line}/{part}" if line else part  # Avoid leading '/'
            length += len(part) + 1  # +1 for the '/'
        else:
            lines.append(f"{line}/")
            line = part
            length = len(part)

    lines.append(line)
    return '\n'.join(lines)


def get_available_ram() -> float:
    available_ram = psutil.virtual_memory().available / (1024**3)
    return available_ram


def wrap_text(text: str, max_length: int) -> str:
    return '\n'.join(textwrap.wrap(text, max_length))


def format_pixel_value(
    value: str | Sequence[float | int] | np.ndarray | None,
) -> str:
    if isinstance(value, str) or value is None:
        return f"Str:{value}"
    elif isinstance(value, (list, tuple, np.ndarray)) and len(value) == 3:
        # RGB, format each value individually
        return f"R:{int(value[0]):3d} G:{int(value[1]):3d} B:{int(value[2]):3d}"
    else:
        # grayscale or single value
        try:
            return f"Val:{int(value):4d}"  # type: ignore
        except (ValueError, TypeError):
            return "Invalid value"
