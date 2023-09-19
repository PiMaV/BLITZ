from dataclasses import dataclass
from typing import Any, Optional

from PyQt5.QtWidgets import QTextEdit


class QTextEditLogger:
    def __init__(self, text_edit: QTextEdit):
        self.text_edit = text_edit


@dataclass
class Stats:
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    mean: Optional[float] = None
    std: Optional[float] = None

    def clear(self):
        self.min_val = None
        self.max_val = None
        self.mean = None
        self.std = None
