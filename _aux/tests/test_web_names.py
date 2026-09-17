"""Hub file_names stamping + viewer_index emit dedup + slim status one-liner."""
from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock


class MockQObject:
    def __init__(self, *args, **kwargs):
        pass

    def moveToThread(self, thread):
        pass


def mock_pyqtSignal(*args, **kwargs):
    return MagicMock()


def _install_mocks():
    pyqt6_core = MagicMock()
    pyqt6_core.QObject = MockQObject
    pyqt6_core.pyqtSignal = mock_pyqtSignal
    pyqt6_core.QThread = MagicMock

    mods = [
        ("PyQt6", MagicMock()),
        ("PyQt6.QtCore", pyqt6_core),
        ("PyQt6.QtWidgets", MagicMock()),
        ("PyQt6.QtGui", MagicMock()),
        ("requests", MagicMock()),
        ("requests.exceptions", MagicMock()),
        ("socketio", MagicMock()),
        ("socketio.exceptions", MagicMock()),
        ("psutil", MagicMock()),
        ("pyqtgraph", MagicMock()),
        ("blitz.settings", MagicMock()),
        ("blitz.layout", MagicMock()),
        ("blitz.theme", MagicMock()),
    ]
    for name, mod in mods:
        sys.modules[name] = mod
    # Ensure package attributes for submodule imports
    sys.modules["requests"].exceptions = sys.modules["requests.exceptions"]
    sys.modules["socketio"].exceptions = sys.modules["socketio.exceptions"]


_install_mocks()

import numpy as np

from blitz.data.image import ImageData, MetaData
from blitz.data.web import WebDataLoader, apply_hub_file_names
from blitz.tools import status_one_liner_text


def _cube(n: int = 3) -> ImageData:
    arr = np.zeros((n, 4, 4, 1), dtype=np.uint8)
    meta = [
        MetaData(
            file_name=f"tmp.npy-{i}",
            file_size_MB=0.0,
            size=(4, 4),
            dtype=np.uint8,
            bit_depth=8,
            color_model="grayscale",
        )
        for i in range(n)
    ]
    return ImageData(arr, meta)


def test_apply_hub_file_names_list():
    img = _cube(2)
    apply_hub_file_names(
        img,
        file_name="__selection__.npy",
        file_names=["images/a.png", "images/b.png"],
    )
    assert [m.file_name for m in img.meta] == ["a.png", "b.png"]


def test_apply_hub_file_names_single_path():
    img = _cube(3)
    apply_hub_file_names(img, file_name="stacks/run_01.npy")
    assert all(m.file_name == "run_01.npy" for m in img.meta)


def test_apply_hub_file_names_selection_fallback():
    img = _cube(2)
    apply_hub_file_names(img, file_name="__selection__.npy")
    assert [m.file_name for m in img.meta] == ["selection[0]", "selection[1]"]


def test_emit_index_dedup():
    loader = WebDataLoader.__new__(WebDataLoader)
    loader._last_emitted_index = None
    sock = MagicMock()
    loader._socket = sock
    loader.emit_index(1)
    loader.emit_index(1)
    loader.emit_index(2)
    assert sock.queue_emit.call_count == 2
    assert sock.queue_emit.call_args_list[0][0] == ("viewer_index", {"index": 1})
    assert sock.queue_emit.call_args_list[1][0] == ("viewer_index", {"index": 2})


def test_status_one_liner_skips_multiline():
    dump = "------Dataset-------\n  |8bit         False|\n  --------------------"
    assert status_one_liner_text(dump) is None
    assert status_one_liner_text("[NET] Listening") == "[NET] Listening"
