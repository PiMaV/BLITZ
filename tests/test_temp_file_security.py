import os
import sys
import tempfile
import types
from pathlib import Path
from unittest.mock import MagicMock

# Mock PyQt5 more realistically
class MockQObject:
    def __init__(self, *args, **kwargs):
        pass
    def moveToThread(self, thread):
        pass

def mock_pyqtSignal(*args, **kwargs):
    return MagicMock()

def mock_module(name):
    return types.ModuleType(name)

def _install_mocks():
    """Install sys.modules mocks; returns dict of original values for cleanup."""
    pyqt5_core = MagicMock()
    pyqt5_core.QObject = MockQObject
    pyqt5_core.pyqtSignal = mock_pyqtSignal
    pyqt5_core.QThread = MagicMock

    orig = {}
    mods = [
        ('PyQt5', MagicMock()),
        ('PyQt5.QtCore', pyqt5_core),
        ('PyQt5.QtWidgets', MagicMock()),
        ('PyQt5.QtGui', MagicMock()),
        ('requests', MagicMock()),
        ('requests.exceptions', MagicMock()),
        ('socketio', MagicMock()),
        ('socketio.exceptions', MagicMock()),
        ('cv2', MagicMock()),
        ('numpy', MagicMock()),
        ('pydicom', MagicMock()),
        ('natsort', MagicMock()),
    ]
    for name, mock in mods:
        orig[name] = sys.modules.get(name)
        sys.modules[name] = mock

    req_exc = sys.modules['requests.exceptions']
    req_exc.ConnectTimeout = type('ConnectTimeout', (Exception,), {})
    sio_exc = sys.modules['socketio.exceptions']
    sio_exc.ConnectionError = type('ConnectionError', (Exception,), {})
    sio_exc.TimeoutError = type('TimeoutError', (Exception,), {})

    for name in ('blitz.settings', 'blitz.tools', 'blitz.data.load', 'blitz.data.tools', 'blitz.layout'):
        orig[name] = sys.modules.get(name)

    settings = mock_module('blitz.settings')
    settings.get = MagicMock(return_value=3)
    sys.modules['blitz.settings'] = settings

    log_mod = mock_module('blitz.tools')
    log_mod.log = MagicMock()
    sys.modules['blitz.tools'] = log_mod

    load_mod = mock_module('blitz.data.load')
    load_mod.DataLoader = MagicMock()
    sys.modules['blitz.data.load'] = load_mod

    sys.modules['blitz.data.tools'] = mock_module('blitz.data.tools')
    sys.modules['blitz.layout'] = mock_module('blitz.layout')

    # blitz.data.web often pre-loaded with real PyQt5; force reload with mocks
    sys.modules.pop('blitz.data.web', None)

    return orig

def _restore_mocks(orig):
    """Restore sys.modules to original state."""
    for name, mod in orig.items():
        if mod is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = mod
    sys.modules.pop('blitz.data.web', None)

def test_download_file_location():
    orig = _install_mocks()
    try:
        requests_mock = sys.modules['requests']

        from blitz.data.web import _WebDownloader

        target_address = "http://example.com/data"
        downloader = _WebDownloader(target_address)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b"fake data"

        requests_mock.get.return_value = mock_response

        downloader.download()

        mock_emit = downloader.download_finished.emit
        if not mock_emit.called:
            raise Exception("emit was not called")

        emitted_path = mock_emit.call_args[0][0]
        assert emitted_path is not None

        is_in_cwd = Path(emitted_path).parent == Path(".")
        temp_dir = Path(tempfile.gettempdir())
        is_in_temp_dir = Path(emitted_path).parent.resolve() == temp_dir.resolve()

        assert emitted_path.suffix == ".npy"

        if emitted_path.exists():
            os.remove(emitted_path)

        assert not is_in_cwd and is_in_temp_dir
    finally:
        _restore_mocks(orig)

if __name__ == "__main__":
    try:
        test_download_file_location()
        print("Security fix verified: File created in secure temp directory.")
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
