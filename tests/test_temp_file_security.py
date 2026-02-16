import sys
import tempfile
from unittest.mock import MagicMock

# Mock PyQt5 more realistically
class MockQObject:
    def __init__(self, *args, **kwargs):
        pass
    def moveToThread(self, thread):
        pass

def mock_pyqtSignal(*args, **kwargs):
    return MagicMock()

pyqt5_core = MagicMock()
pyqt5_core.QObject = MockQObject
pyqt5_core.pyqtSignal = mock_pyqtSignal
pyqt5_core.QThread = MagicMock

sys.modules['PyQt5'] = MagicMock()
sys.modules['PyQt5.QtCore'] = pyqt5_core
sys.modules['PyQt5.QtWidgets'] = MagicMock()
sys.modules['PyQt5.QtGui'] = MagicMock()

# Mock other dependencies
requests_mock = MagicMock()
sys.modules['requests'] = requests_mock
req_exc = MagicMock()
sys.modules['requests.exceptions'] = req_exc
req_exc.ConnectTimeout = type('ConnectTimeout', (Exception,), {})

sio_exc = MagicMock()
sys.modules['socketio'] = MagicMock()
sys.modules['socketio.exceptions'] = sio_exc
sio_exc.ConnectionError = type('ConnectionError', (Exception,), {})
sio_exc.TimeoutError = type('TimeoutError', (Exception,), {})

sys.modules['cv2'] = MagicMock()
sys.modules['numpy'] = MagicMock()
sys.modules['pydicom'] = MagicMock()
sys.modules['natsort'] = MagicMock()

import types
def mock_module(name):
    m = types.ModuleType(name)
    sys.modules[name] = m
    return m

settings = mock_module('blitz.settings')
settings.get = MagicMock(return_value=3)

log_mod = mock_module('blitz.tools')
log_mod.log = MagicMock()

load_mod = mock_module('blitz.data.load')
load_mod.DataLoader = MagicMock()

mock_module('blitz.data.tools')
mock_module('blitz.layout')

from blitz.data.web import _WebDownloader
import os
from pathlib import Path
from unittest.mock import patch

def test_download_file_location():
    # Setup
    target_address = "http://example.com/data"
    downloader = _WebDownloader(target_address)

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.content = b"fake data"

    requests_mock.get.return_value = mock_response

    # download_finished is a MagicMock from our mock_pyqtSignal
    with patch.object(downloader.download_finished, 'emit') as mock_emit:
        downloader.download()

        if not mock_emit.called:
            # Check what happened
            print(f"requests_mock.get.called: {requests_mock.get.called}")
            raise Exception("emit was not called")

        # Get the path that was emitted
        emitted_path = mock_emit.call_args[0][0]
        assert emitted_path is not None

        print(f"File created at: {emitted_path}")

        # Check if it's in the current directory
        # Now it should be in the system temp directory, NOT the current directory
        is_in_cwd = Path(emitted_path).parent == Path(".")
        print(f"Is in current directory: {is_in_cwd}")

        temp_dir = Path(tempfile.gettempdir())
        is_in_temp_dir = Path(emitted_path).parent.resolve() == temp_dir.resolve()
        print(f"Is in system temp directory ({temp_dir}): {is_in_temp_dir}")

        # Check suffix
        assert emitted_path.suffix == ".npy"

        # Clean up
        if emitted_path.exists():
            os.remove(emitted_path)

        return is_in_cwd, is_in_temp_dir

if __name__ == "__main__":
    try:
        is_in_cwd, is_in_temp_dir = test_download_file_location()
        if not is_in_cwd and is_in_temp_dir:
            print("Security fix verified: File created in secure temp directory.")
        elif is_in_cwd:
            print("Security fix failed: File still created in current directory.")
            exit(1)
        else:
            print(f"Unexpected results: is_in_cwd={is_in_cwd}, is_in_temp_dir={is_in_temp_dir}")
            exit(1)
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
