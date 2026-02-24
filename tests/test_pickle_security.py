import sys
from unittest.mock import MagicMock, patch
import os
from pathlib import Path
import pytest

# Mock modules BEFORE importing the application code
sys.modules['PyQt6'] = MagicMock()
sys.modules['PyQt6.QtCore'] = MagicMock()
sys.modules['PyQt6.QtWidgets'] = MagicMock()
sys.modules['PyQt6.QtGui'] = MagicMock()
sys.modules['psutil'] = MagicMock()
sys.modules['pyqtgraph'] = MagicMock()
sys.modules['blitz.settings'] = MagicMock()
sys.modules['blitz.layout'] = MagicMock()
sys.modules['cv2'] = MagicMock()
sys.modules['natsort'] = MagicMock()
sys.modules['numba'] = MagicMock()

# Mock numpy
mock_np = MagicMock()
sys.modules['numpy'] = mock_np
# Mock some numpy constants
mock_np.float32 = MagicMock()
mock_np.newaxis = None

from blitz.data.load import DataLoader

def test_np_load_calls_with_allow_pickle_false_by_default():
    """Verify that DataLoader calls np.load with allow_pickle=False by default."""

    loader = DataLoader() # default allow_pickle=False
    path = Path("fake_path.npy")

    mock_array = MagicMock()
    mock_array.ndim = 2
    mock_array.__len__.return_value = 1
    mock_array.shape = (1, 10, 10)
    mock_array.dtype = MagicMock()
    mock_array.dtype.itemsize = 4
    mock_array.nbytes = 400

    frame = MagicMock()
    type(frame).ndim = 2 # Set it on the type or use property mock if needed, but this usually works
    frame.ndim = 2
    frame.squeeze.return_value = frame
    frame.shape = (10, 10)
    frame.nbytes = 400
    mock_array.__getitem__.return_value = frame
    mock_array.squeeze.return_value = frame

    mock_np.load.return_value = mock_array

    with patch('blitz.data.load.resize_and_convert_to_8_bit', return_value=frame), \
         patch('blitz.data.load.os.path.getsize', return_value=1000), \
         patch('blitz.data.load.ImageData', return_value=MagicMock()), \
         patch('blitz.data.load.log', MagicMock()):
        try:
            loader._load_array(path)
        except Exception as e:
            print(f"Caught exception: {e}")

    args, kwargs = mock_np.load.call_args
    assert kwargs['allow_pickle'] == False

def test_np_load_calls_with_allow_pickle_true_if_requested():
    """Verify that DataLoader calls np.load with allow_pickle=True if requested."""

    loader = DataLoader(allow_pickle=True)
    path = Path("fake_path_pickle.npy")

    mock_array = MagicMock()
    mock_array.ndim = 2
    mock_array.__len__.return_value = 1
    mock_array.shape = (1, 10, 10)
    mock_array.dtype = MagicMock()
    mock_array.dtype.itemsize = 4
    mock_array.nbytes = 400

    frame = MagicMock()
    frame.ndim = 2
    frame.squeeze.return_value = frame
    frame.shape = (10, 10)
    frame.nbytes = 400
    mock_array.__getitem__.return_value = frame
    mock_array.squeeze.return_value = frame

    mock_np.load.return_value = mock_array

    with patch('blitz.data.load.resize_and_convert_to_8_bit', return_value=frame), \
         patch('blitz.data.load.os.path.getsize', return_value=1000), \
         patch('blitz.data.load.ImageData', return_value=MagicMock()), \
         patch('blitz.data.load.log', MagicMock()):
        try:
            loader._load_array(path)
        except Exception as e:
            print(f"Caught exception: {e}")

    args, kwargs = mock_np.load.call_args
    assert kwargs['allow_pickle'] == True

def test_load_single_array_honors_allow_pickle():
    """Verify that _load_single_array honors the allow_pickle setting."""
    loader = DataLoader(allow_pickle=True)
    path = Path("fake_path_single.npy")

    mock_array = MagicMock()
    mock_array.ndim = 2
    mock_array.shape = (10, 10)
    mock_array.dtype = MagicMock()
    mock_array.dtype.itemsize = 4
    mock_array.nbytes = 400
    mock_np.load.return_value = mock_array

    with patch('blitz.data.load.resize_and_convert_to_8_bit', return_value=mock_array), \
         patch('blitz.data.load.os.path.getsize', return_value=1000), \
         patch('blitz.data.load.log', MagicMock()):
        loader._load_single_array(path)

    args, kwargs = mock_np.load.call_args
    assert kwargs['allow_pickle'] == True
