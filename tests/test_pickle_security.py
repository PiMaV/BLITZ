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

from blitz.data.load import DataLoader

def test_np_load_calls_with_allow_pickle_false():
    """Verify that DataLoader calls np.load with allow_pickle=False."""

    loader = DataLoader()
    path = Path("fake_path.npy")

    # Mock return value of np.load to avoid errors in subsequent code
    mock_array = MagicMock()
    mock_array.ndim = 2
    mock_array.__len__.return_value = 1
    mock_array.shape = (1, 10, 10)
    mock_array.dtype = MagicMock()
    mock_array.dtype.itemsize = 4
    mock_array.nbytes = 400

    # Mock array[0]
    frame = MagicMock()
    frame.ndim = 2
    frame.squeeze.return_value = frame
    frame.shape = (10, 10)
    frame.dtype = MagicMock()
    frame.dtype.itemsize = 4
    mock_array.__getitem__.return_value = frame

    mock_np.load.return_value = mock_array

    # Test _load_array
    with patch('blitz.data.load.resize_and_convert_to_8_bit', return_value=frame), \
         patch('blitz.data.load.os.path.getsize', return_value=1000), \
         patch('blitz.data.load.ImageData', return_value=MagicMock()):
        try:
            loader._load_array(path)
        except Exception as e:
            # We don't care if it fails later, as long as np.load was called
            print(f"Caught expected-ish exception: {e}")

    # Verify np.load was called with allow_pickle=False
    args, kwargs = mock_np.load.call_args
    assert kwargs['allow_pickle'] == False

def test_load_single_array_calls_with_allow_pickle_false():
    """Verify that _load_single_array calls np.load with allow_pickle=False."""
    loader = DataLoader()
    path = Path("fake_path_single.npy")

    mock_array = MagicMock()
    mock_array.ndim = 2
    mock_array.shape = (10, 10)
    mock_array.dtype = MagicMock()
    mock_array.dtype.itemsize = 4
    mock_array.nbytes = 400
    mock_np.load.return_value = mock_array

    with patch('blitz.data.load.resize_and_convert_to_8_bit', return_value=mock_array), \
         patch('blitz.data.load.os.path.getsize', return_value=1000):
        loader._load_single_array(path)

    args, kwargs = mock_np.load.call_args
    assert kwargs['allow_pickle'] == False
