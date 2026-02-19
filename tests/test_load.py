import sys
from unittest.mock import MagicMock
import os

# Mock modules BEFORE importing the application code
sys.modules['PyQt6'] = MagicMock()
sys.modules['PyQt6.QtCore'] = MagicMock()
sys.modules['PyQt6.QtWidgets'] = MagicMock()
sys.modules['PyQt6.QtGui'] = MagicMock()
sys.modules['psutil'] = MagicMock()
sys.modules['pyqtgraph'] = MagicMock()

# Mock blitz.settings to avoid import errors there if possible
sys.modules['blitz.settings'] = MagicMock()

# Mock blitz.layout to avoid loading UI code which depends on PyQt
sys.modules['blitz.layout'] = MagicMock()

import pytest
from unittest.mock import patch
import numpy as np
from pathlib import Path

# Now import the code under test
from blitz.data.load import DataLoader
from blitz.data.image import ImageData, MetaData

class MockPath(str, os.PathLike):
    def __new__(cls, name, is_dir_val=False):
        obj = super().__new__(cls, name)
        obj.name = name
        obj.suffix = Path(name).suffix
        obj._is_dir = is_dir_val
        return obj

    def __init__(self, name, is_dir_val=False):
        pass

    def is_dir(self):
        return self._is_dir

    def __fspath__(self):
        return self.name

def test_load_folder_sorting():
    file_names = ["10.png", "1.png", "2.png"]
    mock_files = [MockPath(name) for name in file_names]

    mock_dir = MagicMock(spec=Path)
    mock_dir.is_dir.return_value = True

    # Return a new iterator each time iterdir is called
    mock_dir.iterdir.side_effect = lambda: iter(mock_files)

    with patch('blitz.data.load.DataLoader._load_image') as mock_load_image, \
         patch('blitz.data.load.settings') as mock_settings, \
         patch('blitz.data.load.os.path.getsize', return_value=1000), \
         patch('blitz.data.load.Pool') as mock_pool:

        mock_settings.get.return_value = 100000000

        def load_image_side_effect(path):
            img = np.zeros((10, 10, 3), dtype=np.uint8)
            meta = MetaData(
                file_name=path.name, # Access .name attribute
                file_size_MB=0.1,
                size=(10, 10),
                dtype=np.uint8,
                bit_depth=8,
                color_model="rgb"
            )
            return img, meta

        mock_load_image.side_effect = load_image_side_effect

        loader = DataLoader()
        # Mock crop to avoid errors if any
        loader.crop = None

        result = loader._load_folder(mock_dir)

        loaded_names = [m.file_name for m in result.meta]
        expected_names = ["1.png", "2.png", "10.png"]

        print(f"Loaded names: {loaded_names}")
        assert loaded_names == expected_names