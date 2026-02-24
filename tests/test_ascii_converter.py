import sys
from unittest.mock import MagicMock

# Mock modules BEFORE importing the application code
sys.modules['PyQt6'] = MagicMock()
sys.modules['PyQt6.QtCore'] = MagicMock()
sys.modules['PyQt6.QtWidgets'] = MagicMock()
sys.modules['PyQt6.QtGui'] = MagicMock()
sys.modules['psutil'] = MagicMock()
sys.modules['pyqtgraph'] = MagicMock()
sys.modules['blitz.settings'] = MagicMock()
sys.modules['blitz.layout'] = MagicMock()

import pytest
import numpy as np
from blitz.data.converters.ascii import first_col_looks_like_row_number

def test_first_col_starts_at_zero():
    data = np.array([
        [0, 10, 20],
        [1, 15, 25],
        [2, 20, 30]
    ])
    assert first_col_looks_like_row_number(data) is True

def test_first_col_starts_at_one():
    data = np.array([
        [1, 10, 20],
        [2, 15, 25],
        [3, 20, 30]
    ])
    assert first_col_looks_like_row_number(data) is True

def test_first_col_not_strictly_ascending():
    data = np.array([
        [0, 10, 20],
        [0, 15, 25],
        [1, 20, 30]
    ])
    assert first_col_looks_like_row_number(data) is False

def test_first_col_with_gaps():
    data = np.array([
        [0, 10, 20],
        [2, 15, 25],
        [3, 20, 30]
    ])
    assert first_col_looks_like_row_number(data) is False

def test_first_col_starts_elsewhere():
    data = np.array([
        [5, 10, 20],
        [6, 15, 25],
        [7, 20, 30]
    ])
    assert first_col_looks_like_row_number(data) is False

def test_single_column():
    data = np.array([
        [0],
        [1],
        [2]
    ])
    assert first_col_looks_like_row_number(data) is False

def test_single_row():
    data = np.array([
        [0, 10, 20]
    ])
    assert first_col_looks_like_row_number(data) is False

def test_float_indices():
    data = np.array([
        [0.0, 10.0],
        [1.0, 20.0],
        [2.0, 30.0]
    ])
    assert first_col_looks_like_row_number(data) is True

def test_non_sequential():
    data = np.array([
        [0, 10],
        [1, 20],
        [5, 30]
    ])
    assert first_col_looks_like_row_number(data) is False
