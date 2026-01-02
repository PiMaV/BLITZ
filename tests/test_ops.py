import numpy as np
import pytest
from blitz.data.ops import ReduceDict

def test_reduce_operations():
    # Create a simple 3D array (Time, H, W)
    data = np.array([
        [[1, 2], [3, 4]],
        [[5, 6], [7, 8]]
    ])

    reducer = ReduceDict()

    # Test MEAN
    mean_res = reducer.reduce(data, "MEAN")
    expected_mean = np.array([[[3., 4.], [5., 6.]]])
    np.testing.assert_array_equal(mean_res, expected_mean)

    # Test MAX
    max_res = reducer.reduce(data, "MAX")
    expected_max = np.array([[[5, 6], [7, 8]]])
    np.testing.assert_array_equal(max_res, expected_max)
