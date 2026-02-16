import numpy as np
from blitz.data.tools import sliding_mean_normalization

def test_sliding_mean_normalization():
    # Create a dummy image sequence (T, H, W, C)
    # Shape: (10, 2, 2, 1)
    images = np.arange(40, dtype=np.float32).reshape((10, 2, 2, 1))

    window = 2
    lag = 1

    # n = images.shape[0] - (lag + window) = 10 - (1 + 2) = 7
    result = sliding_mean_normalization(images, window, lag)

    assert result.shape == (7, 2, 2, 1)

    # For t=0:
    # mean = (images[0+1+1] + images[0+2+1]) / 2 = (images[2] + images[3]) / 2
    expected_t0 = (images[2] + images[3]) / 2.0
    np.testing.assert_array_almost_equal(result[0], expected_t0)

    # For t=6:
    # mean = (images[6+1+1] + images[6+2+1]) / 2 = (images[8] + images[9]) / 2
    expected_t6 = (images[8] + images[9]) / 2.0
    np.testing.assert_array_almost_equal(result[6], expected_t6)
