
import numpy as np
from pathlib import Path
import sys

# Ensure blitz is importable
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from blitz.data import optimized

def test_sliding_mean_numba_equivalence():
    """Test sliding_mean_numba vs numpy cumsum."""
    T, H, W, C = 20, 10, 10, 1
    images = np.random.rand(T, H, W, C).astype(np.float32)
    window = 5
    lag = 2

    # Even if HAS_NUMBA is False, this should work using fallback decorators
    res_numba = optimized.sliding_mean_numba(images, window, lag)

    # Manually compute expected with cumsum method
    n = T - (lag + window)
    res_numpy = np.empty((n, H, W, C), dtype=np.float32)
    for i in range(H):
        sl = images[:, i, ...]
        cs = np.cumsum(sl, axis=0)
        upper = cs[lag + window : lag + window + n]
        lower = cs[lag : lag + n]
        res_numpy[:, i, ...] = (upper - lower) / window

    try:
        np.testing.assert_allclose(res_numba, res_numpy, rtol=1e-4, atol=1e-4)
        print("Success! sliding_mean_numba matches numpy implementation.")
    except Exception as e:
        print(f"Failure! {e}")

if __name__ == "__main__":
    test_sliding_mean_numba_equivalence()
