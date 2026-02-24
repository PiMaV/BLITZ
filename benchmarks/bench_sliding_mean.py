
import time
import numpy as np
from pathlib import Path
import sys

# Ensure blitz is importable
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from blitz.data import optimized

def run_benchmark(T=500, H=512, W=512, C=1, window=50, lag=10, n_runs=5):
    if not optimized.HAS_NUMBA:
        print("Numba not available.")
        return

    shape = (T, H, W, C)
    images = np.random.rand(*shape).astype(np.float32)

    print(f"Shape: {shape}, window={window}, lag={lag}")

    # Warmup
    optimized.sliding_mean_numba(images, window, lag)

    times = []
    for i in range(n_runs):
        t0 = time.perf_counter()
        res = optimized.sliding_mean_numba(images, window, lag)
        t1 = time.perf_counter()
        times.append(t1 - t0)
        print(f"Run {i+1}: {t1-t0:.4f}s")

    print(f"Average: {np.mean(times):.4f}s")

if __name__ == "__main__":
    run_benchmark()
