"""
Benchmark: Video Load (single-threaded).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from blitz.data.load import DataLoader


if __name__ == "__main__":
    path = Path(r"D:\Daten_Speicher\BLITZ_testfiles\Video\VID_20230817_155944305.mp4")
    n_runs = 2
    subset_ratio = 0.1
    max_ram = 32.0

    if not path.exists():
        print(f"File not found: {path}")
        sys.exit(1)

    loader = DataLoader(size_ratio=1.0, subset_ratio=subset_ratio, grayscale=False, max_ram=max_ram)
    times = []
    for run in range(n_runs):
        t0 = time.perf_counter()
        data = loader.load(path)
        t1 = time.perf_counter()
        times.append(t1 - t0)
        size_gb = data.image.nbytes / (1024**3)
        print(f"Run {run+1}: {times[-1]:.1f}s, size={size_gb:.2f} GB")
    print(f"\nAvg: {sum(times)/len(times):.1f}s")
