"""Benchmark LUT percentile (nanpercentile) vs min/max (nanmin/nanmax).

Run: python benchmarks/bench_lut_percentile.py

On scroll, _levels_cache is used so percentile is not recomputed.
This bench measures the cold path (load, crop, percentile change).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))


def _nanminmax(img: np.ndarray) -> tuple[float, float]:
    """Current min/max path (percentile=0)."""
    return float(np.nanmin(img)), float(np.nanmax(img))


def _nanpercentile_1pct(img: np.ndarray) -> tuple[float, float]:
    """LUT percentile 1% path."""
    mn, mx = np.nanpercentile(img, [1.0, 99.0])
    return float(mn), float(mx)


def run_benchmark(
    T: int = 100,
    H: int = 512,
    W: int = 512,
    n_warmup: int = 2,
    n_runs: int = 20,
) -> None:
    """Compare nanpercentile vs nanmin/nanmax on (T,H,W) stack."""
    shape = (T, H, W)
    img = np.random.rand(*shape).astype(np.float32)
    img[0, 0, 0] = np.nan  # test nan handling
    size_mb = img.nbytes / 1e6

    print(f"Stack: {shape} = {size_mb:.2f} MB float32")
    print(f"Warmup: {n_warmup}, Timed: {n_runs}")
    print()

    for _ in range(n_warmup):
        _nanminmax(img)
    for _ in range(n_warmup):
        _nanpercentile_1pct(img)

    times_minmax = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        _nanminmax(img)
        times_minmax.append(time.perf_counter() - t0)

    times_pct = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        _nanpercentile_1pct(img)
        times_pct.append(time.perf_counter() - t0)

    mean_minmax = np.mean(times_minmax) * 1000
    mean_pct = np.mean(times_pct) * 1000
    std_minmax = np.std(times_minmax) * 1000
    std_pct = np.std(times_pct) * 1000

    print(f"Min/Max:     {mean_minmax:.2f} +- {std_minmax:.2f} ms")
    print(f"1% pct:      {mean_pct:.2f} +- {std_pct:.2f} ms")
    if mean_minmax > 0:
        ratio = mean_pct / mean_minmax
        print(f"Ratio:       {ratio:.2f}x (percentile vs minmax)")
    print()
    print("Note: On frame scroll, _levels_cache is used -> no recompute.")
    print("Percentile cost only on load, crop, mask, or percentile change.")


def main() -> None:
    print("=== LUT Percentile vs Min/Max Benchmark ===\n")
    for shape in [(50, 512, 512), (200, 512, 512), (1000, 256, 256)]:
        print(f"--- {shape} ---")
        run_benchmark(T=shape[0], H=shape[1], W=shape[2])
        print()


if __name__ == "__main__":
    main()
