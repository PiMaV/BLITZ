"""Benchmark apply_pipeline_fused (Numba) vs NumPy fallback.

Run: uv run python tests/bench_optimized.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

# Ensure blitz is importable (project root = parent of tests/)
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from blitz.data import optimized


def _numpy_fallback(
    image: np.ndarray,
    do_sub: bool,
    sub_ref: np.ndarray,
    sub_amt: float,
    do_div: bool,
    div_ref: np.ndarray,
    div_amt: float,
) -> np.ndarray:
    """NumPy equivalent of apply_pipeline_fused for benchmarking."""
    img = image.astype(np.float32).copy()
    if do_sub:
        if sub_ref.shape[0] == 1:
            img -= sub_amt * sub_ref
        else:
            img = img[: sub_ref.shape[0]]
            img -= sub_amt * sub_ref
    if do_div:
        denom = div_amt * div_ref + (np.float32(1.0) - div_amt)
        denom = np.where(denom != 0, denom, np.float32(np.nan))
        if div_ref.shape[0] == 1:
            img /= denom
        else:
            img = img[: div_ref.shape[0]]
            img /= denom
        np.nan_to_num(img, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return img


def run_benchmark(
    T: int = 100,
    H: int = 256,
    W: int = 256,
    C: int = 1,
    n_warmup: int = 2,
    n_runs: int = 5,
) -> None:
    """Run Numba vs NumPy benchmark for apply_pipeline_fused."""
    if not optimized.HAS_NUMBA:
        print("Numba not available. Install numba to run benchmark.")
        return

    shape = (T, H, W, C)
    img_orig = (np.random.rand(*shape).astype(np.float32) + 1.0)
    sub_ref = np.random.rand(1, H, W, C).astype(np.float32)
    div_ref = (np.random.rand(1, H, W, C).astype(np.float32) + 0.5)
    sub_amt, div_amt = 0.5, 0.8

    print(f"Shape: {shape} = {np.prod(shape) * 4 / 1e6:.2f} MB float32")
    print(f"Warmup: {n_warmup} runs, Timed: {n_runs} runs")
    print()

    # Warmup
    for _ in range(n_warmup):
        img = img_orig.copy()
        optimized.apply_pipeline_fused(
            img, True, sub_ref, sub_amt, True, div_ref, div_amt
        )
    for _ in range(n_warmup):
        _numpy_fallback(
            img_orig, True, sub_ref, sub_amt, True, div_ref, div_amt
        )

    # Numba
    times_numba = []
    for _ in range(n_runs):
        img = img_orig.copy()
        t0 = time.perf_counter()
        optimized.apply_pipeline_fused(
            img, True, sub_ref, sub_amt, True, div_ref, div_amt
        )
        times_numba.append(time.perf_counter() - t0)

    # NumPy
    times_numpy = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        _numpy_fallback(
            img_orig, True, sub_ref, sub_amt, True, div_ref, div_amt
        )
        times_numpy.append(time.perf_counter() - t0)

    mean_numba = np.mean(times_numba) * 1000
    mean_numpy = np.mean(times_numpy) * 1000
    std_numba = np.std(times_numba) * 1000
    std_numpy = np.std(times_numpy) * 1000

    print(f"Numba:  {mean_numba:.2f} +- {std_numba:.2f} ms")
    print(f"NumPy:  {mean_numpy:.2f} +- {std_numpy:.2f} ms")
    if mean_numpy > 0:
        speedup = mean_numpy / mean_numba
        print(f"Speedup: {speedup:.2f}x (Numba vs NumPy)")


def main() -> None:
    run_benchmark()


if __name__ == "__main__":
    main()
