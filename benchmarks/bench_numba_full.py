"""Benchmark full Numba optimizations (Pipeline, Sliding Mean, Reductions).

Run: uv run python tests/bench_numba_full.py
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

def _numpy_pipeline_fallback(
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

def _numpy_sliding_mean(
    images: np.ndarray,
    window: int,
    lag: int,
) -> np.ndarray:
    """NumPy fallback for sliding mean (using cumsum)."""
    n = images.shape[0] - (lag + window)
    if n <= 0:
        return np.empty((0, *images.shape[1:]), dtype=np.float32)

    result = np.empty((n, *images.shape[1:]), dtype=np.float32)
    h = images.shape[1]

    for i in range(h):
        sl = images[:, i, ...].astype(np.float32)
        cs = np.cumsum(sl, axis=0)
        upper = cs[lag + window : lag + window + n]
        lower = cs[lag : lag + n]
        result[:, i, ...] = (upper - lower) / window

    return result

def run_benchmark(
    T: int = 100,
    H: int = 256,
    W: int = 256,
    C: int = 1,
    n_warmup: int = 2,
    n_runs: int = 5,
) -> None:
    """Run full Numba vs NumPy benchmark."""
    if not optimized.HAS_NUMBA:
        print("Numba not available. Install numba to run benchmark.")
        return

    shape = (T, H, W, C)
    print(f"Shape: {shape} = {np.prod(shape) * 4 / 1e6:.2f} MB float32")
    print(f"Warmup: {n_warmup} runs, Timed: {n_runs} runs\n")

    # --- 1. Pipeline Operations ---
    print("--- 1. Pipeline Operations (Fused Subtract/Divide) ---")
    img_orig = (np.random.rand(*shape).astype(np.float32) + 1.0)
    sub_ref = np.random.rand(1, H, W, C).astype(np.float32)
    div_ref = (np.random.rand(1, H, W, C).astype(np.float32) + 0.5)
    sub_amt, div_amt = 0.5, 0.8

    # Warmup
    for _ in range(n_warmup):
        img = img_orig.copy()
        optimized.apply_pipeline_fused(img, True, sub_ref, sub_amt, True, div_ref, div_amt)
    for _ in range(n_warmup):
        _numpy_pipeline_fallback(img_orig, True, sub_ref, sub_amt, True, div_ref, div_amt)

    # Numba
    times_numba = []
    for _ in range(n_runs):
        img = img_orig.copy()
        t0 = time.perf_counter()
        optimized.apply_pipeline_fused(img, True, sub_ref, sub_amt, True, div_ref, div_amt)
        times_numba.append(time.perf_counter() - t0)

    # NumPy
    times_numpy = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        _numpy_pipeline_fallback(img_orig, True, sub_ref, sub_amt, True, div_ref, div_amt)
        times_numpy.append(time.perf_counter() - t0)

    _print_results(times_numba, times_numpy)


    # --- 2. Sliding Mean ---
    print("\n--- 2. Sliding Mean ---")
    window, lag = 10, 5

    # Warmup
    for _ in range(n_warmup):
        optimized.sliding_mean_numba(img_orig, window, lag)
    for _ in range(n_warmup):
        _numpy_sliding_mean(img_orig, window, lag)

    # Numba
    times_numba = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        optimized.sliding_mean_numba(img_orig, window, lag)
        times_numba.append(time.perf_counter() - t0)

    # NumPy
    times_numpy = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        _numpy_sliding_mean(img_orig, window, lag)
        times_numpy.append(time.perf_counter() - t0)

    _print_results(times_numba, times_numpy)


    # --- 3. Reductions (Mean, Std, Max, Min) ---
    ops_list = ["mean", "std", "max", "min"]

    for op in ops_list:
        print(f"\n--- 3. Reduction: {op.upper()} ---")

        # Warmup
        for _ in range(n_warmup):
            optimized._reduce_axis0_numba(img_orig, op)

        # Determine numpy equivalent
        def numpy_reduce(x, op_name):
            if op_name == "mean":
                return np.expand_dims(x.mean(axis=0), axis=0)
            elif op_name == "std":
                return np.expand_dims(x.std(axis=0), axis=0)
            elif op_name == "max":
                return np.expand_dims(x.max(axis=0), axis=0)
            elif op_name == "min":
                return np.expand_dims(x.min(axis=0), axis=0)

        for _ in range(n_warmup):
            numpy_reduce(img_orig, op)

        # Numba
        times_numba = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            optimized._reduce_axis0_numba(img_orig, op)
            times_numba.append(time.perf_counter() - t0)

        # NumPy
        times_numpy = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            numpy_reduce(img_orig, op)
            times_numpy.append(time.perf_counter() - t0)

        _print_results(times_numba, times_numpy)


def _print_results(times_numba, times_numpy):
    mean_numba = np.mean(times_numba) * 1000
    mean_numpy = np.mean(times_numpy) * 1000
    std_numba = np.std(times_numba) * 1000
    std_numpy = np.std(times_numpy) * 1000

    print(f"Numba:  {mean_numba:.2f} +- {std_numba:.2f} ms")
    print(f"NumPy:  {mean_numpy:.2f} +- {std_numpy:.2f} ms")
    if mean_numba > 0:
        speedup = mean_numpy / mean_numba
        print(f"Speedup: {speedup:.2f}x (Numba vs NumPy)")
    else:
         print("Speedup: N/A (Numba took 0 ms)")

def main() -> None:
    # Use larger dimensions for more realistic benchmark
    run_benchmark(T=200, H=512, W=512, C=1) # 200MB
    # run_benchmark(T=300, H=800, W=600, C=1) # 580MB
    # run_benchmark(T=500, H=1200, W=900, C=1) # 2.2GB


if __name__ == "__main__":
    main()
